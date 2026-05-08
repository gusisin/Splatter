from __future__ import annotations

import json
import logging
import math
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import gradio as gr
from PIL import Image

APP_DIR = Path(__file__).resolve().parent
SETTINGS_FILE = APP_DIR / "stills_app_settings.json"
MAX_LOG_CHARS = 250_000
DEFAULT_BASE_DIR = Path.home() / "Pictures" / "Splatter"
GALLERY_CACHE_DIR = APP_DIR / "_stills_gallery_cache"

logger = logging.getLogger("splatter.stills")
logger.propagate = False
if not logger.handlers:
    _handler = logging.StreamHandler(sys.stderr)
    _handler.setFormatter(logging.Formatter("[splatter-stills] %(levelname)s: %(message)s"))
    logger.addHandler(_handler)
logger.setLevel(logging.INFO)


def _console(msg: str, level: str = "INFO") -> None:
    """Always visible in the terminal (Gradio may swallow logging on some setups)."""
    print(f"[splatter-stills] {level}: {msg}", flush=True)
    if level == "ERROR":
        logger.error(msg)
    elif level == "WARNING":
        logger.warning(msg)
    else:
        logger.info(msg)


MAX_GALLERY_FRAMES = 400
GALLERY_THUMB_MAX = 512


def _gallery_items_from_frames(frames: list[Path]) -> list[Any]:
    """Write thumbnails under the repo (allowed by splatter_app launch) and return (path, caption) pairs."""
    items: list[Any] = []
    try:
        if GALLERY_CACHE_DIR.exists():
            shutil.rmtree(GALLERY_CACHE_DIR)
        GALLERY_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        _console(f"Could not reset gallery cache at {GALLERY_CACHE_DIR}: {exc}", "ERROR")
        return items

    shown = frames[:MAX_GALLERY_FRAMES]
    if len(frames) > MAX_GALLERY_FRAMES:
        _console(f"Gallery capped: showing first {MAX_GALLERY_FRAMES} of {len(frames)} frames", "WARNING")

    _resample = getattr(Image, "Resampling", Image).LANCZOS
    for i, p in enumerate(shown):
        rp = p.resolve()
        try:
            if not rp.is_file():
                _console(f"Gallery skip (not a file): {rp}", "WARNING")
                continue
            with Image.open(rp) as im:
                thumb = im.convert("RGB")
                thumb.thumbnail((GALLERY_THUMB_MAX, GALLERY_THUMB_MAX), _resample)
                to_save = thumb.copy()
            safe_name = re.sub(r"[^a-zA-Z0-9._-]+", "_", rp.name) or "frame.png"
            out_path = GALLERY_CACHE_DIR / f"{i:04d}_{safe_name}"
            to_save.save(out_path, format="PNG")
            items.append((str(out_path.resolve()), rp.name))
        except OSError as exc:
            _console(f"Gallery skip {rp}: {exc}", "WARNING")
        except Exception as exc:
            _console(f"Gallery unexpected error for {rp}: {exc}", "ERROR")
            traceback.print_exc()

    _console(f"Gallery cache: wrote {len(items)} PNG(s) under {GALLERY_CACHE_DIR} (source frames on disk: {len(frames)})")
    return items


def _frames_state_default() -> dict[str, list[Any]]:
    """Tracks the current gallery contents and which indices are selected for deletion."""
    return {"frames": [], "thumbs": [], "selected": []}


def _selection_summary(fs: dict[str, Any]) -> str:
    n_total = len(fs.get("frames") or [])
    n_sel = len(fs.get("selected") or [])
    if n_total == 0:
        return "_No frames yet — extract in Step 3, then click thumbnails to mark for deletion._"
    if n_sel == 0:
        return f"**Selected:** 0 / {n_total} _(click a thumbnail to mark for deletion)_"
    return f"**Selected:** {n_sel} / {n_total}"


def _render_gallery_items(fs: dict[str, Any]) -> list[Any]:
    """Build (thumb_path, caption) pairs with a marker on selected items."""
    selected = set(fs.get("selected") or [])
    out: list[Any] = []
    thumbs = fs.get("thumbs") or []
    frames = fs.get("frames") or []
    for i, (thumb, frame) in enumerate(zip(thumbs, frames)):
        name = Path(frame).name
        caption = f"[X] {name}" if i in selected else name
        out.append((thumb, caption))
    return out


def _update_frame_review_ui(state: dict[str, Any]):
    """Rebuild the frames_state, gallery, and selection summary in a follow-up step.

    Includes a short pause so Windows can finish releasing ffmpeg file handles before we read PNGs.
    """
    if not state.get("initialized"):
        empty = _frames_state_default()
        return empty, gr.update(value=[]), gr.update(value=_selection_summary(empty))
    time.sleep(0.15)
    stills_dir = Path(state["stills_dir"])
    splat_name = state["splat_name"]
    frames = sorted(stills_dir.glob(f"{splat_name}-*.png"))
    _console(
        f"_update_frame_review_ui: after 0.15s settle, matched {len(frames)} file(s) under {stills_dir}"
    )
    try:
        thumb_items = _gallery_items_from_frames(frames)
    except Exception as exc:
        _console(f"Gallery build raised (review ui): {exc}", "ERROR")
        traceback.print_exc()
        thumb_items = []

    fs: dict[str, Any] = {
        "frames": [str(p) for p in frames[: len(thumb_items)]],
        "thumbs": [item[0] for item in thumb_items],
        "selected": [],
    }
    return fs, gr.update(value=_render_gallery_items(fs)), gr.update(value=_selection_summary(fs))


def on_gallery_click(fs: dict[str, Any], evt: gr.SelectData):
    fs = fs or _frames_state_default()
    idx = getattr(evt, "index", None)
    if idx is None:
        return fs, gr.update(), gr.update()
    idx = int(idx)
    sel = list(fs.get("selected") or [])
    if idx in sel:
        sel.remove(idx)
    else:
        sel.append(idx)
        sel.sort()
    fs["selected"] = sel
    return fs, gr.update(value=_render_gallery_items(fs)), gr.update(value=_selection_summary(fs))


def select_all_frames(fs: dict[str, Any]):
    fs = fs or _frames_state_default()
    fs["selected"] = list(range(len(fs.get("frames") or [])))
    return fs, gr.update(value=_render_gallery_items(fs)), gr.update(value=_selection_summary(fs))


def clear_selection(fs: dict[str, Any]):
    fs = fs or _frames_state_default()
    fs["selected"] = []
    return fs, gr.update(value=_render_gallery_items(fs)), gr.update(value=_selection_summary(fs))


def invert_selection(fs: dict[str, Any]):
    fs = fs or _frames_state_default()
    n = len(fs.get("frames") or [])
    cur = set(fs.get("selected") or [])
    fs["selected"] = sorted(set(range(n)) - cur)
    return fs, gr.update(value=_render_gallery_items(fs)), gr.update(value=_selection_summary(fs))


def _append_log(current: str, text: str) -> str:
    merged = current + text
    return merged[-MAX_LOG_CHARS:] if len(merged) > MAX_LOG_CHARS else merged


def _load_settings() -> dict[str, Any]:
    defaults = {"base_output_dir": str(DEFAULT_BASE_DIR), "default_fps": 1.0, "default_max_width": 0}
    if not SETTINGS_FILE.exists():
        SETTINGS_FILE.write_text(json.dumps(defaults, indent=2), encoding="utf-8")
        return defaults
    try:
        loaded = json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
        if not isinstance(loaded, dict):
            return defaults
        return {
            "base_output_dir": str(loaded.get("base_output_dir", defaults["base_output_dir"])),
            "default_fps": float(loaded.get("default_fps", defaults["default_fps"])),
            "default_max_width": int(loaded.get("default_max_width", defaults["default_max_width"])),
        }
    except Exception:
        return defaults


def _save_settings(base_output_dir: str, default_fps: float, default_max_width: int) -> None:
    payload = {
        "base_output_dir": str(Path(base_output_dir).expanduser()),
        "default_fps": float(default_fps),
        "default_max_width": int(default_max_width),
    }
    SETTINGS_FILE.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _probe_duration_seconds(path: Path) -> float:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "ffprobe failed.")
    return float(result.stdout.strip())


def _extract_file_paths(file_value: Any) -> list[str]:
    if file_value is None:
        return []
    if isinstance(file_value, str):
        return [file_value]
    if isinstance(file_value, dict):
        for key in ("path", "name"):
            value = file_value.get(key)
            if isinstance(value, str) and value:
                return [value]
        return []
    if isinstance(file_value, list):
        out: list[str] = []
        for entry in file_value:
            out.extend(_extract_file_paths(entry))
        return out
    return []


def _sanitize_splat_name(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]+", "-", name.strip())


def _estimate_frame_count(duration: float, fps: float) -> int:
    if fps <= 0:
        return 0
    return max(0, int(math.floor(duration * fps)) + 1)


def _build_media_table(media: list[dict[str, Any]]) -> list[list[Any]]:
    rows: list[list[Any]] = []
    for m in media:
        path = m["path"]
        rows.append(
            [
                Path(path).name,
                f"{m['duration']:.2f}",
                f"{m['start']:.2f}",
                f"{m['end']:.2f}",
                m["status"],
                path,
            ]
        )
    return rows


def _queue_summary_text(media: list[dict[str, Any]]) -> str:
    n = len([m for m in media if m.get("status") == "ready"])
    if n == 0:
        return "_Queue: empty — drag video(s) above, then click **Add to Queue**._"
    return f"**Queue: {n} video(s) ready.**"


def _add_to_queue_update(upload_value: Any, state: dict[str, Any]):
    """Highlight the Add to Queue button in primary colour when uploads are pending."""
    if not (state or {}).get("initialized"):
        return gr.update(interactive=False, variant="secondary")
    if _extract_file_paths(upload_value):
        return gr.update(interactive=True, variant="primary")
    return gr.update(interactive=False, variant="secondary")


def _estimate_summary(media: list[dict[str, Any]], fps: float) -> str:
    total_duration = sum(float(m["duration"]) for m in media if m["status"] == "ready")
    total_frames = sum(_estimate_frame_count(float(m["duration"]), fps) for m in media if m["status"] == "ready")
    return f"Estimated total duration: **{total_duration:.2f}s** | Estimated stills: **{total_frames}**"


def _state_frame_count(state: dict[str, Any]) -> int:
    if not state.get("initialized"):
        return 0
    stills_dir = Path(str(state.get("stills_dir", "")))
    splat_name = str(state.get("splat_name", ""))
    if not stills_dir.exists() or not splat_name:
        return 0
    return len(list(stills_dir.glob(f"{splat_name}-*.png")))


def _workflow_summary(state: dict[str, Any]) -> str:
    initialized = bool(state.get("initialized"))
    media = state.get("media", [])
    ready_media = len([m for m in media if m.get("status") == "ready"])
    frame_count = _state_frame_count(state)
    colmap_ready = bool(state.get("colmap_prepared", False))
    splat_name = str(state.get("splat_name", ""))
    lines = [
        "### Workflow Status",
        f"- Session: {'Ready (' + splat_name + ')' if initialized else 'Not created'}",
        f"- Media queued: {ready_media}",
        f"- Frames extracted: {frame_count}",
        f"- COLMAP dataset: {'Ready' if colmap_ready else 'Not prepared'}",
    ]
    return "\n".join(lines)


def _next_action_hint(state: dict[str, Any]) -> str:
    initialized = bool(state.get("initialized"))
    media = state.get("media", [])
    ready_media = len([m for m in media if m.get("status") == "ready"])
    frame_count = _state_frame_count(state)
    colmap_ready = bool(state.get("colmap_prepared", False))
    if not initialized:
        return "### Next Action\nCreate a session in Step 1."
    if ready_media == 0:
        return "### Next Action\nAdd one or more videos in Step 2."
    if frame_count == 0:
        return "### Next Action\nRun extraction in Step 3."
    if not colmap_ready:
        return "### Next Action\nReview frames (optional), then build COLMAP dataset in Step 5."
    return "### Next Action\nDataset is ready. Switch to the Train Splat tab."


def _workflow_ui_updates(state: dict[str, Any]):
    initialized = bool(state.get("initialized"))
    ready_media = len([m for m in state.get("media", []) if m.get("status") == "ready"])
    frame_count = _state_frame_count(state)
    return (
        gr.update(value=_workflow_summary(state)),
        gr.update(value=_next_action_hint(state)),
        gr.update(visible=initialized),
        gr.update(visible=initialized),
        gr.update(visible=initialized and (ready_media > 0)),
        gr.update(visible=initialized and (frame_count > 0)),
    )


def _norm_path(p: Path | str | None) -> str:
    if not p:
        return ""
    try:
        return os.path.normcase(os.path.normpath(str(Path(str(p)).expanduser().resolve())))
    except OSError:
        return os.path.normcase(os.path.normpath(str(p)))


def _preview_session_path(name: str, base: str, state: dict[str, Any]):
    """Live preview of the resolved splat directory + Create button gating.

    After a successful Create, the folder exists by design — that must not be shown as a conflict.
    """
    state = state or {}
    raw = (name or "").strip()
    cleaned = _sanitize_splat_name(raw)
    if not cleaned:
        return (
            gr.update(value="_Enter a session name to continue._"),
            gr.update(interactive=False),
        )
    base_path = Path((base or "").strip()).expanduser() if (base or "").strip() else DEFAULT_BASE_DIR
    splat_dir = base_path / cleaned
    sanitized_note = "" if cleaned == raw else f" _(sanitized from `{raw}`)_"

    if state.get("initialized") and str(state.get("splat_name") or "") == cleaned:
        if _norm_path(state.get("splat_dir")) == _norm_path(splat_dir):
            return (
                gr.update(
                    value=(
                        f"**Active session** at `{splat_dir}`{sanitized_note}\n\n"
                        "_Folder created — continue with Step 2._"
                    )
                ),
                gr.update(interactive=False),
            )

    if splat_dir.exists():
        try:
            entries = list(splat_dir.iterdir())
        except OSError:
            entries = []
        empties = "empty folder" if not entries else f"non-empty folder ({len(entries)} item(s))"
        return (
            gr.update(
                value=(
                    f"**Conflict (build-test):** {empties} already exists at `{splat_dir}`. "
                    "Choose a different name or delete that folder."
                )
            ),
            gr.update(interactive=False),
        )
    return (
        gr.update(value=f"Will create: `{splat_dir}`{sanitized_note}"),
        gr.update(interactive=True),
    )


def _extract_inline_error(log_text: str):
    """Surface the most recent [ERROR] line as a visible banner.

    Walks the log from the bottom up: the first [ERROR] wins, but if an
    [INFO] line is seen first the banner is cleared (the error has been
    superseded by a later successful action).
    """
    if not log_text:
        return gr.update(value="", visible=False)
    for line in reversed(log_text.splitlines()):
        stripped = line.strip()
        if not stripped:
            continue
        if "[ERROR]" in stripped:
            msg = stripped.split("[ERROR]", 1)[-1].strip()
            return gr.update(value=f"**Last error:** {msg}", visible=True)
        if stripped.startswith("[INFO]"):
            return gr.update(value="", visible=False)
    return gr.update(value="", visible=False)


def on_fps_changed(fps: float, state: dict[str, Any]):
    if not state.get("initialized"):
        return state, gr.update(value="No media loaded."), gr.update(interactive=False)
    safe_fps = float(fps) if fps is not None else float(state.get("fps", 1.0))
    state["fps"] = safe_fps
    media = state.get("media", [])
    can_generate = len([m for m in media if m.get("status") == "ready"]) > 0 and safe_fps > 0
    return state, gr.update(value=_estimate_summary(media, safe_fps)), gr.update(interactive=can_generate)


def _write_manifest(state: dict[str, Any], fps: float, max_width: int) -> None:
    if not state.get("initialized"):
        return
    splat_dir = Path(state["splat_dir"])
    stills_dir = Path(state["stills_dir"])
    splat_name = state["splat_name"]
    frames = sorted(stills_dir.glob(f"{splat_name}-*.png"))
    payload = {
        "splat_name": splat_name,
        "splat_dir": str(splat_dir),
        "stills_dir": str(stills_dir),
        "settings": {"fps": float(fps), "max_width": int(max_width)},
        "source_media": state.get("media", []),
        "dataset_path": state.get("dataset_path", ""),
        "colmap_prepared": bool(state.get("colmap_prepared", False)),
        "stills_count": len(frames),
        "stills_files": [p.name for p in frames],
    }
    (splat_dir / "manifest.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _run_command(cmd: list[str]) -> tuple[int, str]:
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    output = (result.stdout or "") + (result.stderr or "")
    return result.returncode, output


def _find_colmap_executable() -> str | None:
    candidates = []
    venv_scripts = Path(sys.executable).resolve().parent
    candidates.append(venv_scripts / "colmap.exe")
    candidates.append(venv_scripts / "colmap.cmd")
    candidates.append(venv_scripts / "colmap.bat")
    for c in candidates:
        if c.exists():
            return str(c)
    return shutil.which("colmap")


def _largest_model_dir(sparse_dir: Path) -> Path | None:
    candidates = [p for p in sparse_dir.iterdir() if p.is_dir()]
    if not candidates:
        return None
    best: tuple[int, Path] | None = None
    for c in candidates:
        points_bin = c / "points3D.bin"
        points_txt = c / "points3D.txt"
        score = 0
        if points_bin.exists():
            score = points_bin.stat().st_size
        elif points_txt.exists():
            score = points_txt.stat().st_size
        if best is None or score > best[0]:
            best = (score, c)
    return best[1] if best else None


def prepare_colmap_dataset(state: dict[str, Any], log_text: str, progress=gr.Progress(track_tqdm=False)):
    if not state.get("initialized"):
        return _append_log(log_text, "[ERROR] Initialize splat first.\n"), state, gr.update(value="COLMAP not prepared.")
    colmap_exe = _find_colmap_executable()
    if colmap_exe is None:
        return _append_log(log_text, "[ERROR] COLMAP executable not found in PATH.\n"), state, gr.update(
            value="COLMAP not found in PATH."
        )

    splat_dir = Path(state["splat_dir"])
    stills_dir = Path(state["stills_dir"])
    frames = sorted(stills_dir.glob(f"{state['splat_name']}-*.png"))
    if not frames:
        return _append_log(log_text, "[ERROR] No extracted stills found to build COLMAP dataset.\n"), state, gr.update(
            value="No stills found."
        )

    dataset_dir = splat_dir / "dataset"
    images_dir = dataset_dir / "images"
    sparse_dir = dataset_dir / "sparse"
    sparse0 = sparse_dir / "0"
    db_path = dataset_dir / "database.db"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    if images_dir.exists():
        shutil.rmtree(images_dir)
    images_dir.mkdir(parents=True, exist_ok=True)
    if sparse_dir.exists():
        shutil.rmtree(sparse_dir)
    sparse_dir.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()

    log = _append_log(log_text, f"[INFO] Preparing COLMAP dataset at {dataset_dir}\n")
    progress(0.05, desc="Copying stills")
    for f in frames:
        shutil.copy2(f, images_dir / f.name)
    log = _append_log(log, f"[INFO] Copied {len(frames)} stills into {images_dir}\n")

    commands = [
        (
            "feature_extractor",
            [
                colmap_exe,
                "feature_extractor",
                "--database_path",
                str(db_path),
                "--image_path",
                str(images_dir),
                "--ImageReader.single_camera",
                "1",
                "--FeatureExtraction.use_gpu",
                "1",
            ],
        ),
        (
            "sequential_matcher",
            [
                colmap_exe,
                "sequential_matcher",
                "--database_path",
                str(db_path),
                "--FeatureMatching.use_gpu",
                "1",
                "--SequentialMatching.overlap",
                "20",
                "--SequentialMatching.quadratic_overlap",
                "1",
            ],
        ),
    ]

    for idx, (name, cmd) in enumerate(commands, start=1):
        progress(0.2 + idx * 0.2, desc=f"Running COLMAP {name}")
        log = _append_log(log, f"$ {' '.join(shlex.quote(p) for p in cmd)}\n")
        code, out = _run_command(cmd)
        if out:
            log = _append_log(log, out + "\n")
        if code != 0:
            return _append_log(log, f"[ERROR] COLMAP {name} failed with exit code {code}.\n"), state, gr.update(
                value=f"COLMAP {name} failed."
            )

    mapper_attempts = [
        (
            "mapper_default",
            [
                colmap_exe,
                "mapper",
                "--database_path",
                str(db_path),
                "--image_path",
                str(images_dir),
                "--output_path",
                str(sparse_dir),
            ],
        ),
        (
            "mapper_relaxed",
            [
                colmap_exe,
                "mapper",
                "--database_path",
                str(db_path),
                "--image_path",
                str(images_dir),
                "--output_path",
                str(sparse_dir),
                "--Mapper.min_num_matches",
                "8",
                "--Mapper.init_min_num_inliers",
                "30",
                "--Mapper.init_min_tri_angle",
                "4",
                "--Mapper.init_max_forward_motion",
                "0.99",
                "--Mapper.abs_pose_min_num_inliers",
                "15",
            ],
        ),
    ]

    model_dir = None
    for idx, (name, cmd) in enumerate(mapper_attempts, start=1):
        progress(0.7 + idx * 0.12, desc=f"Running COLMAP {name}")
        log = _append_log(log, f"$ {' '.join(shlex.quote(p) for p in cmd)}\n")
        code, out = _run_command(cmd)
        if out:
            log = _append_log(log, out + "\n")
        model_dir = _largest_model_dir(sparse_dir)
        if model_dir is not None:
            break
        log = _append_log(log, f"[WARN] COLMAP {name} did not produce a sparse model (exit code {code}).\n")

    if model_dir is None:
        return _append_log(log, "[ERROR] COLMAP mapper failed to create any sparse model.\n"), state, gr.update(
            value="No sparse model produced."
        )

    if sparse0.exists():
        shutil.rmtree(sparse0)
    if model_dir != sparse0:
        shutil.copytree(model_dir, sparse0)

    state["dataset_path"] = str(dataset_dir)
    state["colmap_prepared"] = True
    _write_manifest(state, float(state.get("fps", 1.0)), int(state.get("max_width", 0)))
    progress(1.0, desc="COLMAP ready")
    return (
        _append_log(log, f"[INFO] COLMAP dataset ready: {dataset_dir}\n"),
        state,
        gr.update(value=f"COLMAP ready: `{dataset_dir}`"),
    )


def initialize_splat(splat_name: str, base_output_dir: str, fps: float, max_width: int, log_text: str):
    cleaned = _sanitize_splat_name(splat_name)
    empty_fs = _frames_state_default()
    if not cleaned:
        return (
            _append_log(log_text, "[ERROR] Enter a valid splat name.\n"),
            {"initialized": False, "media": []},
            gr.update(value=""),
            gr.update(interactive=False),
            gr.update(interactive=False),
            gr.update(interactive=False),
            gr.update(interactive=False, variant="secondary"),
            gr.update(interactive=False),
            gr.update(value=[]),
            gr.update(value=_queue_summary_text([])),
            gr.update(value="No media loaded."),
            gr.update(interactive=False),
            gr.update(value=[]),
            empty_fs,
        )

    base = Path(base_output_dir).expanduser()
    splat_dir = base / cleaned
    stills_dir = splat_dir / "stills"
    if splat_dir.exists():
        return (
            _append_log(log_text, f"[ERROR] Splat '{cleaned}' already exists: {splat_dir}\n"),
            {"initialized": False, "media": []},
            gr.update(value=""),
            gr.update(interactive=False),
            gr.update(interactive=False),
            gr.update(interactive=False),
            gr.update(interactive=False, variant="secondary"),
            gr.update(interactive=False),
            gr.update(value=[]),
            gr.update(value=_queue_summary_text([])),
            gr.update(value="No media loaded."),
            gr.update(interactive=False),
            gr.update(value=[]),
            empty_fs,
        )

    stills_dir.mkdir(parents=True, exist_ok=True)
    _save_settings(str(base), float(fps), int(max_width))
    state = {
        "initialized": True,
        "splat_name": cleaned,
        "splat_dir": str(splat_dir),
        "stills_dir": str(stills_dir),
        "fps": float(fps),
        "max_width": int(max_width),
        "media": [],
    }
    _write_manifest(state, float(fps), int(max_width))
    log = _append_log(log_text, f"[INFO] Initialized splat '{cleaned}' at {splat_dir}\n")
    return (
        log,
        state,
        gr.update(value=str(stills_dir)),
        gr.update(interactive=True),
        gr.update(interactive=True),
        gr.update(interactive=True),
        gr.update(interactive=False, variant="secondary"),
        gr.update(interactive=False),
        gr.update(value=[]),
        gr.update(value=_queue_summary_text([])),
        gr.update(value=_estimate_summary([], float(fps))),
        gr.update(interactive=True),
        gr.update(value=[]),
        empty_fs,
    )


def add_media(uploaded_files: Any, fps: float, state: dict[str, Any], log_text: str):
    if not state.get("initialized"):
        return (
            _append_log(log_text, "[ERROR] Initialize a unique splat name first.\n"),
            state,
            gr.update(value=[]),
            gr.update(value=_queue_summary_text([])),
            gr.update(value="No media loaded."),
            gr.update(interactive=False),
            gr.update(value=None),
            gr.update(interactive=False, variant="secondary"),
        )
    media = list(state.get("media", []))
    existing = {m["path"] for m in media}
    candidates: list[str] = []
    candidates.extend(_extract_file_paths(uploaded_files))

    log = log_text
    for raw in candidates:
        path = str(Path(raw).expanduser())
        if path in existing:
            continue
        p = Path(path)
        if not p.exists():
            log = _append_log(log, f"[WARN] Skipped missing path: {p}\n")
            continue
        if p.suffix.lower() not in {".mp4", ".mov", ".mkv", ".avi"}:
            log = _append_log(log, f"[WARN] Skipped unsupported media: {p.name}\n")
            continue
        try:
            dur = _probe_duration_seconds(p)
            media.append({"path": str(p), "duration": dur, "start": 0.0, "end": dur, "status": "ready"})
            existing.add(str(p))
            log = _append_log(log, f"[INFO] Added media: {p} ({dur:.2f}s)\n")
        except Exception as exc:
            log = _append_log(log, f"[WARN] Failed probing media '{p}': {exc}\n")

    state["media"] = media
    _write_manifest(state, float(state.get("fps", fps)), int(state.get("max_width", 0)))
    can_generate = len(media) > 0
    return (
        log,
        state,
        gr.update(value=_build_media_table(media)),
        gr.update(value=_queue_summary_text(media)),
        gr.update(value=_estimate_summary(media, float(fps))),
        gr.update(interactive=can_generate),
        gr.update(value=None),
        gr.update(interactive=False, variant="secondary"),
    )


def generate_stills(
    fps: float,
    max_width: int,
    state: dict[str, Any],
    log_text: str,
    progress=gr.Progress(track_tqdm=False),
):
    if not state.get("initialized"):
        return log_text, state, gr.update(value="No media loaded.")
    if shutil.which("ffmpeg") is None:
        _console("ffmpeg not found on PATH", "ERROR")
        return (
            _append_log(log_text, "[ERROR] ffmpeg not found in PATH.\n"),
            state,
            gr.update(value="No media loaded."),
        )
    media = [m for m in state.get("media", []) if m.get("status") == "ready"]
    if not media:
        return (
            _append_log(log_text, "[ERROR] No ready media to process.\n"),
            state,
            gr.update(value="No media loaded."),
        )
    if fps <= 0:
        return (
            _append_log(log_text, "[ERROR] FPS must be greater than 0.\n"),
            state,
            gr.update(value="No media loaded."),
        )
    state["fps"] = float(fps)
    state["max_width"] = int(max_width)

    stills_dir = Path(state["stills_dir"])
    splat_name = state["splat_name"]
    pattern = stills_dir / f"{splat_name}-%06d.png"
    existing = sorted(stills_dir.glob(f"{splat_name}-*.png"))
    next_index = len(existing) + 1
    log = _append_log(log_text, f"[INFO] Starting extraction from {len(media)} media files.\n")

    total = len(media)
    for idx, item in enumerate(media, start=1):
        src = Path(item["path"])
        vf = f"fps={fps}"
        if int(max_width) > 0:
            vf = f"{vf},scale='min({int(max_width)},iw)':-2"
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(src),
            "-vf",
            vf,
            "-start_number",
            str(next_index),
            str(pattern),
        ]
        log = _append_log(log, f"$ {' '.join(shlex.quote(p) for p in cmd)}\n")
        progress((idx - 1) / total, desc=f"Processing media {idx}/{total}")
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            log = _append_log(log, f"[ERROR] ffmpeg failed for {src.name}\n{result.stderr}\n")
            _console(f"ffmpeg failed for {src.name} rc={result.returncode}", "ERROR")
            continue
        if result.stderr:
            log = _append_log(log, result.stderr + "\n")
        produced = _estimate_frame_count(float(item["duration"]), float(fps))
        next_index += produced
        progress(idx / total, desc=f"Completed {idx}/{total}")

    frames = sorted(stills_dir.glob(f"{splat_name}-*.png"))
    frame_paths = [str(p) for p in frames]
    _console(f"generate_stills: glob matched {len(frames)} file(s) under {stills_dir} (UI refresh is next step)")
    _write_manifest(state, float(fps), int(max_width))
    log = _append_log(log, f"[INFO] Extraction complete. Total frames: {len(frame_paths)}\n")
    _console(
        f"Extraction finished splat={splat_name} count={len(frames)} "
        f"first={frame_paths[0] if frame_paths else None}"
    )
    return log, state, gr.update(value=_estimate_summary(media, float(fps)))


def refresh_frames(state: dict[str, Any], log_text: str):
    if not state.get("initialized"):
        return log_text
    stills_dir = Path(state["stills_dir"])
    splat_name = state["splat_name"]
    frames = sorted(stills_dir.glob(f"{splat_name}-*.png"))
    log = _append_log(log_text, f"[INFO] Refreshed frames: {len(frames)}\n")
    _console(f"refresh_frames splat={splat_name} count={len(frames)} dir={stills_dir}")
    return log


def reject_selected_frames(fs: dict[str, Any], state: dict[str, Any], log_text: str):
    """Permanently delete the frames the user has selected in the gallery."""
    if not state.get("initialized"):
        return log_text
    fs = fs or _frames_state_default()
    sel = list(fs.get("selected") or [])
    frames = list(fs.get("frames") or [])
    if not sel:
        return _append_log(log_text, "[INFO] No frames selected for deletion.\n")

    deleted = 0
    log = log_text
    for idx in sel:
        if idx < 0 or idx >= len(frames):
            continue
        src = Path(frames[idx])
        if not src.exists():
            _console(f"Delete skip (missing): {src}", "WARNING")
            continue
        try:
            src.unlink()
            deleted += 1
        except OSError as exc:
            log = _append_log(log, f"[WARN] Failed to delete {src.name}: {exc}\n")
            _console(f"Delete failed {src}: {exc}", "WARNING")

    _write_manifest(state, float(state.get("fps", 1.0)), int(state.get("max_width", 0)))
    log = _append_log(log, f"[INFO] Deleted {deleted} frame(s). Manifest updated.\n")
    _console(f"reject_selected deleted={deleted}")
    return log


settings = _load_settings()
HAS_COLMAP = _find_colmap_executable() is not None

with gr.Blocks(title="Splatter Stills Extractor V2") as demo:
    gr.Markdown("## Splatter Stills Extractor")
    gr.Markdown("Follow the workflow from top to bottom: create session, add media, extract, review, then build COLMAP.")
    state = gr.State({"initialized": False, "media": []})
    with gr.Row():
        with gr.Column(scale=2):
            with gr.Group():
                gr.Markdown("### Step 1 - Create Session")
                with gr.Row():
                    splat_name = gr.Textbox(label="Splat name (must be unique)", placeholder="my_splat")
                    initialize_button = gr.Button("Create Session", variant="primary", interactive=False)
                name_preview = gr.Markdown("_Enter a session name to continue._")
                with gr.Row():
                    base_output_dir = gr.Textbox(label="Base output directory", value=settings["base_output_dir"], scale=3)
                    output_dir_preview = gr.Textbox(label="Stills output directory", value="", interactive=False, scale=2)

            step2_group = gr.Group(visible=False)
            with step2_group:
                gr.Markdown("### Step 2 - Add Media")
                gr.Markdown("_Drag video files into the box below, then click **Add to Queue**._")
                media_upload = gr.File(
                    label="Upload media files",
                    file_count="multiple",
                    file_types=["video"],
                    type="filepath",
                    interactive=False,
                )
                add_media_button = gr.Button("Add to Queue", variant="secondary", interactive=False)
                queue_summary = gr.Markdown(_queue_summary_text([]))
                media_table = gr.Dataframe(
                    headers=["name", "duration_sec", "start_sec", "end_sec", "status", "path"],
                    value=[],
                    interactive=False,
                    wrap=True,
                    label="Source media queue",
                )

            step3_group = gr.Group(visible=False)
            with step3_group:
                gr.Markdown("### Step 3 - Extract Frames")
                with gr.Row():
                    fps = gr.Number(label="Images per second", value=float(settings["default_fps"]), precision=3, interactive=False)
                    max_width = gr.Number(
                        label="Max width (0 = original resolution)",
                        value=int(settings["default_max_width"]),
                        precision=0,
                        interactive=False,
                    )
                generate_button = gr.Button("Extract Frames", variant="primary", interactive=False)
                estimate = gr.Markdown("Add at least one video to continue.")

            step4_group = gr.Group(visible=False)
            with step4_group:
                gr.Markdown("### Step 4 - Review and Curate")
                gr.Markdown("_Click a thumbnail to mark it for deletion. Selected items get an `[X]` in the caption._")
                frames_state = gr.State(_frames_state_default())
                selection_summary = gr.Markdown(_selection_summary(_frames_state_default()))
                with gr.Row():
                    select_all_button = gr.Button("Select All", interactive=False)
                    invert_button = gr.Button("Invert Selection", interactive=False)
                    clear_button = gr.Button("Clear Selection", interactive=False)
                    refresh_frames_button = gr.Button("Refresh Frames", interactive=False)
                frame_gallery = gr.Gallery(
                    label="Extracted frames (click to select)",
                    columns=6,
                    height=360,
                    object_fit="cover",
                    show_label=True,
                    interactive=False,
                )
                reject_button = gr.Button("Delete Selected Frames", variant="stop", interactive=False)

            step5_group = gr.Group(visible=False)
            with step5_group:
                gr.Markdown("### Step 5 - Build COLMAP Dataset")
                prepare_colmap_button = gr.Button("Build COLMAP Dataset", interactive=False)

        with gr.Column(scale=1):
            workflow_status = gr.Markdown(_workflow_summary({"initialized": False, "media": []}))
            next_action = gr.Markdown(_next_action_hint({"initialized": False, "media": []}))

    initial_colmap_msg = (
        "COLMAP status: Ready to run."
        if HAS_COLMAP
        else "COLMAP status: `colmap` not found in PATH. Install COLMAP to enable preparation."
    )
    colmap_status = gr.Markdown(initial_colmap_msg)
    last_error = gr.Markdown("", visible=False)
    with gr.Accordion("Execution log", open=False):
        log_output = gr.Textbox(
            label="Execution Console",
            value="",
            lines=16,
            max_lines=32,
            interactive=False,
        )

    splat_name.change(
        fn=_preview_session_path,
        inputs=[splat_name, base_output_dir, state],
        outputs=[name_preview, initialize_button],
    )
    base_output_dir.change(
        fn=_preview_session_path,
        inputs=[splat_name, base_output_dir, state],
        outputs=[name_preview, initialize_button],
    )

    initialize_button.click(
        fn=initialize_splat,
        inputs=[splat_name, base_output_dir, fps, max_width, log_output],
        outputs=[
            log_output,
            state,
            output_dir_preview,
            fps,
            max_width,
            media_upload,
            add_media_button,
            generate_button,
            media_table,
            queue_summary,
            estimate,
            refresh_frames_button,
            frame_gallery,
            frames_state,
        ],
    ).then(
        fn=lambda: (
            gr.update(interactive=True),
            gr.update(interactive=HAS_COLMAP),
            gr.update(interactive=True),
            gr.update(interactive=True),
            gr.update(interactive=True),
        ),
        outputs=[reject_button, prepare_colmap_button, select_all_button, invert_button, clear_button],
    ).then(
        fn=_workflow_ui_updates,
        inputs=[state],
        outputs=[workflow_status, next_action, step2_group, step3_group, step4_group, step5_group],
    ).then(
        fn=_preview_session_path,
        inputs=[splat_name, base_output_dir, state],
        outputs=[name_preview, initialize_button],
    ).then(
        fn=_extract_inline_error,
        inputs=[log_output],
        outputs=[last_error],
    )

    add_media_button.click(
        fn=add_media,
        inputs=[media_upload, fps, state, log_output],
        outputs=[
            log_output,
            state,
            media_table,
            queue_summary,
            estimate,
            generate_button,
            media_upload,
            add_media_button,
        ],
    ).then(
        fn=_workflow_ui_updates,
        inputs=[state],
        outputs=[workflow_status, next_action, step2_group, step3_group, step4_group, step5_group],
    ).then(
        fn=_extract_inline_error,
        inputs=[log_output],
        outputs=[last_error],
    )

    media_upload.change(
        fn=_add_to_queue_update,
        inputs=[media_upload, state],
        outputs=[add_media_button],
    )

    fps.change(
        fn=on_fps_changed,
        inputs=[fps, state],
        outputs=[state, estimate, generate_button],
    ).then(
        fn=_workflow_ui_updates,
        inputs=[state],
        outputs=[workflow_status, next_action, step2_group, step3_group, step4_group, step5_group],
    )

    generate_button.click(
        fn=generate_stills,
        inputs=[fps, max_width, state, log_output],
        outputs=[log_output, state, estimate],
    ).then(
        fn=_update_frame_review_ui,
        inputs=[state],
        outputs=[frames_state, frame_gallery, selection_summary],
    ).then(
        fn=_workflow_ui_updates,
        inputs=[state],
        outputs=[workflow_status, next_action, step2_group, step3_group, step4_group, step5_group],
    ).then(
        fn=_extract_inline_error,
        inputs=[log_output],
        outputs=[last_error],
    )

    refresh_frames_button.click(
        fn=refresh_frames,
        inputs=[state, log_output],
        outputs=[log_output],
    ).then(
        fn=_update_frame_review_ui,
        inputs=[state],
        outputs=[frames_state, frame_gallery, selection_summary],
    ).then(
        fn=_workflow_ui_updates,
        inputs=[state],
        outputs=[workflow_status, next_action, step2_group, step3_group, step4_group, step5_group],
    ).then(
        fn=_extract_inline_error,
        inputs=[log_output],
        outputs=[last_error],
    )

    reject_button.click(
        fn=reject_selected_frames,
        inputs=[frames_state, state, log_output],
        outputs=[log_output],
    ).then(
        fn=_update_frame_review_ui,
        inputs=[state],
        outputs=[frames_state, frame_gallery, selection_summary],
    ).then(
        fn=_workflow_ui_updates,
        inputs=[state],
        outputs=[workflow_status, next_action, step2_group, step3_group, step4_group, step5_group],
    ).then(
        fn=_extract_inline_error,
        inputs=[log_output],
        outputs=[last_error],
    )

    frame_gallery.select(
        fn=on_gallery_click,
        inputs=[frames_state],
        outputs=[frames_state, frame_gallery, selection_summary],
    )
    select_all_button.click(
        fn=select_all_frames,
        inputs=[frames_state],
        outputs=[frames_state, frame_gallery, selection_summary],
    )
    invert_button.click(
        fn=invert_selection,
        inputs=[frames_state],
        outputs=[frames_state, frame_gallery, selection_summary],
    )
    clear_button.click(
        fn=clear_selection,
        inputs=[frames_state],
        outputs=[frames_state, frame_gallery, selection_summary],
    )
    prepare_colmap_button.click(
        fn=prepare_colmap_dataset,
        inputs=[state, log_output],
        outputs=[log_output, state, colmap_status],
    ).then(
        fn=_workflow_ui_updates,
        inputs=[state],
        outputs=[workflow_status, next_action, step2_group, step3_group, step4_group, step5_group],
    ).then(
        fn=_extract_inline_error,
        inputs=[log_output],
        outputs=[last_error],
    )


if __name__ == "__main__":
    # Allow Gradio to serve generated stills from user-selected output directories.
    demo.queue().launch(allowed_paths=[str(APP_DIR), str(Path.home())])
