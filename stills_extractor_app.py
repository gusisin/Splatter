from __future__ import annotations

import json
import math
import os
import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import gradio as gr

APP_DIR = Path(__file__).resolve().parent
SETTINGS_FILE = APP_DIR / "stills_app_settings.json"
MAX_LOG_CHARS = 250_000
DEFAULT_BASE_DIR = Path.home() / "Pictures" / "Splatter"


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
    return [[m["path"], f"{m['duration']:.2f}", f"{m['start']:.2f}", f"{m['end']:.2f}", m["status"]] for m in media]


def _estimate_summary(media: list[dict[str, Any]], fps: float) -> str:
    total_duration = sum(float(m["duration"]) for m in media if m["status"] == "ready")
    total_frames = sum(_estimate_frame_count(float(m["duration"]), fps) for m in media if m["status"] == "ready")
    return f"Estimated total duration: **{total_duration:.2f}s** | Estimated stills: **{total_frames}**"


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
    rejected_dir = Path(state["rejected_dir"])
    splat_name = state["splat_name"]
    frames = sorted(stills_dir.glob(f"{splat_name}-*.png"))
    rejected = sorted(rejected_dir.glob(f"{splat_name}-*.png"))
    payload = {
        "splat_name": splat_name,
        "splat_dir": str(splat_dir),
        "stills_dir": str(stills_dir),
        "rejected_dir": str(rejected_dir),
        "settings": {"fps": float(fps), "max_width": int(max_width)},
        "source_media": state.get("media", []),
        "dataset_path": state.get("dataset_path", ""),
        "colmap_prepared": bool(state.get("colmap_prepared", False)),
        "stills_count": len(frames),
        "rejected_count": len(rejected),
        "stills_files": [p.name for p in frames],
        "rejected_files": [p.name for p in rejected],
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
    if not cleaned:
        return (
            _append_log(log_text, "[ERROR] Enter a valid splat name.\n"),
            {"initialized": False, "media": []},
            gr.update(value=""),
            gr.update(interactive=False),
            gr.update(interactive=False),
            gr.update(interactive=False, value=""),
            gr.update(interactive=False),
            gr.update(interactive=False),
            gr.update(interactive=False),
            gr.update(value=[]),
            gr.update(value="No media loaded."),
            gr.update(interactive=False),
            gr.update(value=[]),
            gr.update(choices=[], value=[]),
        )

    base = Path(base_output_dir).expanduser()
    splat_dir = base / cleaned
    stills_dir = splat_dir / "stills"
    rejected_dir = splat_dir / "rejected"
    if splat_dir.exists():
        return (
            _append_log(log_text, f"[ERROR] Splat '{cleaned}' already exists: {splat_dir}\n"),
            {"initialized": False, "media": []},
            gr.update(value=""),
            gr.update(interactive=False),
            gr.update(interactive=False),
            gr.update(interactive=False, value=""),
            gr.update(interactive=False),
            gr.update(interactive=False),
            gr.update(interactive=False),
            gr.update(value=[]),
            gr.update(value="No media loaded."),
            gr.update(interactive=False),
            gr.update(value=[]),
            gr.update(choices=[], value=[]),
        )

    stills_dir.mkdir(parents=True, exist_ok=True)
    rejected_dir.mkdir(parents=True, exist_ok=True)
    _save_settings(str(base), float(fps), int(max_width))
    state = {
        "initialized": True,
        "splat_name": cleaned,
        "splat_dir": str(splat_dir),
        "stills_dir": str(stills_dir),
        "rejected_dir": str(rejected_dir),
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
        gr.update(interactive=True, value=""),
        gr.update(interactive=True),
        gr.update(interactive=True),
        gr.update(interactive=False),
        gr.update(value=[]),
        gr.update(value=_estimate_summary([], float(fps))),
        gr.update(interactive=True),
        gr.update(value=[]),
        gr.update(choices=[], value=[]),
    )


def add_media(paths_text: str, uploaded_files: Any, fps: float, state: dict[str, Any], log_text: str):
    if not state.get("initialized"):
        return (
            _append_log(log_text, "[ERROR] Initialize a unique splat name first.\n"),
            state,
            gr.update(value=[]),
            gr.update(value="No media loaded."),
            gr.update(interactive=False),
            gr.update(value=""),
        )
    media = list(state.get("media", []))
    existing = {m["path"] for m in media}
    candidates: list[str] = []
    if paths_text.strip():
        candidates.extend([line.strip() for line in paths_text.splitlines() if line.strip()])
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
        gr.update(value=_estimate_summary(media, float(fps))),
        gr.update(interactive=can_generate),
        gr.update(value=""),
    )


def generate_stills(
    fps: float,
    max_width: int,
    state: dict[str, Any],
    log_text: str,
    progress=gr.Progress(track_tqdm=False),
):
    if not state.get("initialized"):
        return log_text, state, gr.update(value=[]), gr.update(choices=[], value=[]), gr.update(value="No media loaded.")
    if shutil.which("ffmpeg") is None:
        return (
            _append_log(log_text, "[ERROR] ffmpeg not found in PATH.\n"),
            state,
            gr.update(value=[]),
            gr.update(choices=[], value=[]),
            gr.update(value="No media loaded."),
        )
    media = [m for m in state.get("media", []) if m.get("status") == "ready"]
    if not media:
        return (
            _append_log(log_text, "[ERROR] No ready media to process.\n"),
            state,
            gr.update(value=[]),
            gr.update(choices=[], value=[]),
            gr.update(value="No media loaded."),
        )
    if fps <= 0:
        return (
            _append_log(log_text, "[ERROR] FPS must be greater than 0.\n"),
            state,
            gr.update(value=[]),
            gr.update(choices=[], value=[]),
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
            continue
        if result.stderr:
            log = _append_log(log, result.stderr + "\n")
        produced = _estimate_frame_count(float(item["duration"]), float(fps))
        next_index += produced
        progress(idx / total, desc=f"Completed {idx}/{total}")

    frames = sorted(stills_dir.glob(f"{splat_name}-*.png"))
    frame_paths = [str(p) for p in frames]
    choices = [(p.name, str(p)) for p in frames]
    _write_manifest(state, float(fps), int(max_width))
    log = _append_log(log, f"[INFO] Extraction complete. Total frames: {len(frame_paths)}\n")
    return log, state, gr.update(value=frame_paths), gr.update(choices=choices, value=[]), gr.update(
        value=_estimate_summary(media, float(fps))
    )


def refresh_frames(state: dict[str, Any], log_text: str):
    if not state.get("initialized"):
        return log_text, gr.update(value=[]), gr.update(choices=[], value=[])
    stills_dir = Path(state["stills_dir"])
    splat_name = state["splat_name"]
    frames = sorted(stills_dir.glob(f"{splat_name}-*.png"))
    frame_paths = [str(p) for p in frames]
    choices = [(p.name, str(p)) for p in frames]
    log = _append_log(log_text, f"[INFO] Refreshed frames: {len(frame_paths)}\n")
    return log, gr.update(value=frame_paths), gr.update(choices=choices, value=[])


def reject_selected_frames(selected: list[str], state: dict[str, Any], log_text: str):
    if not state.get("initialized"):
        return log_text, gr.update(value=[]), gr.update(choices=[], value=[])
    if not selected:
        return _append_log(log_text, "[INFO] No frames selected for rejection.\n"), gr.update(), gr.update()

    rejected_dir = Path(state["rejected_dir"])
    moved = 0
    log = log_text
    for raw in selected:
        src = Path(raw)
        if not src.exists():
            continue
        dst = rejected_dir / src.name
        try:
            shutil.move(str(src), str(dst))
            moved += 1
        except Exception as exc:
            log = _append_log(log, f"[WARN] Failed to move {src.name}: {exc}\n")

    _write_manifest(state, float(state.get("fps", 1.0)), int(state.get("max_width", 0)))
    log = _append_log(log, f"[INFO] Rejected {moved} frames (moved to {rejected_dir}). Manifest updated.\n")
    return refresh_frames(state, log)


settings = _load_settings()
HAS_COLMAP = _find_colmap_executable() is not None

with gr.Blocks(title="Splatter Stills Extractor V2") as demo:
    gr.Markdown("## Splatter Stills Extractor V2")
    state = gr.State({"initialized": False, "media": []})

    with gr.Row():
        splat_name = gr.Textbox(label="Splat name (must be unique)", placeholder="my_splat")
        initialize_button = gr.Button("Initialize Splat", variant="primary")

    with gr.Row():
        base_output_dir = gr.Textbox(label="Base output directory", value=settings["base_output_dir"], scale=3)
        output_dir_preview = gr.Textbox(label="Stills output directory", value="", interactive=False, scale=2)

    with gr.Row():
        fps = gr.Number(label="Images per second", value=float(settings["default_fps"]), precision=3, interactive=False)
        max_width = gr.Number(
            label="Max width (0 = original resolution)",
            value=int(settings["default_max_width"]),
            precision=0,
            interactive=False,
        )

    with gr.Row():
        media_paths = gr.Textbox(
            label="Media paths (one per line) - preferred",
            placeholder="E:\\video1.mp4\nE:\\video2.mp4",
            lines=4,
            interactive=False,
            scale=2,
        )
        media_upload = gr.File(
            label="Or upload media files",
            file_count="multiple",
            file_types=["video"],
            type="filepath",
            interactive=False,
            scale=1,
        )

    with gr.Row():
        add_media_button = gr.Button("Add Media", interactive=False)
        generate_button = gr.Button("Generate Stills", variant="primary", interactive=False)

    estimate = gr.Markdown("No media loaded.")
    media_table = gr.Dataframe(
        headers=["path", "duration_sec", "start_sec", "end_sec", "status"],
        value=[],
        interactive=False,
        wrap=True,
        label="Source media queue",
    )

    with gr.Row():
        refresh_frames_button = gr.Button("Refresh Frames", interactive=False)
        reject_button = gr.Button("Reject Selected Frames", interactive=False)
        prepare_colmap_button = gr.Button("Prepare COLMAP Dataset", interactive=False)
    initial_colmap_msg = (
        "COLMAP status: Ready to run."
        if HAS_COLMAP
        else "COLMAP status: `colmap` not found in PATH. Install COLMAP to enable preparation."
    )
    colmap_status = gr.Markdown(initial_colmap_msg)
    frame_gallery = gr.Gallery(label="Extracted frames", columns=6, height=320)
    frame_selector = gr.CheckboxGroup(label="Select frames to reject", choices=[], value=[])
    log_output = gr.Code(label="Execution Console", value="", language="shell", lines=16, interactive=False)

    initialize_button.click(
        fn=initialize_splat,
        inputs=[splat_name, base_output_dir, fps, max_width, log_output],
        outputs=[
            log_output,
            state,
            output_dir_preview,
            fps,
            max_width,
            media_paths,
            media_upload,
            add_media_button,
            generate_button,
            media_table,
            estimate,
            refresh_frames_button,
            frame_gallery,
            frame_selector,
        ],
    ).then(
        fn=lambda: (gr.update(interactive=True), gr.update(interactive=HAS_COLMAP)),
        outputs=[reject_button, prepare_colmap_button],
    )

    add_media_button.click(
        fn=add_media,
        inputs=[media_paths, media_upload, fps, state, log_output],
        outputs=[log_output, state, media_table, estimate, generate_button, media_paths],
    )

    fps.change(
        fn=on_fps_changed,
        inputs=[fps, state],
        outputs=[state, estimate, generate_button],
    )

    generate_button.click(
        fn=generate_stills,
        inputs=[fps, max_width, state, log_output],
        outputs=[log_output, state, frame_gallery, frame_selector, estimate],
    )

    refresh_frames_button.click(
        fn=refresh_frames,
        inputs=[state, log_output],
        outputs=[log_output, frame_gallery, frame_selector],
    )

    reject_button.click(
        fn=reject_selected_frames,
        inputs=[frame_selector, state, log_output],
        outputs=[log_output, frame_gallery, frame_selector],
    )
    prepare_colmap_button.click(
        fn=prepare_colmap_dataset,
        inputs=[state, log_output],
        outputs=[log_output, state, colmap_status],
    )


if __name__ == "__main__":
    # Allow Gradio to serve generated stills from user-selected output directories.
    demo.queue().launch(allowed_paths=[str(APP_DIR), str(Path.home())])
