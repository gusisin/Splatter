from __future__ import annotations

import json
import logging
import math
import os
import re
import shlex
import shutil
import struct
import subprocess
import sys
import threading
import time
import traceback
import queue
from pathlib import Path
from typing import Any

import gradio as gr
from PIL import Image

try:
    import plotly.graph_objects as go

    HAS_PLOTLY = True
except ImportError:
    go = None  # type: ignore[assignment,misc]
    HAS_PLOTLY = False

APP_DIR = Path(__file__).resolve().parent
SETTINGS_FILE = APP_DIR / "stills_app_settings.json"
MAX_LOG_CHARS = 250_000
DEFAULT_BASE_DIR = Path.home() / "Pictures" / "Splatter"
GALLERY_CACHE_DIR = APP_DIR / "_stills_gallery_cache"
COLMAP_VIEWER_MAX_POINTS = 8_000
COLMAP_MAPPER_SNAPSHOT_FREQ = 5
COLMAP_VIEWER_REFRESH_SEC = 3.0

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

# Equirectangular source → 6 cubemap pinhole faces for COLMAP (90° FOV each, full sphere).
EQUIRECT_FACE_FOV_DEG = 90
EQUIRECT_CUBE_FACES: tuple[tuple[str, int, int], ...] = (
    ("f", 0, 0),      # front
    ("r", 90, 0),     # right
    ("b", 180, 0),    # back
    ("l", -90, 0),    # left (yaw=270 fails in ffmpeg v360 — use -90)
    ("u", 0, 90),     # up / zenith (ffmpeg v360 pitch sign is opposite naive expectation)
    ("d", 0, -90),    # down / nadir
)
CUBE_FACE_NAMES = {
    "f": "front",
    "r": "right",
    "b": "back",
    "l": "left",
    "u": "up",
    "d": "down",
}
# Full-res v360 faces from 8K equirect are huge; default cap when UI max_width=0.
EQUIRECT_DEFAULT_TILE_MAX_WIDTH = 1920


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


def _equirect_cube_face_jobs() -> list[tuple[str, int, int, str]]:
    return [(face, yaw, pitch, f"_{face}.png") for face, yaw, pitch in EQUIRECT_CUBE_FACES]


def _equirect_faces_per_frame() -> int:
    return len(EQUIRECT_CUBE_FACES)


def _is_equirect_projection(projection: dict[str, Any]) -> bool:
    mode = str(projection.get("mode", ""))
    return mode in ("equirectangular_cubemap", "equirectangular_tiles")


def _default_projection_block(equirect: bool = False, detection: str = "manual") -> dict[str, Any]:
    if not equirect:
        return {"mode": "flat"}
    return {
        "mode": "equirectangular_cubemap",
        "detection": detection,
        "face_fov_deg": EQUIRECT_FACE_FOV_DEG,
        "faces": [face for face, _, _ in EQUIRECT_CUBE_FACES],
        "faces_per_equirect_frame": _equirect_faces_per_frame(),
    }


def _projection_detection_from_media(media: list[dict[str, Any]]) -> str:
    hints: set[str] = set()
    for item in media:
        for h in item.get("projection_hints") or []:
            hints.add(str(h))
    if "spherical_metadata" in hints:
        return "ffprobe_spherical"
    if "aspect_ratio_2_1" in hints:
        return "aspect_ratio_2_1"
    return "manual"


def _probe_video_stream_info(path: Path) -> dict[str, Any]:
    """Probe width/height and heuristics for equirectangular 360 content."""
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_streams",
        "-show_format",
        str(path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    info: dict[str, Any] = {
        "width": 0,
        "height": 0,
        "aspect": 0.0,
        "hints": [],
        "suggestion": "flat",
    }
    if result.returncode != 0:
        return info
    try:
        payload = json.loads(result.stdout or "{}")
    except json.JSONDecodeError:
        return info

    video_stream: dict[str, Any] | None = None
    for stream in payload.get("streams") or []:
        if isinstance(stream, dict) and stream.get("codec_type") == "video":
            video_stream = stream
            break
    if video_stream is None:
        return info

    width = int(video_stream.get("width") or 0)
    height = int(video_stream.get("height") or 0)
    info["width"] = width
    info["height"] = height
    info["aspect"] = (width / height) if height > 0 else 0.0

    hints: list[str] = []
    blob = json.dumps(payload).lower()
    if any(token in blob for token in ("spherical", "equirect", "360", "projection= equirect")):
        hints.append("spherical_metadata")

    side_data = video_stream.get("side_data_list") or []
    if isinstance(side_data, list):
        for entry in side_data:
            if isinstance(entry, dict):
                sd_type = str(entry.get("side_data_type") or entry.get("type") or "").lower()
                if "spherical" in sd_type or "360" in sd_type:
                    hints.append("spherical_metadata")

    if 1.85 <= info["aspect"] <= 2.15:
        hints.append("aspect_ratio_2_1")

    info["hints"] = sorted(set(hints))
    if "spherical_metadata" in info["hints"] or (
        "aspect_ratio_2_1" in info["hints"] and info["aspect"] >= 1.95
    ):
        info["suggestion"] = "equirect"
    return info


def _equirect_hint_markdown(media: list[dict[str, Any]]) -> tuple[bool, str]:
    flagged = [m for m in media if m.get("projection_suggestion") == "equirect"]
    if not flagged:
        return False, ""
    names = ", ".join(Path(m["path"]).name for m in flagged[:4])
    extra = f" (+{len(flagged) - 4} more)" if len(flagged) > 4 else ""
    return True, (
        f"**360 hint:** ffprobe suggests **stitched equirectangular** for "
        f"{len(flagged)} queued file(s) ({names}{extra}). "
        "Enable **Stitched equirectangular (360°) → 6 cubemap faces** in Step 3 before extracting."
    )


def _effective_equirect_max_width(max_width: int) -> int:
    w = int(max_width)
    return w if w > 0 else EQUIRECT_DEFAULT_TILE_MAX_WIDTH


def _normalize_v360_angle(deg: int) -> int:
    """ffmpeg v360 accepts yaw/pitch in [-180, 180]; 270 must be written as -90."""
    d = int(deg) % 360
    if d > 180:
        d -= 360
    return d


def _build_flat_vf(fps: float, max_width: int) -> str:
    vf = f"fps={fps}"
    if int(max_width) > 0:
        vf = f"{vf},scale='min({int(max_width)},iw)':-2"
    return vf


def _build_equirect_tile_vf(fps: float, yaw: int, pitch: int, max_width: int) -> str:
    h_fov = EQUIRECT_FACE_FOV_DEG
    tile_w = _effective_equirect_max_width(max_width)
    yaw = _normalize_v360_angle(yaw)
    pitch = _normalize_v360_angle(pitch)
    vf = (
        f"fps={fps},"
        f"v360=input=equirect:output=rectilinear:yaw={yaw}:pitch={pitch}:roll=0:"
        f"ih_fov=360:iv_fov=180:h_fov={h_fov}:v_fov={h_fov},"
        f"scale='min({tile_w},iw)':-2"
    )
    return vf


def _ffmpeg_extract_to_pattern(
    src: Path,
    pattern: Path,
    vf: str,
    start_sec: float,
    trim_sec: float | None,
    start_number: int,
) -> tuple[int, str]:
    cmd: list[str] = ["ffmpeg", "-y"]
    if start_sec > 0.0:
        cmd.extend(["-ss", f"{start_sec:.3f}"])
    cmd.extend(["-i", str(src)])
    if trim_sec is not None and trim_sec > 0.0:
        cmd.extend(["-t", f"{trim_sec:.3f}"])
    cmd.extend(["-vf", vf, "-an", "-start_number", str(start_number), str(pattern)])
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    output = (result.stdout or "") + (result.stderr or "")
    code = int(result.returncode)
    if code != 0 and code > 255:
        code = int.from_bytes(code.to_bytes(4, "little", signed=False), "little", signed=True)
    return code, output


def _is_gradio_temp_path(path: Path) -> bool:
    lowered = [part.lower() for part in path.parts]
    return "gradio" in lowered or ("temp" in lowered and "appdata" in lowered)


def _is_file_lock_error(exc: BaseException) -> bool:
    msg = str(exc).lower()
    if "being used by another process" in msg or "permission denied" in msg:
        return True
    winerr = getattr(exc, "winerror", None)
    return winerr in (5, 32)


def _windows_shared_read_copy(src: Path, dest: Path) -> bool:
    """Best-effort copy while Gradio still has the temp file open (Windows only)."""
    if os.name != "nt":
        return False
    try:
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.windll.kernel32
        GENERIC_READ = 0x80000000
        FILE_SHARE_READ = 0x00000001
        FILE_SHARE_WRITE = 0x00000002
        FILE_SHARE_DELETE = 0x00000004
        OPEN_EXISTING = 3
        INVALID_HANDLE_VALUE = -1

        handle = kernel32.CreateFileW(
            str(src),
            GENERIC_READ,
            FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
            None,
            OPEN_EXISTING,
            0,
            None,
        )
        if handle == INVALID_HANDLE_VALUE:
            return False

        chunks: list[bytes] = []
        buf = ctypes.create_string_buffer(1024 * 1024)
        bytes_read = wintypes.DWORD(0)
        while True:
            ok = kernel32.ReadFile(handle, buf, len(buf), ctypes.byref(bytes_read), None)
            if not ok:
                break
            n = int(bytes_read.value)
            if n <= 0:
                break
            chunks.append(buf.raw[:n])
        kernel32.CloseHandle(handle)
        dest.write_bytes(b"".join(chunks))
        return dest.is_file() and dest.stat().st_size > 0
    except OSError:
        return False
    except Exception:
        return False


def _copy_file_resilient(src: Path, dest: Path) -> None:
    last_err: Exception | None = None
    for attempt in range(20):
        try:
            shutil.copy2(src, dest)
            return
        except OSError as exc:
            last_err = exc
            if not _is_file_lock_error(exc):
                raise
            time.sleep(0.2 + 0.1 * attempt)

    for attempt in range(10):
        try:
            with src.open("rb") as inf:
                dest.write_bytes(inf.read())
            if dest.is_file() and dest.stat().st_size > 0:
                return
        except OSError as exc:
            last_err = exc
            if not _is_file_lock_error(exc):
                raise
            time.sleep(0.25 + 0.15 * attempt)

    if _windows_shared_read_copy(src, dest):
        return

    raise OSError(f"Could not copy media into session source folder: {last_err}") from last_err


def _stage_media_file(src: Path, splat_dir: Path) -> Path:
    """Copy browser uploads into the session so ffprobe/ffmpeg are not blocked by Gradio temp locks."""
    src = src.expanduser().resolve()
    source_dir = (splat_dir / "source").resolve()
    source_dir.mkdir(parents=True, exist_ok=True)
    try:
        src.relative_to(source_dir)
        return src
    except ValueError:
        pass

    dest = source_dir / src.name
    counter = 1
    while dest.exists():
        dest = source_dir / f"{src.stem}_{counter}{src.suffix}"
        counter += 1

    if src.is_file() and dest.is_file() and src.stat().st_size == dest.stat().st_size:
        return dest

    _copy_file_resilient(src, dest)
    return dest


def _resolve_staged_media_path(raw_path: str, state: dict[str, Any]) -> Path:
    splat_dir = Path(state["splat_dir"])
    p = Path(raw_path).expanduser().resolve()
    staged_map = state.get("staged_sources") or {}
    key = str(p)
    if key in staged_map:
        candidate = Path(staged_map[key]).expanduser()
        if candidate.is_file():
            return candidate.resolve()
    return _stage_media_file(p, splat_dir)


def _local_path_queue_button(local_path: str, upload_value: Any, state: dict[str, Any]):
    if not state.get("initialized"):
        return gr.update(interactive=False, variant="secondary")
    local = (local_path or "").strip().strip('"')
    has_local = bool(local) and Path(local).expanduser().is_file()
    has_upload = bool(_extract_file_paths(upload_value))
    if has_local or has_upload:
        return gr.update(interactive=True, variant="primary")
    return gr.update(interactive=False, variant="secondary")


def _on_media_upload_changed(upload_value: Any, state: dict[str, Any]):
    """Enable Add to Queue when files are selected (staging happens on button click)."""
    if not (state or {}).get("initialized"):
        return gr.update(interactive=False, variant="secondary"), gr.update(value="")
    has_files = bool(_extract_file_paths(upload_value))
    btn = gr.update(interactive=has_files, variant="primary" if has_files else "secondary")
    status = (
        "_Upload ready — click **Add to Queue** to copy into the session `source` folder._"
        if has_files
        else ""
    )
    return btn, gr.update(value=status)


def _probe_media_metadata(path: Path) -> tuple[float, dict[str, Any]]:
    """Probe duration + stream info, retrying when Windows/Gradio briefly locks the file."""
    last_exc: Exception | None = None
    for attempt in range(10):
        try:
            duration = _probe_duration_seconds(path)
            stream_info = _probe_video_stream_info(path)
            return duration, stream_info
        except (RuntimeError, OSError, ValueError) as exc:
            last_exc = exc
            if _is_file_lock_error(exc):
                time.sleep(0.25 + 0.15 * attempt)
                continue
            raise
    raise RuntimeError(str(last_exc) if last_exc else "ffprobe failed")


def _extract_file_paths(file_value: Any) -> list[str]:
    if file_value is None:
        return []
    if isinstance(file_value, str):
        return [file_value]
    if isinstance(file_value, dict):
        for key in ("path", "name", "orig_name"):
            value = file_value.get(key)
            if isinstance(value, str) and value:
                return [value]
        return []
    if isinstance(file_value, (list, tuple)):
        out: list[str] = []
        for entry in file_value:
            out.extend(_extract_file_paths(entry))
        return out
    path_attr = getattr(file_value, "path", None)
    if isinstance(path_attr, str) and path_attr:
        return [path_attr]
    name_attr = getattr(file_value, "name", None)
    if isinstance(name_attr, str) and name_attr:
        return [name_attr]
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


def _collect_media_candidates(uploaded_files: Any, local_path: str = "") -> list[str]:
    candidates: list[str] = []
    seen: set[str] = set()
    for raw in _extract_file_paths(uploaded_files):
        path = str(Path(raw).expanduser())
        if path not in seen:
            seen.add(path)
            candidates.append(path)
    local = (local_path or "").strip().strip('"')
    if local:
        path = str(Path(local).expanduser())
        if path not in seen:
            candidates.append(path)
    return candidates


def _trimmed_duration(m: dict[str, Any]) -> float:
    """Effective duration after applying user trim (start/end)."""
    duration = float(m.get("duration", 0.0))
    start = max(0.0, float(m.get("start", 0.0)))
    end = float(m.get("end", duration))
    end = min(end, duration)
    return max(0.0, end - start)


def _load_calibration() -> dict[str, float]:
    """Read persisted extraction calibration (bytes/s per frame) from settings."""
    try:
        loaded = json.loads(SETTINGS_FILE.read_text(encoding="utf-8")) if SETTINGS_FILE.exists() else {}
    except Exception:
        loaded = {}
    cal = loaded.get("calibration", {}) if isinstance(loaded, dict) else {}
    return {
        "bytes_per_frame": float(cal.get("bytes_per_frame", 0.0)) if isinstance(cal, dict) else 0.0,
        "seconds_per_frame": float(cal.get("seconds_per_frame", 0.0)) if isinstance(cal, dict) else 0.0,
    }


def _save_calibration(bytes_per_frame: float, seconds_per_frame: float) -> None:
    """Merge calibration into stills_app_settings.json without dropping other keys."""
    try:
        loaded = json.loads(SETTINGS_FILE.read_text(encoding="utf-8")) if SETTINGS_FILE.exists() else {}
    except Exception:
        loaded = {}
    if not isinstance(loaded, dict):
        loaded = {}
    loaded["calibration"] = {
        "bytes_per_frame": float(bytes_per_frame),
        "seconds_per_frame": float(seconds_per_frame),
    }
    SETTINGS_FILE.write_text(json.dumps(loaded, indent=2), encoding="utf-8")


def _heuristic_bytes_per_frame(max_width: int) -> float:
    """Rough PNG-on-disk size estimate before we have real calibration."""
    if int(max_width) > 0:
        w = int(max_width)
    else:
        w = 1920
    h = max(1, int(round(w * 9 / 16)))
    return float(w * h) * 1.0


def _format_size(num_bytes: float) -> str:
    if num_bytes < 1024:
        return f"{num_bytes:.0f} B"
    units = ["KB", "MB", "GB", "TB"]
    val = num_bytes / 1024.0
    for u in units:
        if val < 1024 or u == units[-1]:
            return f"{val:.1f} {u}"
        val /= 1024.0
    return f"{val:.1f} {units[-1]}"


def _format_seconds(secs: float) -> str:
    if secs < 1:
        return "<1 s"
    if secs < 60:
        return f"~{secs:.0f} s"
    mins = secs / 60.0
    if mins < 60:
        return f"~{mins:.1f} min"
    hours = mins / 60.0
    return f"~{hours:.2f} h"


def _estimate_summary(
    media: list[dict[str, Any]], fps: float, max_width: int = 0, equirect_mode: bool = False
) -> str:
    ready = [m for m in media if m.get("status") == "ready"]
    total_duration = sum(_trimmed_duration(m) for m in ready)
    samples = sum(_estimate_frame_count(_trimmed_duration(m), fps) for m in ready)
    tiles = _equirect_faces_per_frame() if equirect_mode else 1
    total_frames = samples * tiles
    if total_frames <= 0:
        return f"Estimated trimmed duration: **{total_duration:.2f}s** | Estimated stills: **0**"

    cal = _load_calibration()
    est_width = _effective_equirect_max_width(max_width) if equirect_mode else int(max_width)
    bpf = cal["bytes_per_frame"] if cal["bytes_per_frame"] > 0 else _heuristic_bytes_per_frame(est_width)
    spf = cal["seconds_per_frame"]
    size_str = _format_size(bpf * total_frames)
    if spf > 0:
        time_str = _format_seconds(spf * total_frames)
        time_block = f" · {time_str} extraction"
    else:
        time_block = ""
    cal_tag = " _(calibrated)_" if cal["bytes_per_frame"] > 0 or cal["seconds_per_frame"] > 0 else " _(rough)_"
    tile_note = (
        f" ({samples} equirect sample(s) × **{tiles}** cubemap faces @ {EQUIRECT_FACE_FOV_DEG}° FOV)"
        if equirect_mode and samples > 0
        else ""
    )
    return (
        f"Estimated trimmed duration: **{total_duration:.2f}s** | "
        f"≈ **{total_frames}** stills{tile_note} · ~**{size_str}**{time_block}{cal_tag}"
    )


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
    projection = state.get("projection") or {"mode": "flat"}
    proj_label = "flat video"
    if _is_equirect_projection(projection):
        faces = projection.get(
            "faces_per_equirect_frame",
            projection.get("tiles_per_equirect_frame", _equirect_faces_per_frame()),
        )
        proj_label = f"360 equirect → {faces} cubemap faces/sample"
    lines = [
        "### Workflow Status",
        f"- Session: {'Ready (' + splat_name + ')' if initialized else 'Not created'}",
        f"- Source projection: {proj_label}",
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
        return "### Next Action\nReview frames (optional), then open the **Build COLMAP Dataset** tab."
    return "### Next Action\nDataset is ready. Switch to the **Train Splat** tab."


def _workflow_ui_updates(state: dict[str, Any]):
    initialized = bool(state.get("initialized"))
    ready_media = len([m for m in state.get("media", []) if m.get("status") == "ready"])
    frame_count = _state_frame_count(state)
    return (
        gr.update(value=_workflow_summary(state)),
        gr.update(value=_next_action_hint(state)),
        gr.update(visible=initialized),
        gr.update(visible=initialized),
        gr.update(visible=initialized and (frame_count > 0)),
    )


def _discover_splat_sessions(base_dir: str) -> list[str]:
    base = Path(base_dir).expanduser()
    if not base.exists():
        return []
    return sorted([p.name for p in base.iterdir() if p.is_dir() and (p / "manifest.json").exists()])


def _sparse0_for_state(state: dict[str, Any]) -> Path | None:
    if not state.get("initialized"):
        return None
    dataset = Path(str(state.get("dataset_path") or "")).expanduser()
    if dataset.is_dir():
        candidate = dataset / "sparse" / "0"
        if candidate.is_dir():
            return candidate
    splat_dir = Path(str(state.get("splat_dir") or "")).expanduser()
    candidate = splat_dir / "dataset" / "sparse" / "0"
    return candidate if candidate.is_dir() else None


def _read_colmap_points3d_xyz_rgb(model_dir: Path) -> tuple[list[tuple[float, float, float]], list[tuple[int, int, int]]] | None:
    """Read XYZ + RGB from COLMAP points3D.bin (preferred) or points3D.txt."""
    bin_path = model_dir / "points3D.bin"
    if bin_path.is_file():
        try:
            data = bin_path.read_bytes()
        except OSError:
            return None
        if len(data) < 8:
            return None
        offset = 0
        (num_points,) = struct.unpack_from("<Q", data, offset)
        offset += 8
        xyz: list[tuple[float, float, float]] = []
        rgb: list[tuple[int, int, int]] = []
        for _ in range(num_points):
            if offset + 8 + 24 + 3 + 8 + 8 > len(data):
                break
            offset += 8  # point3D_id
            x, y, z = struct.unpack_from("<ddd", data, offset)
            offset += 24
            r, g, b = struct.unpack_from("<BBB", data, offset)
            offset += 3
            offset += 8  # error (double)
            (track_len,) = struct.unpack_from("<Q", data, offset)
            offset += 8
            track_bytes = int(track_len) * 8
            if offset + track_bytes > len(data):
                break
            offset += track_bytes
            xyz.append((float(x), float(y), float(z)))
            rgb.append((int(r), int(g), int(b)))
        if xyz:
            return xyz, rgb
        return None

    txt_path = model_dir / "points3D.txt"
    if not txt_path.is_file():
        return None
    xyz = []
    rgb = []
    try:
        with txt_path.open("r", encoding="utf-8", errors="replace") as f:
            for line in f:
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                parts = stripped.split()
                if len(parts) < 8:
                    continue
                x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                r, g, b = int(parts[4]), int(parts[5]), int(parts[6])
                xyz.append((x, y, z))
                rgb.append((r, g, b))
    except OSError:
        return None
    return (xyz, rgb) if xyz else None


def _subsample_colmap_points(
    xyz: list[tuple[float, float, float]],
    rgb: list[tuple[int, int, int]],
    max_points: int,
) -> tuple[list[tuple[float, float, float]], list[tuple[int, int, int]]]:
    if len(xyz) <= max_points:
        return xyz, rgb
    stride = max(1, len(xyz) // max_points)
    indices = list(range(0, len(xyz), stride))[:max_points]
    return [xyz[i] for i in indices], [rgb[i] for i in indices]


def _build_colmap_plotly_figure(
    model_dir: Path,
    *,
    max_points: int = COLMAP_VIEWER_MAX_POINTS,
) -> Any | None:
    """Interactive 3D scatter for COLMAP points (reliable in-browser preview)."""
    if not HAS_PLOTLY:
        return None
    parsed = _read_colmap_points3d_xyz_rgb(model_dir)
    if not parsed:
        return None
    xyz, rgb = parsed
    xyz, rgb = _subsample_colmap_points(xyz, rgb, max_points)
    if not xyz:
        return None

    xs = [p[0] for p in xyz]
    ys = [p[1] for p in xyz]
    zs = [p[2] for p in xyz]
    colors = [f"rgb({int(r)},{int(g)},{int(b)})" for r, g, b in rgb]

    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=xs,
                y=ys,
                z=zs,
                mode="markers",
                marker=dict(size=1.8, color=colors, opacity=0.92),
                hovertemplate="x=%{x:.2f}<br>y=%{y:.2f}<br>z=%{z:.2f}<extra></extra>",
            )
        ]
    )
    axis_style = dict(
        title="",
        showgrid=False,
        zeroline=False,
        showticklabels=False,
        showline=False,
        showbackground=False,
        ticks="",
    )
    fig.update_layout(
        template=None,
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor="#000000",
        plot_bgcolor="#000000",
        height=560,
        showlegend=False,
        scene=dict(
            bgcolor="#000000",
            xaxis=axis_style,
            yaxis=axis_style,
            zaxis=axis_style,
            aspectmode="data",
        ),
    )
    _console(f"Viewer plot: {len(xyz)} points from {model_dir}")
    return fig


def _viewer_plot_update_for_sparse(model_dir: Path | None) -> Any:
    hidden = gr.update(value=None, visible=False)
    if model_dir is None:
        return hidden
    if not HAS_PLOTLY:
        _console("plotly not installed — COLMAP 3D preview disabled (pip install plotly)", "WARNING")
        return hidden
    figure = _build_colmap_plotly_figure(model_dir)
    if figure is None:
        return hidden
    return gr.update(value=figure, visible=True)


def _snapshot_has_points(model_dir: Path) -> bool:
    return (model_dir / "points3D.bin").is_file() or (model_dir / "points3D.txt").is_file()


def _latest_snapshot_model_dir(snapshot_root: Path) -> Path | None:
    """COLMAP writes snapshots to timestamped subfolders under snapshot_path."""
    if not snapshot_root.is_dir():
        return None
    best: tuple[float, Path] | None = None
    for entry in snapshot_root.iterdir():
        if not entry.is_dir() or not _snapshot_has_points(entry):
            continue
        mtime = entry.stat().st_mtime
        if best is None or mtime > best[0]:
            best = (mtime, entry)
    return best[1] if best else None


def _run_mapper_with_snapshots(
    cmd: list[str],
    *,
    snapshot_dir: Path,
    snapshot_freq: int,
    min_refresh_sec: float = COLMAP_VIEWER_REFRESH_SEC,
) -> Any:
    """Run COLMAP mapper; yield (exit_code, output, viewer_update | None)."""
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    full_cmd = [
        *cmd,
        "--Mapper.snapshot_path",
        str(snapshot_dir),
        "--Mapper.snapshot_frames_freq",
        str(snapshot_freq),
    ]
    proc = subprocess.Popen(
        full_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    events: queue.Queue[tuple[str, Any]] = queue.Queue()
    output_chunks: list[str] = []

    def _stdout_reader() -> None:
        assert proc.stdout is not None
        try:
            for line in proc.stdout:
                events.put(("line", line))
        finally:
            events.put(("done", proc.wait()))

    def _snapshot_poller() -> None:
        last_mtime = 0.0
        while proc.poll() is None:
            time.sleep(0.5)
            latest = _latest_snapshot_model_dir(snapshot_dir)
            if latest is None:
                continue
            mtime = latest.stat().st_mtime
            if mtime > last_mtime:
                last_mtime = mtime
                events.put(("snapshot", latest))

    threading.Thread(target=_stdout_reader, daemon=True).start()
    threading.Thread(target=_snapshot_poller, daemon=True).start()

    last_preview_at = 0.0
    last_snap_path: Path | None = None
    viewer_update: Any | None = None
    code: int | None = None

    while code is None:
        try:
            kind, payload = events.get(timeout=0.5)
        except queue.Empty:
            continue

        if kind == "line":
            output_chunks.append(str(payload))
            continue

        if kind == "snapshot":
            snap_model = payload
            now = time.time()
            if snap_model != last_snap_path or (now - last_preview_at) >= min_refresh_sec:
                last_snap_path = snap_model
                last_preview_at = now
                viewer_update = _viewer_plot_update_for_sparse(snap_model)
                yield None, "".join(output_chunks), viewer_update
                output_chunks = []
            continue

        if kind == "done":
            code = int(payload)
            break

    tail = "".join(output_chunks)
    if viewer_update is None:
        latest = _latest_snapshot_model_dir(snapshot_dir)
        if latest is not None:
            viewer_update = _viewer_plot_update_for_sparse(latest)
    yield code, tail, viewer_update


def refresh_colmap_sessions(base_dir: str):
    choices = _discover_splat_sessions(base_dir)
    first = choices[0] if choices else None
    msg = f"Found **{len(choices)}** session(s) under `{Path(base_dir).expanduser()}`."
    return gr.update(choices=choices, value=first), msg


def load_colmap_session(base_dir: str, splat_name: str | None, log_text: str):
    hidden_viewer = gr.update(value=None, visible=False)
    hidden_card = gr.update(visible=False)
    if not splat_name:
        return (
            _append_log(log_text, "[INFO] Select a session to load.\n"),
            {"initialized": False, "media": []},
            gr.update(value="No session loaded."),
            gr.update(value="_Select a session and click **Load Session**._"),
            hidden_viewer,
            hidden_card,
            gr.update(interactive=False),
        )
    manifest_path = Path(base_dir).expanduser() / splat_name / "manifest.json"
    if not manifest_path.is_file():
        return (
            _append_log(log_text, f"[ERROR] Manifest not found: {manifest_path}\n"),
            {"initialized": False, "media": []},
            gr.update(value="Manifest missing."),
            gr.update(value="_Could not load session._"),
            hidden_viewer,
            hidden_card,
            gr.update(interactive=False),
        )
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return (
            _append_log(log_text, f"[ERROR] Failed to read manifest: {exc}\n"),
            {"initialized": False, "media": []},
            gr.update(value="Manifest parse error."),
            gr.update(value="_Could not load session._"),
            hidden_viewer,
            hidden_card,
            gr.update(interactive=False),
        )

    settings_block = manifest.get("settings") if isinstance(manifest.get("settings"), dict) else {}
    projection = manifest.get("projection") if isinstance(manifest.get("projection"), dict) else {"mode": "flat"}
    equirect_mode = bool(
        settings_block.get(
            "equirect_mode",
            _is_equirect_projection(projection if isinstance(projection, dict) else {}),
        )
    )
    state: dict[str, Any] = {
        "initialized": True,
        "splat_name": str(manifest.get("splat_name", splat_name)),
        "splat_dir": str(manifest.get("splat_dir", manifest_path.parent)),
        "stills_dir": str(manifest.get("stills_dir", manifest_path.parent / "stills")),
        "fps": float(settings_block.get("fps", 1.0)),
        "max_width": int(settings_block.get("max_width", 0)),
        "equirect_mode": equirect_mode,
        "projection": projection,
        "media": manifest.get("source_media", []) if isinstance(manifest.get("source_media"), list) else [],
        "dataset_path": str(manifest.get("dataset_path", "")),
        "colmap_prepared": bool(manifest.get("colmap_prepared", False)),
    }
    frame_count = _state_frame_count(state)
    sparse0 = _sparse0_for_state(state)
    colmap_ready = bool(state.get("colmap_prepared")) and sparse0 is not None
    summary = _format_sparse_summary(sparse0, stills_count=frame_count) if colmap_ready else (
        f"_Session **{state['splat_name']}** loaded — **{frame_count}** still(s) on disk. "
        "Click **Build COLMAP Dataset** when ready._"
    )
    status = (
        f"Loaded `{state['splat_name']}` — COLMAP {'ready' if colmap_ready else 'not prepared yet'}."
    )
    log = _append_log(log_text, f"[INFO] Loaded session '{state['splat_name']}' from {manifest_path}\n")
    viewer_update = _viewer_plot_update_for_sparse(sparse0) if colmap_ready else hidden_viewer
    card_update = gr.update(visible=colmap_ready)
    can_build = frame_count > 0 and HAS_COLMAP
    return (
        log,
        state,
        gr.update(value=status),
        gr.update(value=summary),
        viewer_update,
        card_update,
        gr.update(interactive=can_build),
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
    max_w = int(state.get("max_width", 0))
    equirect = bool(state.get("equirect_mode", False))
    can_generate = len([m for m in media if m.get("status") == "ready"]) > 0 and safe_fps > 0
    return (
        state,
        gr.update(value=_estimate_summary(media, safe_fps, max_w, equirect)),
        gr.update(interactive=can_generate),
    )


def on_max_width_changed(max_width: int, state: dict[str, Any]):
    if not state.get("initialized"):
        return state, gr.update()
    safe_w = int(max_width) if max_width is not None else int(state.get("max_width", 0))
    state["max_width"] = safe_w
    media = state.get("media", [])
    fps = float(state.get("fps", 1.0))
    equirect = bool(state.get("equirect_mode", False))
    return state, gr.update(value=_estimate_summary(media, fps, safe_w, equirect))


def on_equirect_mode_changed(equirect_mode: bool, fps: float, max_width: int, state: dict[str, Any]):
    if not state.get("initialized"):
        return state, gr.update(value="No media loaded.")
    enabled = bool(equirect_mode)
    state["equirect_mode"] = enabled
    state["projection"] = _default_projection_block(enabled)
    media = state.get("media", [])
    safe_fps = float(fps) if fps is not None else float(state.get("fps", 1.0))
    safe_w = int(max_width) if max_width is not None else int(state.get("max_width", 0))
    return state, gr.update(value=_estimate_summary(media, safe_fps, safe_w, enabled))


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
        "settings": {
            "fps": float(fps),
            "max_width": int(max_width),
            "equirect_mode": bool(state.get("equirect_mode", False)),
        },
        "projection": state.get("projection") or _default_projection_block(False),
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


CUBEMAP_FACE_ORDER = "frblud"


def _parse_cubemap_still_name(name: str) -> tuple[str, int] | None:
    m = re.match(r".*-(\d+)_([frblud])\.png$", name, re.I)
    if not m:
        return None
    return m.group(2).lower(), int(m.group(1))


def _parse_colmap_export_name(name: str) -> tuple[int, str] | None:
    """Parse time-major COLMAP export names like 000042_f.png."""
    m = re.match(r"^(\d+)_([frblud])\.png$", name, re.I)
    if not m:
        return None
    return int(m.group(1)), m.group(2).lower()


def _colmap_still_sort_key(path: Path) -> tuple[int, int, str]:
    """Time-major: all cube faces for sample N, then sample N+1 (matches filesystem sort)."""
    parsed = _parse_cubemap_still_name(path.name)
    if parsed:
        face, idx = parsed
        face_rank = CUBEMAP_FACE_ORDER.index(face) if face in CUBEMAP_FACE_ORDER else 99
        return (idx, face_rank, path.name)
    return (99, 99, path.name)


def _colmap_export_filename(path: Path) -> str:
    """Rename cubemap stills for COLMAP: 000001_f.png keeps six faces per timestamp adjacent."""
    parsed = _parse_cubemap_still_name(path.name)
    if parsed:
        face, idx = parsed
        return f"{idx:06d}_{face}.png"
    return path.name


def _sparse_model_image_names(model_dir: Path, colmap_exe: str) -> list[str]:
    """Read image filenames from a sparse model via a temporary TXT export."""
    import tempfile

    txt_dir = Path(tempfile.mkdtemp(prefix="splatter_colmap_txt_"))
    try:
        code, _ = _run_command(
            [
                colmap_exe,
                "model_converter",
                "--input_path",
                str(model_dir),
                "--output_path",
                str(txt_dir),
                "--output_type",
                "TXT",
            ]
        )
        if code != 0:
            return []
        images_txt = txt_dir / "images.txt"
        if not images_txt.is_file():
            return []
        names: list[str] = []
        with images_txt.open("r", encoding="utf-8", errors="replace") as f:
            for line in f:
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                parts = stripped.split()
                if len(parts) >= 10 and parts[-1].lower().endswith(".png"):
                    names.append(parts[-1])
        return names
    finally:
        shutil.rmtree(txt_dir, ignore_errors=True)


def _registered_face_summary(model_dir: Path, colmap_exe: str) -> str:
    names = _sparse_model_image_names(model_dir, colmap_exe)
    if not names:
        return ""
    counts: dict[str, int] = {face: 0 for face in CUBEMAP_FACE_ORDER}
    for name in names:
        parsed = _parse_colmap_export_name(name)
        if parsed is None:
            legacy = _parse_cubemap_still_name(name)
            if legacy:
                face = legacy[0]
            else:
                face = "?"
        else:
            face = parsed[1]
        counts[face] = counts.get(face, 0) + 1
    parts = [f"{face}={counts.get(face, 0)}" for face in CUBEMAP_FACE_ORDER if counts.get(face, 0)]
    return ", ".join(parts) if parts else "unknown naming"


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


def _read_uint64_count(path: Path) -> int | None:
    """COLMAP .bin files start with a uint64 element count. Read just that."""
    try:
        with path.open("rb") as f:
            head = f.read(8)
    except OSError:
        return None
    if len(head) != 8:
        return None
    return int.from_bytes(head, byteorder="little", signed=False)


def _count_text_records(path: Path) -> int | None:
    """COLMAP cameras/points3D .txt files have one record per non-comment line; images.txt has 2 lines/record."""
    try:
        with path.open("r", encoding="utf-8", errors="replace") as f:
            lines = [ln.strip() for ln in f if ln.strip() and not ln.startswith("#")]
    except OSError:
        return None
    return len(lines)


def _read_sparse_stats(model_dir: Path) -> dict[str, int | None]:
    """Return registered image count, 3D point count, and camera count from a COLMAP sparse model."""
    stats: dict[str, int | None] = {"images": None, "points": None, "cameras": None}

    images_bin = model_dir / "images.bin"
    points_bin = model_dir / "points3D.bin"
    cameras_bin = model_dir / "cameras.bin"
    if images_bin.exists():
        stats["images"] = _read_uint64_count(images_bin)
    elif (model_dir / "images.txt").exists():
        n = _count_text_records(model_dir / "images.txt")
        stats["images"] = (n // 2) if isinstance(n, int) else None  # images.txt has 2 lines per image
    if points_bin.exists():
        stats["points"] = _read_uint64_count(points_bin)
    elif (model_dir / "points3D.txt").exists():
        stats["points"] = _count_text_records(model_dir / "points3D.txt")
    if cameras_bin.exists():
        stats["cameras"] = _read_uint64_count(cameras_bin)
    elif (model_dir / "cameras.txt").exists():
        stats["cameras"] = _count_text_records(model_dir / "cameras.txt")
    return stats


def _format_sparse_summary(model_dir: Path | None, stills_count: int = 0) -> str:
    if model_dir is None or not model_dir.exists():
        return "_No COLMAP sparse model yet — click **Build COLMAP Dataset**._"
    stats = _read_sparse_stats(model_dir)
    images = stats["images"]
    points = stats["points"]
    cameras = stats["cameras"]

    def fmt(n: int | None) -> str:
        return f"{n:,}" if isinstance(n, int) else "?"

    registered = fmt(images)
    coverage = ""
    if isinstance(images, int) and stills_count > 0:
        pct = (images / stills_count) * 100.0
        coverage = f" / {stills_count:,} stills ({pct:.0f}%)"
    lines = [
        "**COLMAP sparse model**",
        f"- Registered images: **{registered}**{coverage}",
        f"- 3D points: **{fmt(points)}**",
        f"- Cameras: **{fmt(cameras)}**",
        f"- Model path: `{model_dir}`",
    ]
    return "\n".join(lines)


def _largest_model_dir(sparse_dir: Path) -> Path | None:
    if not sparse_dir.exists():
        return None
    best: tuple[int, Path] | None = None
    for c in sparse_dir.iterdir():
        if not c.is_dir():
            continue
        points_bin = c / "points3D.bin"
        points_txt = c / "points3D.txt"
        if points_bin.exists():
            score = points_bin.stat().st_size
        elif points_txt.exists():
            score = points_txt.stat().st_size
        else:
            # A subfolder with no points file is not a real sparse model;
            # COLMAP sometimes leaves these behind on partial/failed runs.
            continue
        if best is None or score > best[0]:
            best = (score, c)
    return best[1] if best else None


def _generate_downscaled_images(
    images_dir: Path, dataset_dir: Path, factors: tuple[int, ...] = (2, 4, 8)
) -> tuple[dict[int, int], str]:
    """Create images_<N>/ siblings of images_dir with PIL-downscaled copies.

    3DGS / 3DGRUT COLMAP loaders use dataset.downsample_factor=N by looking
    for a sibling folder named `images_N`. Without these, training fails with
    'Image not found. Cannot determine dimensions...'. We pre-generate the
    common set (2, 4, 8) once after COLMAP succeeds so the downsample knob
    in the Train tab works out of the box.

    Returns (counts_per_factor, log_text). counts_per_factor maps factor -> images written.
    """
    counts: dict[int, int] = {}
    log_chunks: list[str] = []
    source_images = sorted(images_dir.glob("*.png")) + sorted(images_dir.glob("*.jpg"))
    if not source_images:
        return counts, ""
    for factor in factors:
        if factor <= 1:
            continue
        target_dir = dataset_dir / f"images_{factor}"
        if target_dir.exists():
            shutil.rmtree(target_dir)
        target_dir.mkdir(parents=True, exist_ok=True)
        written = 0
        for src in source_images:
            try:
                with Image.open(src) as im:
                    new_w = max(1, im.width // factor)
                    new_h = max(1, im.height // factor)
                    resized = im.resize((new_w, new_h), Image.LANCZOS)
                    resized.save(target_dir / src.name)
                    written += 1
            except Exception as exc:
                log_chunks.append(
                    f"[WARN] Failed to downscale {src.name} for images_{factor}: {exc}"
                )
        counts[factor] = written
        log_chunks.append(
            f"[INFO] images_{factor}/ ready: {written} image(s) at 1/{factor} scale."
        )
    log_text = "\n".join(log_chunks)
    return counts, (log_text + "\n") if log_text else ""


def _export_sparse_ply(colmap_exe: str, model_dir: Path) -> tuple[Path | None, str]:
    """Convert a COLMAP sparse model directory into a portable PLY point cloud.

    Returns (ply_path_or_none, log_text). The log text always includes the
    invoked command so the user can see what we ran in the Execution Log.
    """
    ply_path = model_dir / "points3D.ply"
    cmd = [
        colmap_exe,
        "model_converter",
        "--input_path",
        str(model_dir),
        "--output_path",
        str(ply_path),
        "--output_type",
        "PLY",
    ]
    code, out = _run_command(cmd)
    log_chunks = [f"$ {' '.join(shlex.quote(p) for p in cmd)}"]
    if out:
        log_chunks.append(out)
    if code == 0 and ply_path.exists() and ply_path.stat().st_size > 0:
        log_chunks.append(f"[INFO] Exported point cloud to {ply_path}")
        return ply_path, "\n".join(log_chunks) + "\n"
    log_chunks.append(
        f"[WARN] PLY export failed (exit code {code}); 3D preview will be hidden."
    )
    return None, "\n".join(log_chunks) + "\n"


def prepare_colmap_dataset(state: dict[str, Any], log_text: str, progress=gr.Progress(track_tqdm=False)):
    hidden_viewer = gr.update(value=None, visible=False)
    hidden_card = gr.update(visible=False)
    live_card = gr.update(visible=True)
    live_viewer = gr.update(value=None, visible=True)
    progress_done = "_COLMAP complete._"

    def emit(
        log: str,
        st: dict[str, Any],
        status: str,
        summary: str,
        viewer: Any,
        card: Any,
        progress_text: str,
    ):
        return (
            log,
            st,
            gr.update(value=status),
            gr.update(value=summary),
            viewer,
            card,
            gr.update(value=progress_text),
        )

    if not state.get("initialized"):
        yield emit(
            _append_log(log_text, "[ERROR] Initialize splat first.\n"),
            state,
            "COLMAP not prepared.",
            _format_sparse_summary(None),
            hidden_viewer,
            hidden_card,
            "_COLMAP failed — session not initialized._",
        )
        return
    colmap_exe = _find_colmap_executable()
    if colmap_exe is None:
        yield emit(
            _append_log(log_text, "[ERROR] COLMAP executable not found in PATH.\n"),
            state,
            "COLMAP not found in PATH.",
            _format_sparse_summary(None),
            hidden_viewer,
            hidden_card,
            "_COLMAP failed — executable not found._",
        )
        return

    splat_dir = Path(state["splat_dir"])
    stills_dir = Path(state["stills_dir"])
    projection = state.get("projection") or {}
    is_cubemap = _is_equirect_projection(projection if isinstance(projection, dict) else {})
    frames = sorted(
        stills_dir.glob(f"{state['splat_name']}-*.png"),
        key=_colmap_still_sort_key,
    )
    if not frames:
        yield emit(
            _append_log(log_text, "[ERROR] No extracted stills found to build COLMAP dataset.\n"),
            state,
            "No stills found.",
            _format_sparse_summary(None),
            hidden_viewer,
            hidden_card,
            "_COLMAP failed — no stills found._",
        )
        return

    dataset_dir = splat_dir / "dataset"
    images_dir = dataset_dir / "images"
    sparse_dir = dataset_dir / "sparse"
    sparse0 = sparse_dir / "0"
    snapshot_dir = dataset_dir / "_mapper_snapshots"
    db_path = dataset_dir / "database.db"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    if images_dir.exists():
        shutil.rmtree(images_dir)
    images_dir.mkdir(parents=True, exist_ok=True)
    if sparse_dir.exists():
        shutil.rmtree(sparse_dir)
    sparse_dir.mkdir(parents=True, exist_ok=True)
    if snapshot_dir.exists():
        shutil.rmtree(snapshot_dir)
    if db_path.exists():
        db_path.unlink()

    log = _append_log(log_text, f"[INFO] Preparing COLMAP dataset at {dataset_dir}\n")
    progress(0.05, desc="Copying stills")
    if is_cubemap:
        log = _append_log(
            log,
            "[INFO] Cubemap export: time-major names (e.g. 000001_f.png … 000001_u.png) so all six "
            "faces per timestamp stay adjacent. Matching uses exhaustive_matcher so every "
            "same-face pair across time is linked (face-major order previously registered only "
            "the first face block, e.g. all back views).\n",
        )
    for f in frames:
        shutil.copy2(f, images_dir / _colmap_export_filename(f))
    log = _append_log(log, f"[INFO] Copied {len(frames)} stills into {images_dir}\n")
    yield emit(
        log,
        state,
        "Running COLMAP feature extraction…",
        "_Building COLMAP dataset…_",
        live_viewer,
        live_card,
        "**Copying stills**",
    )
    matcher_name: str
    if is_cubemap and len(frames) <= 1000:
        matcher_name = "exhaustive_matcher"
        matcher_cmd = [
            colmap_exe,
            "exhaustive_matcher",
            "--database_path",
            str(db_path),
            "--FeatureMatching.use_gpu",
            "1",
        ]
    else:
        matcher_name = "sequential_matcher"
        matcher_cmd = [
            colmap_exe,
            "sequential_matcher",
            "--database_path",
            str(db_path),
            "--FeatureMatching.use_gpu",
            "1",
            "--SequentialMatching.overlap",
            "20" if not is_cubemap else "40",
            "--SequentialMatching.quadratic_overlap",
            "1",
        ]

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
        (matcher_name, matcher_cmd),
    ]

    for idx, (name, cmd) in enumerate(commands, start=1):
        progress(0.2 + idx * 0.2, desc=f"Running COLMAP {name}")
        log = _append_log(log, f"$ {' '.join(shlex.quote(p) for p in cmd)}\n")
        code, out = _run_command(cmd)
        if out:
            log = _append_log(log, out + "\n")
        if code != 0:
            yield emit(
                _append_log(log, f"[ERROR] COLMAP {name} failed with exit code {code}.\n"),
                state,
                f"COLMAP {name} failed.",
                _format_sparse_summary(None),
                hidden_viewer,
                hidden_card,
                f"_COLMAP failed — {name} exited with code {code}._",
            )
            return

    log = _append_log(
        log,
        f"[INFO] Mapper live preview: snapshots every {COLMAP_MAPPER_SNAPSHOT_FREQ} registered "
        f"images under {snapshot_dir} (viewer refreshes about every "
        f"{COLMAP_VIEWER_REFRESH_SEC:.0f}s once mapping begins).\n",
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
        (
            "mapper_cubemap",
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
                "6",
                "--Mapper.init_min_num_inliers",
                "15",
                "--Mapper.init_min_tri_angle",
                "2",
                "--Mapper.init_max_forward_motion",
                "0.99",
                "--Mapper.init_max_error",
                "8",
                "--Mapper.abs_pose_min_num_inliers",
                "10",
                "--Mapper.filter_max_reproj_error",
                "8",
            ],
        ),
    ]
    if not is_cubemap:
        mapper_attempts = mapper_attempts[:2]

    yield emit(
        log,
        state,
        "Feature extraction and matching complete — starting mapper (point cloud preview begins now)…",
        "_Building COLMAP dataset — mapping in progress…_",
        live_viewer,
        live_card,
        "**Starting COLMAP mapper**",
    )

    model_dir = None
    for idx, (name, cmd) in enumerate(mapper_attempts, start=1):
        progress(0.7 + idx * 0.12, desc=f"Running COLMAP {name}")
        log = _append_log(log, f"$ {' '.join(shlex.quote(p) for p in cmd)}\n")
        if snapshot_dir.exists():
            shutil.rmtree(snapshot_dir)
        code: int | None = None
        for code, chunk, viewer_update in _run_mapper_with_snapshots(
            cmd,
            snapshot_dir=snapshot_dir,
            snapshot_freq=COLMAP_MAPPER_SNAPSHOT_FREQ,
        ):
            if chunk:
                log = _append_log(log, chunk)
            if viewer_update is not None:
                snap_model = _latest_snapshot_model_dir(snapshot_dir) or snapshot_dir
                snap_stats = _read_sparse_stats(snap_model)
                summary = _format_sparse_summary(snap_model, stills_count=len(frames))
                if isinstance(snap_stats.get("images"), int) and isinstance(snap_stats.get("points"), int):
                    status = (
                        f"COLMAP mapping ({name}) — "
                        f"{snap_stats['images']} images, {snap_stats['points']:,} points…"
                    )
                else:
                    status = f"COLMAP mapping ({name})…"
                yield emit(
                    log,
                    state,
                    status,
                    summary,
                    viewer_update,
                    live_card,
                    f"**Running COLMAP {name}**",
                )
            if code is not None:
                break
        if code not in (None, 0):
            log = _append_log(log, f"[WARN] COLMAP {name} exited with code {code}.\n")
        model_dir = _largest_model_dir(sparse_dir)
        if model_dir is not None:
            break
        log = _append_log(log, f"[WARN] COLMAP {name} did not produce a sparse model.\n")

    if model_dir is None:
        yield emit(
            _append_log(log, "[ERROR] COLMAP mapper failed to create any sparse model.\n"),
            state,
            "No sparse model produced.",
            _format_sparse_summary(None),
            hidden_viewer,
            hidden_card,
            "_COLMAP failed — no sparse model produced._",
        )
        return

    # Normalise the chosen model into sparse0. Critical: only rmtree sparse0
    # when it is *not* the model itself, otherwise we'd delete the very files
    # COLMAP just produced (the typical case is model_dir == sparse_dir/"0",
    # which is exactly sparse0).
    if model_dir.resolve() != sparse0.resolve():
        if sparse0.exists():
            shutil.rmtree(sparse0)
        shutil.copytree(model_dir, sparse0)

    state["dataset_path"] = str(dataset_dir)
    state["colmap_prepared"] = True
    _write_manifest(state, float(state.get("fps", 1.0)), int(state.get("max_width", 0)))

    progress(0.9, desc="Generating downscaled image pyramids")
    downscale_counts, downscale_log = _generate_downscaled_images(images_dir, dataset_dir)
    if downscale_log:
        log = _append_log(log, downscale_log)
    if downscale_counts:
        _console(
            "Downscale pyramids written: "
            + ", ".join(f"images_{f}={n}" for f, n in sorted(downscale_counts.items()))
        )

    progress(0.95, desc="Exporting point cloud (PLY)")

    ply_path, ply_log = _export_sparse_ply(colmap_exe, sparse0)
    log = _append_log(log, ply_log)
    if ply_path is not None:
        state["point_cloud_ply"] = str(ply_path)

    sparse_stats = _read_sparse_stats(sparse0)
    face_summary = _registered_face_summary(sparse0, colmap_exe) if is_cubemap else ""
    _console(
        f"COLMAP done: images={sparse_stats['images']} points={sparse_stats['points']} "
        f"cameras={sparse_stats['cameras']} stills={len(frames)} ply={ply_path}"
        + (f" faces=[{face_summary}]" if face_summary else "")
    )

    if face_summary:
        log = _append_log(
            log,
            f"[INFO] Registered images by cubemap face: {face_summary}\n",
        )

    progress(1.0, desc="COLMAP ready")

    viewer_update = _viewer_plot_update_for_sparse(sparse0)
    yield emit(
        _append_log(log, f"[INFO] COLMAP dataset ready: {dataset_dir}\n"),
        state,
        f"COLMAP ready: `{dataset_dir}`",
        _format_sparse_summary(sparse0, stills_count=len(frames)),
        viewer_update,
        gr.update(visible=True),
        progress_done,
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
            gr.update(value=False, interactive=False),
            gr.update(value="", visible=False),
            gr.update(value=[]),
            empty_fs,
            gr.update(interactive=False),
            gr.update(value=""),
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
            gr.update(value=False, interactive=False),
            gr.update(value="", visible=False),
            gr.update(value=[]),
            empty_fs,
            gr.update(interactive=False),
            gr.update(value=""),
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
        "equirect_mode": False,
        "projection": _default_projection_block(False),
        "media": [],
        "staged_sources": {},
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
        gr.update(value=False, interactive=True),
        gr.update(value="", visible=False),
        gr.update(value=[]),
        empty_fs,
        gr.update(interactive=True),
        gr.update(value=""),
    )


def add_media(uploaded_files: Any, local_path: str, fps: float, state: dict[str, Any], log_text: str):
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
            gr.update(value=_active_row_label({})),
            gr.update(value=0.0, interactive=False),
            gr.update(value=0.0, interactive=False),
            gr.update(interactive=False),
            gr.update(interactive=False),
            gr.update(value="", visible=False),
            gr.update(value=""),
        )
    media = list(state.get("media", []))
    existing = {m["path"] for m in media}
    existing_names = {Path(m["path"]).name for m in media}
    candidates = _collect_media_candidates(uploaded_files, local_path)
    staged_map: dict[str, str] = dict(state.get("staged_sources") or {})
    log = log_text
    for raw in candidates:
        path = str(Path(raw).expanduser())
        if path in existing:
            continue
        p = Path(path)
        if not p.is_file():
            log = _append_log(log, f"[WARN] Skipped missing path: {p}\n")
            continue
        if p.suffix.lower() not in {".mp4", ".mov", ".mkv", ".avi", ".webm"}:
            log = _append_log(log, f"[WARN] Skipped unsupported media: {p.name}\n")
            continue
        try:
            staged = _resolve_staged_media_path(path, state)
            staged_map[str(p.resolve())] = str(staged)
            if staged.resolve() != p.resolve():
                log = _append_log(log, f"[INFO] Staged upload to session source: {staged}\n")
            elif _is_gradio_temp_path(p):
                log = _append_log(log, f"[INFO] Using staged session copy: {staged}\n")
            staged_key = str(staged)
            if staged_key in existing or staged.name in existing_names:
                log = _append_log(log, f"[WARN] Skipped duplicate media: {staged.name}\n")
                continue
            dur, stream_info = _probe_media_metadata(staged)
            media.append(
                {
                    "path": staged_key,
                    "original_upload": str(p) if staged.resolve() != p.resolve() else "",
                    "duration": dur,
                    "start": 0.0,
                    "end": dur,
                    "status": "ready",
                    "width": stream_info.get("width", 0),
                    "height": stream_info.get("height", 0),
                    "projection_hints": stream_info.get("hints", []),
                    "projection_suggestion": stream_info.get("suggestion", "flat"),
                }
            )
            existing.add(staged_key)
            existing_names.add(staged.name)
            hint_txt = ", ".join(stream_info.get("hints") or []) or "none"
            res = stream_info.get("width"), stream_info.get("height")
            log = _append_log(
                log,
                f"[INFO] Added media: {staged.name} ({dur:.2f}s, {res[0]}×{res[1]}, projection hints: {hint_txt})\n",
            )
        except Exception as exc:
            log = _append_log(log, f"[WARN] Failed probing media '{p.name}': {exc}\n")
            if _is_gradio_temp_path(p):
                log = _append_log(
                    log,
                    "[WARN] Gradio temp upload could not be read. "
                    "Use **Or local file path** below (paste the path to pano.mp4 on disk), "
                    "or copy the file into the session `source` folder and retry.\n",
                )

    state["staged_sources"] = staged_map
    state["media"] = media
    state["active_row"] = -1
    _write_manifest(state, float(state.get("fps", fps)), int(state.get("max_width", 0)))
    can_generate = len(media) > 0
    equirect = bool(state.get("equirect_mode", False))
    show_hint, hint_md = _equirect_hint_markdown(media)
    return (
        log,
        state,
        gr.update(value=_build_media_table(media)),
        gr.update(value=_queue_summary_text(media)),
        gr.update(value=_estimate_summary(media, float(fps), int(state.get("max_width", 0)), equirect)),
        gr.update(interactive=can_generate),
        gr.update(value=None),
        gr.update(interactive=False, variant="secondary"),
        gr.update(value=_active_row_label(state)),
        gr.update(value=0.0, interactive=False),
        gr.update(value=0.0, interactive=False),
        gr.update(interactive=False),
        gr.update(interactive=False),
        gr.update(value=hint_md, visible=show_hint),
        gr.update(value=""),
    )


def _active_row_label(state: dict[str, Any]) -> str:
    media = state.get("media") or []
    idx = int(state.get("active_row", -1))
    if idx < 0 or idx >= len(media):
        return "_Active row: none. Click a row in the queue to enable Trim / Remove._"
    m = media[idx]
    name = Path(m["path"]).name
    duration = float(m.get("duration", 0.0))
    start = float(m.get("start", 0.0))
    end = float(m.get("end", duration))
    return (
        f"**Active row #{idx + 1}:** `{name}` "
        f"(duration **{duration:.2f}s**, current trim **{start:.2f}–{end:.2f}s**)"
    )


def on_media_row_select(state: dict[str, Any], evt: gr.SelectData):
    media = state.get("media") or []
    if not media:
        state["active_row"] = -1
        return (
            state,
            gr.update(value=_active_row_label(state)),
            gr.update(value=0.0, interactive=False),
            gr.update(value=0.0, interactive=False),
            gr.update(interactive=False),
            gr.update(interactive=False),
        )
    idx_raw = getattr(evt, "index", None)
    if isinstance(idx_raw, (list, tuple)) and idx_raw:
        row_idx = int(idx_raw[0])
    elif isinstance(idx_raw, int):
        row_idx = idx_raw
    else:
        row_idx = -1
    if row_idx < 0 or row_idx >= len(media):
        return (
            state,
            gr.update(value=_active_row_label(state)),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
        )
    state["active_row"] = row_idx
    m = media[row_idx]
    duration = float(m.get("duration", 0.0))
    start = float(m.get("start", 0.0))
    end = float(m.get("end", duration))
    return (
        state,
        gr.update(value=_active_row_label(state)),
        gr.update(value=start, interactive=True, maximum=duration),
        gr.update(value=end, interactive=True, maximum=duration),
        gr.update(interactive=True, variant="primary"),
        gr.update(interactive=True, variant="stop"),
    )


def apply_trim(start_val: float, end_val: float, fps: float, state: dict[str, Any], log_text: str):
    media = list(state.get("media") or [])
    idx = int(state.get("active_row", -1))
    if idx < 0 or idx >= len(media):
        return (
            _append_log(log_text, "[WARN] Apply Trim: no active row selected.\n"),
            state,
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
        )
    m = dict(media[idx])
    duration = float(m.get("duration", 0.0))
    start = max(0.0, float(start_val or 0.0))
    end = min(duration, float(end_val or duration))
    if end <= start:
        return (
            _append_log(log_text, f"[WARN] Apply Trim: end ({end:.2f}s) must be greater than start ({start:.2f}s).\n"),
            state,
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
        )
    m["start"] = start
    m["end"] = end
    media[idx] = m
    state["media"] = media
    _write_manifest(state, float(state.get("fps", fps)), int(state.get("max_width", 0)))
    log = _append_log(
        log_text,
        f"[INFO] Trim applied to {Path(m['path']).name}: {start:.2f}s -> {end:.2f}s "
        f"(effective {(end - start):.2f}s).\n",
    )
    return (
        log,
        state,
        gr.update(value=_build_media_table(media)),
        gr.update(value=_queue_summary_text(media)),
        gr.update(value=_estimate_summary(
            media, float(fps), int(state.get("max_width", 0)), bool(state.get("equirect_mode", False))
        )),
        gr.update(value=_active_row_label(state)),
        gr.update(interactive=len(media) > 0),
    )


def remove_media_row(fps: float, state: dict[str, Any], log_text: str):
    media = list(state.get("media") or [])
    idx = int(state.get("active_row", -1))
    if idx < 0 or idx >= len(media):
        return (
            _append_log(log_text, "[WARN] Remove: no active row selected.\n"),
            state,
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(value=0.0, interactive=False),
            gr.update(value=0.0, interactive=False),
            gr.update(interactive=False),
            gr.update(interactive=False),
            gr.update(),
        )
    removed = media.pop(idx)
    state["media"] = media
    state["active_row"] = -1
    _write_manifest(state, float(state.get("fps", fps)), int(state.get("max_width", 0)))
    log = _append_log(log_text, f"[INFO] Removed from queue: {Path(removed['path']).name}.\n")
    can_generate = len(media) > 0
    return (
        log,
        state,
        gr.update(value=_build_media_table(media)),
        gr.update(value=_queue_summary_text(media)),
        gr.update(value=_estimate_summary(
            media, float(fps), int(state.get("max_width", 0)), bool(state.get("equirect_mode", False))
        )),
        gr.update(value=_active_row_label(state)),
        gr.update(value=0.0, interactive=False),
        gr.update(value=0.0, interactive=False),
        gr.update(interactive=False, variant="primary"),
        gr.update(interactive=False, variant="stop"),
        gr.update(interactive=can_generate),
    )


def generate_stills(
    fps: float,
    max_width: int,
    equirect_mode: bool,
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
    state["equirect_mode"] = bool(equirect_mode)
    if equirect_mode:
        detection = _projection_detection_from_media(media)
        state["projection"] = _default_projection_block(True, detection)
    else:
        state["projection"] = _default_projection_block(False)

    stills_dir = Path(state["stills_dir"])
    splat_name = state["splat_name"]
    existing = sorted(stills_dir.glob(f"{splat_name}-*.png"))
    log = _append_log(log_text, f"[INFO] Starting extraction from {len(media)} media files.\n")
    if equirect_mode:
        faces = _equirect_faces_per_frame()
        face_list = ", ".join(CUBE_FACE_NAMES[f] for f, _, _ in EQUIRECT_CUBE_FACES)
        log = _append_log(
            log,
            f"[INFO] Equirect cubemap mode: {faces} pinhole face(s) per sampled frame "
            f"({face_list}, {EQUIRECT_FACE_FOV_DEG}° FOV). "
            f"Face max width: {_effective_equirect_max_width(int(max_width))}px"
            f"{' (default)' if int(max_width) <= 0 else ''}.\n",
        )
    if existing:
        removed = 0
        for old in existing:
            try:
                old.unlink()
                removed += 1
            except OSError as exc:
                log = _append_log(log, f"[WARN] Could not remove old still {old.name}: {exc}\n")
        log = _append_log(
            log,
            f"[INFO] Removed {removed} existing still(s) before re-extraction (avoids duplicate frames).\n",
        )
        if state.get("colmap_prepared"):
            state["colmap_prepared"] = False
            log = _append_log(
                log,
                "[INFO] COLMAP marked stale — rebuild on the **Build COLMAP Dataset** tab after review.\n",
            )
    next_index = 1

    tile_jobs: list[tuple[str | None, int | None, int | None, str]] = []
    if equirect_mode:
        for face, yaw, pitch, suffix in _equirect_cube_face_jobs():
            tile_jobs.append((face, yaw, pitch, suffix))
    else:
        tile_jobs.append((None, None, None, ".png"))

    total_steps = len(media) * len(tile_jobs)
    step_done = 0
    extract_started = time.monotonic()
    for idx, item in enumerate(media, start=1):
        src = Path(item["path"])
        duration = float(item.get("duration", 0.0))
        start = max(0.0, float(item.get("start", 0.0)))
        end = float(item.get("end", duration))
        end = min(end, duration)
        trim_sec: float | None = (end - start) if end > start and (start > 0.0 or end < duration) else None
        produced = _estimate_frame_count(_trimmed_duration(item), float(fps))
        frame_start = next_index
        tile_ok = 0

        for face, yaw, pitch, suffix in tile_jobs:
            step_done += 1
            if face is None:
                pattern = stills_dir / f"{splat_name}-%06d.png"
                vf = _build_flat_vf(float(fps), int(max_width))
                desc = f"Video {idx}/{len(media)}"
            else:
                pattern = stills_dir / f"{splat_name}-%06d{suffix}"
                vf = _build_equirect_tile_vf(float(fps), int(yaw), int(pitch), int(max_width))
                desc = f"Video {idx}/{len(media)} face {face} ({CUBE_FACE_NAMES.get(face, face)})"

            code, out = _ffmpeg_extract_to_pattern(
                src, pattern, vf, start, trim_sec, frame_start
            )
            cmd_preview = (
                f"ffmpeg … -vf {vf!r} -start_number {frame_start} {pattern.name}"
            )
            log = _append_log(log, f"$ {cmd_preview}\n")
            progress(step_done / max(1, total_steps), desc=desc)
            if code != 0:
                log = _append_log(log, f"[ERROR] ffmpeg failed for {src.name} ({desc})\n{out}\n")
                _console(f"ffmpeg failed for {src.name} rc={code}", "ERROR")
                continue
            tile_ok += 1
            if out:
                log = _append_log(log, out + "\n")

        if tile_ok == len(tile_jobs):
            next_index += produced
        elif tile_ok > 0:
            log = _append_log(
                log,
                f"[WARN] Only {tile_ok}/{len(tile_jobs)} cubemap face pass(es) succeeded for {src.name}; "
                "frame indices may be incomplete. Re-run extraction after fixing errors.\n",
            )
            next_index += produced
    extract_elapsed = max(0.001, time.monotonic() - extract_started)

    frames = sorted(stills_dir.glob(f"{splat_name}-*.png"))
    frame_paths = [str(p) for p in frames]
    _console(f"generate_stills: glob matched {len(frames)} file(s) under {stills_dir} (UI refresh is next step)")
    _write_manifest(state, float(fps), int(max_width))
    log = _append_log(log, f"[INFO] Extraction complete. Total frames: {len(frame_paths)}\n")
    _console(
        f"Extraction finished splat={splat_name} count={len(frames)} "
        f"first={frame_paths[0] if frame_paths else None}"
    )

    # Persist calibration so next session's estimate is more accurate.
    if frames:
        try:
            total_bytes = sum(p.stat().st_size for p in frames)
            bytes_per_frame = total_bytes / float(len(frames))
            seconds_per_frame = extract_elapsed / float(len(frames))
            _save_calibration(bytes_per_frame, seconds_per_frame)
            _console(
                f"Calibration saved: {bytes_per_frame:.0f} B/frame, "
                f"{seconds_per_frame:.3f} s/frame ({extract_elapsed:.2f}s for {len(frames)} frames)"
            )
        except OSError as exc:
            _console(f"Calibration save failed: {exc}", "WARNING")

    return log, state, gr.update(
        value=_estimate_summary(
            media, float(fps), int(max_width), bool(state.get("equirect_mode", False))
        )
    )


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
    if deleted > 0 and state.get("colmap_prepared"):
        state["colmap_prepared"] = False
        log = _append_log(
            log,
            "[INFO] COLMAP marked stale — rebuild on the **Build COLMAP Dataset** tab.\n",
        )
        _write_manifest(state, float(state.get("fps", 1.0)), int(state.get("max_width", 0)))
    _console(f"reject_selected deleted={deleted}")
    return log


settings = _load_settings()
HAS_COLMAP = _find_colmap_executable() is not None

SPLATTER_CSS = """
.splatter-result-card {
    border: 2px solid #1f9d55;
    border-radius: 8px;
    padding: 12px 14px 6px 14px;
    margin-top: 12px;
    background: rgba(31, 157, 85, 0.06);
}
.splatter-result-header h2 {
    color: #1f9d55;
    margin: 0 0 6px 0;
}
.splatter-pointcloud-viewer {
    min-height: 520px;
    background: #000000 !important;
}
.splatter-pointcloud-viewer .plotly,
.splatter-pointcloud-viewer .js-plotly-plot,
.splatter-pointcloud-viewer .user-select-none {
    background: #000000 !important;
}
.colmap-progress-tile {
    min-height: 56px;
    padding: 10px 14px;
    margin: 10px 0 14px 0;
    border: 1px dashed rgba(255, 255, 255, 0.18);
    border-radius: 8px;
    background: rgba(255, 255, 255, 0.02);
}
.colmap-progress-tile p {
    margin: 0;
}
"""

with gr.Blocks(title="Splatter Stills Extractor V2") as demo:
    gr.Markdown("## Splatter Stills Extractor")
    gr.Markdown(
        "Create a session, add media, extract stills, and review frames. "
        "When ready, open the **Build COLMAP Dataset** tab."
    )
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
                gr.Markdown(
                    "_Drag video files into the box below, then click **Add to Queue** "
                    "(copies into the session `source` folder). For large 360° files on Windows, "
                    "paste a local path instead of uploading._"
                )
                media_upload = gr.File(
                    label="Upload media files",
                    file_count="multiple",
                    file_types=["video"],
                    type="filepath",
                    interactive=False,
                )
                upload_staging_status = gr.Markdown("")
                with gr.Row():
                    media_path_input = gr.Textbox(
                        label="Or local file path",
                        placeholder=r"C:\Videos\pano.mp4",
                        scale=4,
                        interactive=False,
                    )
                add_media_button = gr.Button("Add to Queue", variant="secondary", interactive=False)
                queue_summary = gr.Markdown(_queue_summary_text([]))
                media_table = gr.Dataframe(
                    headers=["name", "duration_sec", "start_sec", "end_sec", "status", "path"],
                    value=[],
                    interactive=False,
                    wrap=True,
                    label="Source media queue (click a row to enable Trim / Remove)",
                )
                active_row_label = gr.Markdown(_active_row_label({}))
                with gr.Row():
                    trim_start = gr.Number(label="Trim start (s)", value=0.0, precision=2, interactive=False)
                    trim_end = gr.Number(label="Trim end (s)", value=0.0, precision=2, interactive=False)
                    apply_trim_button = gr.Button("Apply Trim", variant="secondary", interactive=False)
                    remove_button = gr.Button("Remove from Queue", variant="secondary", interactive=False)

            step3_group = gr.Group(visible=False)
            with step3_group:
                gr.Markdown("### Step 3 - Extract Frames")
                equirect_mode = gr.Checkbox(
                    label="Stitched equirectangular (360°) → 6 cubemap faces",
                    value=False,
                    interactive=False,
                )
                equirect_hint = gr.Markdown("", visible=False)
                with gr.Row():
                    fps = gr.Number(label="Images per second", value=float(settings["default_fps"]), precision=3, interactive=False)
                    max_width = gr.Number(
                        label=f"Max width per cube face (0 = {EQUIRECT_DEFAULT_TILE_MAX_WIDTH}px in equirect mode)",
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

        with gr.Column(scale=1):
            workflow_status = gr.Markdown(_workflow_summary({"initialized": False, "media": []}))
            next_action = gr.Markdown(_next_action_hint({"initialized": False, "media": []}))

    initial_colmap_msg = (
        "COLMAP runs on the **Build COLMAP Dataset** tab."
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
            equirect_mode,
            equirect_hint,
            frame_gallery,
            frames_state,
            media_path_input,
            upload_staging_status,
        ],
    ).then(
        fn=lambda: (
            gr.update(interactive=True),
            gr.update(interactive=True),
            gr.update(interactive=True),
            gr.update(interactive=True),
        ),
        outputs=[reject_button, select_all_button, invert_button, clear_button],
    ).then(
        fn=_workflow_ui_updates,
        inputs=[state],
        outputs=[workflow_status, next_action, step2_group, step3_group, step4_group],
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
        inputs=[media_upload, media_path_input, fps, state, log_output],
        outputs=[
            log_output,
            state,
            media_table,
            queue_summary,
            estimate,
            generate_button,
            media_upload,
            add_media_button,
            active_row_label,
            trim_start,
            trim_end,
            apply_trim_button,
            remove_button,
            equirect_hint,
            media_path_input,
        ],
    ).then(
        fn=_workflow_ui_updates,
        inputs=[state],
        outputs=[workflow_status, next_action, step2_group, step3_group, step4_group],
    ).then(
        fn=_extract_inline_error,
        inputs=[log_output],
        outputs=[last_error],
    )

    media_upload.change(
        fn=_on_media_upload_changed,
        inputs=[media_upload, state],
        outputs=[add_media_button, upload_staging_status],
    )

    media_path_input.change(
        fn=_local_path_queue_button,
        inputs=[media_path_input, media_upload, state],
        outputs=[add_media_button],
    )

    media_table.select(
        fn=on_media_row_select,
        inputs=[state],
        outputs=[state, active_row_label, trim_start, trim_end, apply_trim_button, remove_button],
    )

    apply_trim_button.click(
        fn=apply_trim,
        inputs=[trim_start, trim_end, fps, state, log_output],
        outputs=[
            log_output,
            state,
            media_table,
            queue_summary,
            estimate,
            active_row_label,
            generate_button,
        ],
    ).then(
        fn=_workflow_ui_updates,
        inputs=[state],
        outputs=[workflow_status, next_action, step2_group, step3_group, step4_group],
    ).then(
        fn=_extract_inline_error,
        inputs=[log_output],
        outputs=[last_error],
    )

    remove_button.click(
        fn=remove_media_row,
        inputs=[fps, state, log_output],
        outputs=[
            log_output,
            state,
            media_table,
            queue_summary,
            estimate,
            active_row_label,
            trim_start,
            trim_end,
            apply_trim_button,
            remove_button,
            generate_button,
        ],
    ).then(
        fn=_workflow_ui_updates,
        inputs=[state],
        outputs=[workflow_status, next_action, step2_group, step3_group, step4_group],
    ).then(
        fn=_extract_inline_error,
        inputs=[log_output],
        outputs=[last_error],
    )

    fps.change(
        fn=on_fps_changed,
        inputs=[fps, state],
        outputs=[state, estimate, generate_button],
    ).then(
        fn=_workflow_ui_updates,
        inputs=[state],
        outputs=[workflow_status, next_action, step2_group, step3_group, step4_group],
    )

    max_width.change(
        fn=on_max_width_changed,
        inputs=[max_width, state],
        outputs=[state, estimate],
    )

    equirect_mode.change(
        fn=on_equirect_mode_changed,
        inputs=[equirect_mode, fps, max_width, state],
        outputs=[state, estimate],
    ).then(
        fn=_workflow_ui_updates,
        inputs=[state],
        outputs=[workflow_status, next_action, step2_group, step3_group, step4_group],
    )

    generate_button.click(
        fn=generate_stills,
        inputs=[fps, max_width, equirect_mode, state, log_output],
        outputs=[log_output, state, estimate],
    ).then(
        fn=_update_frame_review_ui,
        inputs=[state],
        outputs=[frames_state, frame_gallery, selection_summary],
    ).then(
        fn=_workflow_ui_updates,
        inputs=[state],
        outputs=[workflow_status, next_action, step2_group, step3_group, step4_group],
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
        outputs=[workflow_status, next_action, step2_group, step3_group, step4_group],
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
        outputs=[workflow_status, next_action, step2_group, step3_group, step4_group],
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


with gr.Blocks(title="Splatter COLMAP Builder") as colmap_demo:
    gr.Markdown("## Build COLMAP Dataset")
    gr.Markdown(
        "Load a splat session that already has extracted stills, run COLMAP, and preview the "
        "sparse point cloud (vertex colors from reconstruction, auto-scaled to fit the viewer)."
    )
    colmap_state = gr.State({"initialized": False, "media": []})
    with gr.Row():
        with gr.Column(scale=2):
            with gr.Group():
                gr.Markdown("### Step 1 — Load session")
                with gr.Row():
                    colmap_base_dir = gr.Textbox(
                        label="Splatter sessions base dir",
                        value=settings["base_output_dir"],
                        scale=3,
                    )
                    colmap_refresh_btn = gr.Button("Refresh Sessions", scale=1)
                colmap_session_selector = gr.Dropdown(
                    label="Splat session",
                    choices=[],
                    value=None,
                )
                colmap_load_btn = gr.Button("Load Session", variant="primary")
                colmap_session_status = gr.Markdown("_Select a session and click **Load Session**._")

            with gr.Group():
                gr.Markdown("### Step 2 — Run COLMAP")
                colmap_prepare_button = gr.Button(
                    "Build COLMAP Dataset",
                    variant="primary",
                    interactive=False,
                )
                colmap_progress = gr.Markdown(
                    "_Progress appears here while COLMAP runs._",
                    elem_classes=["colmap-progress-tile"],
                )
                colmap_summary = gr.Markdown(_format_sparse_summary(None))

            colmap_result_card = gr.Group(visible=False, elem_classes=["splatter-result-card"])
            with colmap_result_card:
                gr.Markdown("## COLMAP preview", elem_classes=["splatter-result-header"])
                gr.Markdown(
                    "_Drag to rotate, scroll to zoom. The scatter plot appears once **mapping** starts "
                    "(after feature extraction and matching finish), then updates every few "
                    "registered images. Download the raw model from "
                    "`<session>/dataset/sparse/0/points3D.ply`_"
                )
                colmap_point_cloud_viewer = gr.Plot(
                    label="Sparse point cloud",
                    show_label=False,
                    container=True,
                    elem_classes=["splatter-pointcloud-viewer"],
                )

        with gr.Column(scale=1):
            colmap_workflow_status = gr.Markdown("### COLMAP workflow\n- Session: not loaded")
            colmap_next_action = gr.Markdown(
                "### Next Action\nRefresh sessions, pick a splat, and click **Load Session**."
            )

    colmap_initial_msg = (
        "COLMAP status: Ready to run."
        if HAS_COLMAP
        else "COLMAP status: `colmap` not found in PATH. Install COLMAP to enable preparation."
    )
    colmap_run_status = gr.Markdown(colmap_initial_msg)
    colmap_last_error = gr.Markdown("", visible=False)
    with gr.Accordion("Execution log", open=False):
        colmap_log_output = gr.Textbox(
            label="Execution Console",
            value="",
            lines=16,
            max_lines=32,
            interactive=False,
        )

    colmap_refresh_btn.click(
        fn=refresh_colmap_sessions,
        inputs=[colmap_base_dir],
        outputs=[colmap_session_selector, colmap_session_status],
    )
    colmap_load_btn.click(
        fn=load_colmap_session,
        inputs=[colmap_base_dir, colmap_session_selector, colmap_log_output],
        outputs=[
            colmap_log_output,
            colmap_state,
            colmap_run_status,
            colmap_summary,
            colmap_point_cloud_viewer,
            colmap_result_card,
            colmap_prepare_button,
        ],
    ).then(
        fn=lambda st: (
            gr.update(
                value=(
                    "### COLMAP workflow\n"
                    f"- Session: **{st.get('splat_name', '?')}**\n"
                    f"- Frames on disk: **{_state_frame_count(st)}**\n"
                    f"- COLMAP: **{'ready' if st.get('colmap_prepared') else 'not prepared'}**"
                )
            ),
            gr.update(
                value=(
                    "### Next Action\n"
                    + (
                        "Dataset ready — switch to **Train Splat**."
                        if st.get("colmap_prepared")
                        else "Click **Build COLMAP Dataset** when stills look good."
                    )
                )
            ),
        ),
        inputs=[colmap_state],
        outputs=[colmap_workflow_status, colmap_next_action],
    ).then(
        fn=_extract_inline_error,
        inputs=[colmap_log_output],
        outputs=[colmap_last_error],
    )

    colmap_prepare_button.click(
        fn=lambda: (
            gr.update(visible=True),
            gr.update(value=None, visible=True),
            gr.update(value="_COLMAP running — see progress below._"),
        ),
        inputs=None,
        outputs=[colmap_result_card, colmap_point_cloud_viewer, colmap_progress],
    ).then(
        fn=prepare_colmap_dataset,
        inputs=[colmap_state, colmap_log_output],
        outputs=[
            colmap_log_output,
            colmap_state,
            colmap_run_status,
            colmap_summary,
            colmap_point_cloud_viewer,
            colmap_result_card,
            colmap_progress,
        ],
        show_progress="full",
        show_progress_on=colmap_progress,
    ).then(
        fn=lambda st: (
            gr.update(
                value=(
                    "### COLMAP workflow\n"
                    f"- Session: **{st.get('splat_name', '?')}**\n"
                    f"- Frames on disk: **{_state_frame_count(st)}**\n"
                    f"- COLMAP: **{'ready' if st.get('colmap_prepared') else 'not prepared'}**"
                )
            ),
            gr.update(
                value=(
                    "### Next Action\n"
                    + (
                        "Dataset ready — switch to **Train Splat**."
                        if st.get("colmap_prepared")
                        else "Review the log and retry COLMAP if needed."
                    )
                )
            ),
        ),
        inputs=[colmap_state],
        outputs=[colmap_workflow_status, colmap_next_action],
    ).then(
        fn=_extract_inline_error,
        inputs=[colmap_log_output],
        outputs=[colmap_last_error],
    )


if __name__ == "__main__":
    # Allow Gradio to serve generated stills from user-selected output directories.
    demo.queue().launch(
        allowed_paths=[str(APP_DIR), str(Path.home())],
        css=SPLATTER_CSS,
    )
