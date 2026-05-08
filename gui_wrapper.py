from __future__ import annotations

import json
import os
import queue
import shlex
import shutil
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any

import gradio as gr

WRAPPER_DIR = Path(__file__).resolve().parent
DEFAULT_REPO_DIR = Path(r"C:\Users\GusFr\3dgrut")
PRESETS_FILE = WRAPPER_DIR / "gui_presets.json"
PROCESS_LOCK = threading.Lock()
MAX_LOG_CHARS = 200_000
DEFAULT_SPLAT_BASE_DIR = Path.home() / "Pictures" / "Splatter"

MODE_CONFIG_CHOICES = {
    "3DGUT": [
        "apps/colmap_3dgut.yaml",
        "apps/nerf_synthetic_3dgut.yaml",
        "apps/scannetpp_3dgut.yaml",
        "apps/ncore_3dgut.yaml",
        "apps/ncore_3dgut_mcmc.yaml",
        "apps/colmap_3dgut_mcmc.yaml",
        "apps/cusfm_3dgut.yaml",
        "apps/cusfm_3dgut_mcmc.yaml",
    ],
    "3DGRT": [
        "apps/colmap_3dgrt.yaml",
        "apps/nerf_synthetic_3dgrt.yaml",
        "apps/scannetpp_3dgrt.yaml",
        "apps/ncore_3dgrt.yaml",
        "apps/ncore_3dgrt_mcmc.yaml",
        "apps/colmap_3dgrt_mcmc.yaml",
    ],
}


def _append_log(current: str, text: str) -> str:
    merged = current + text
    return merged[-MAX_LOG_CHARS:] if len(merged) > MAX_LOG_CHARS else merged


def _load_presets() -> dict[str, dict[str, Any]]:
    if not PRESETS_FILE.exists():
        return {}
    try:
        data = json.loads(PRESETS_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _save_presets(presets: dict[str, dict[str, Any]]) -> None:
    PRESETS_FILE.write_text(json.dumps(presets, indent=2), encoding="utf-8")


def _preset_choices() -> list[str]:
    return sorted(_load_presets().keys())


def _splat_choices(base_dir: str) -> list[str]:
    base = Path(base_dir).expanduser()
    if not base.exists():
        return []
    names: list[str] = []
    for child in base.iterdir():
        if child.is_dir() and (child / "manifest.json").exists():
            names.append(child.name)
    return sorted(names)


def refresh_splats(base_dir: str, current_log: str):
    choices = _splat_choices(base_dir)
    msg = f"[INFO] Found {len(choices)} splat sessions in {Path(base_dir).expanduser()}.\n"
    return _append_log(current_log, msg), gr.update(choices=choices, value=(choices[0] if choices else None))


def load_splat_session(splat_base_dir: str, splat_name: str, current_log: str):
    if not splat_name:
        return current_log, gr.update(), gr.update(), gr.update(), gr.update(value="No splat selected.")
    manifest_path = Path(splat_base_dir).expanduser() / splat_name / "manifest.json"
    if not manifest_path.exists():
        return (
            _append_log(current_log, f"[ERROR] Manifest not found: {manifest_path}\n"),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(value=f"Manifest not found for '{splat_name}'."),
        )
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return (
            _append_log(current_log, f"[ERROR] Failed to read manifest: {exc}\n"),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(value=f"Failed to parse manifest for '{splat_name}'."),
        )

    dataset_path = manifest.get("dataset_path", "")
    stills_dir = manifest.get("stills_dir", "")
    chosen_data_path = ""
    if dataset_path:
        ds = Path(dataset_path).expanduser()
        has_sparse = (ds / "sparse" / "0" / "images.bin").exists() or (ds / "sparse" / "0" / "images.txt").exists()
        if has_sparse:
            chosen_data_path = str(ds)
    if not chosen_data_path and stills_dir:
        chosen_data_path = stills_dir
    if chosen_data_path and Path(chosen_data_path).name.lower() == "stills":
        sibling_dataset = Path(chosen_data_path).parent / "dataset"
        sibling_sparse = sibling_dataset / "sparse" / "0"
        sibling_has_sparse = (sibling_sparse / "images.bin").exists() or (sibling_sparse / "images.txt").exists()
        if sibling_has_sparse:
            chosen_data_path = str(sibling_dataset)
    if not chosen_data_path:
        return (
            _append_log(current_log, "[ERROR] Manifest missing usable dataset/stills path.\n"),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(value=f"Manifest for '{splat_name}' missing usable data path."),
        )
    out_dir = str((Path(splat_base_dir).expanduser() / splat_name / "runs").expanduser())
    source_label = "dataset_path" if chosen_data_path == str(Path(dataset_path).expanduser()) and dataset_path else "stills_dir"
    colmap_ready = bool(manifest.get("colmap_prepared", False))
    log = _append_log(
        current_log,
        f"[INFO] Loaded splat session '{splat_name}' from {manifest_path} (using {source_label}).\n",
    )
    status_extra = "COLMAP prepared" if colmap_ready else "COLMAP not prepared"
    return (
        log,
        gr.update(value=chosen_data_path),
        gr.update(value=splat_name),
        gr.update(value=out_dir),
        gr.update(value=f"Loaded splat: `{splat_name}` ({status_extra})"),
    )


def _probe_video_duration_seconds(video_path: Path) -> float:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(video_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "ffprobe failed to read video duration.")
    return float(result.stdout.strip())


def _extract_video_path(video_value: Any) -> str | None:
    if video_value is None:
        return None
    if isinstance(video_value, str):
        return video_value
    if isinstance(video_value, dict):
        for key in ("path", "video", "name"):
            value = video_value.get(key)
            if isinstance(value, str) and value:
                return value
        return None
    if isinstance(video_value, (tuple, list)) and video_value:
        first = video_value[0]
        if isinstance(first, str):
            return first
        if isinstance(first, dict):
            for key in ("path", "video", "name"):
                value = first.get(key)
                if isinstance(value, str) and value:
                    return value
    return None


def _resolve_video_source(video_path_text: str, video_upload_value: Any) -> str | None:
    if video_path_text and video_path_text.strip():
        return video_path_text.strip()
    return _extract_video_path(video_upload_value)


def _create_video_component():
    variants = [
        {"sources": ["upload"], "type": "filepath"},
        {"sources": ["upload"]},
        {},
    ]
    for kwargs in variants:
        try:
            return gr.Video(label="Video file (.mp4)", **kwargs)
        except TypeError:
            continue
    return gr.Video(label="Video file (.mp4)")


def _repo_paths(repo_dir: str) -> tuple[Path, Path]:
    root = Path(repo_dir).expanduser()
    return root, root / "train.py"


def _query_gpus() -> list[dict[str, Any]]:
    if shutil.which("nvidia-smi") is None:
        return []
    cmd = [
        "nvidia-smi",
        "--query-gpu=index,name,memory.total,memory.free,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        return []
    gpus: list[dict[str, Any]] = []
    for line in result.stdout.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 5:
            continue
        try:
            gpus.append(
                {
                    "index": parts[0],
                    "name": parts[1],
                    "memory_total_mb": int(parts[2]),
                    "memory_free_mb": int(parts[3]),
                    "util_percent": int(parts[4]),
                }
            )
        except ValueError:
            continue
    return gpus


def _gpu_choice_value(gpu_index: str) -> str:
    return f"gpu:{gpu_index}"


def _gpu_choices() -> list[tuple[str, str]]:
    choices: list[tuple[str, str]] = [("Auto (best available GPU)", "auto")]
    for gpu in _query_gpus():
        label = (
            f"GPU {gpu['index']} | {gpu['name']} | "
            f"Free {gpu['memory_free_mb']}MB / {gpu['memory_total_mb']}MB | "
            f"Util {gpu['util_percent']}%"
        )
        choices.append((label, _gpu_choice_value(gpu["index"])))
    return choices


def _pick_best_gpu_index(gpus: list[dict[str, Any]]) -> str | None:
    if not gpus:
        return None
    best = sorted(gpus, key=lambda g: (-g["memory_free_mb"], g["util_percent"], -g["memory_total_mb"]))[0]
    return best["index"]


def _resolve_gpu_selection(gpu_selection: str) -> tuple[str | None, str]:
    gpus = _query_gpus()
    if not gpus:
        return None, "[INFO] No NVIDIA GPU detected via nvidia-smi. Using system default device selection."
    if gpu_selection.startswith("gpu:"):
        chosen = gpu_selection.split(":", 1)[1]
        if any(g["index"] == chosen for g in gpus):
            return chosen, f"[INFO] Using manually selected GPU {chosen}."
    auto = _pick_best_gpu_index(gpus)
    if auto is None:
        return None, "[INFO] Could not determine best GPU. Using system default device selection."
    return auto, f"[INFO] Auto-selected GPU {auto}."


def refresh_gpu_list(current_choice: str, current_log: str):
    choices = _gpu_choices()
    values = {value for _, value in choices}
    next_choice = current_choice if current_choice in values else "auto"
    status = "Detected GPUs via nvidia-smi." if len(choices) > 1 else "No GPUs found via nvidia-smi."
    return (
        _append_log(current_log, f"[INFO] {status}\n"),
        gr.update(choices=choices, value=next_choice),
        gr.update(value=f"GPU status: {status}"),
    )


def _build_command(
    repo_dir: str,
    config_name: str,
    data_path: str,
    experiment_name: str,
    out_dir: str,
    resume_checkpoint: str,
    downsample: int,
    iterations: int,
    optimizer: str,
    export_usdz: bool,
    conda_env: str,
) -> list[str]:
    _, train_script = _repo_paths(repo_dir)
    command: list[str] = []
    if conda_env.strip():
        command.extend(["conda", "run", "-n", conda_env.strip(), "python"])
    else:
        command.append(sys.executable)

    command.extend(
        [
            str(train_script),
            "--config-name",
            config_name,
            f"path={str(Path(data_path).expanduser())}",
            f"out_dir={str(Path(out_dir).expanduser())}",
            f"experiment_name={experiment_name.strip()}",
            f"dataset.downsample_factor={int(downsample)}",
            f"n_iterations={int(iterations)}",
            f"optimizer.type={optimizer}",
            f"export_usd.enabled={str(bool(export_usdz)).lower()}",
        ]
    )
    if resume_checkpoint.strip():
        command.append(f"resume={str(Path(resume_checkpoint).expanduser())}")
    return command


def _validate_inputs(
    repo_dir: str,
    data_path: str,
    mode: str,
    config_name: str,
    out_dir: str,
    resume_checkpoint: str,
) -> str | None:
    repo_root, train_script = _repo_paths(repo_dir)
    if not repo_root.exists():
        return f"Repository directory not found: {repo_root}"
    if not train_script.exists():
        return f"Missing train script: {train_script}"

    candidate = Path(data_path).expanduser()
    if not candidate.exists():
        return f"Invalid data_path: '{candidate}' does not exist."

    if mode not in MODE_CONFIG_CHOICES:
        return f"Unknown mode '{mode}'. Expected one of: {', '.join(MODE_CONFIG_CHOICES.keys())}"
    if config_name not in MODE_CONFIG_CHOICES[mode]:
        return f"Config '{config_name}' is not valid for mode '{mode}'."

    config_path = repo_root / "configs" / config_name
    if not config_path.exists():
        return f"Missing Hydra config file: {config_path}"

    if not out_dir.strip():
        return "Output directory cannot be empty."
    try:
        Path(out_dir).expanduser().mkdir(parents=True, exist_ok=True)
    except Exception as exc:
        return f"Could not create/access out_dir '{out_dir}': {exc}"

    if resume_checkpoint.strip() and not Path(resume_checkpoint).expanduser().exists():
        return f"Resume checkpoint not found: {Path(resume_checkpoint).expanduser()}"

    return None


def _resolve_data_path_for_config(data_path: str, config_name: str) -> tuple[str, str | None]:
    """
    Resolve common path mixups for COLMAP-style configs.
    Returns: (resolved_data_path, info_message_or_none)
    Raises ValueError when no valid path can be resolved.
    """
    data_root = Path(data_path).expanduser()
    is_colmap_style = any(token in config_name for token in ("colmap", "cusfm"))
    if not is_colmap_style:
        return str(data_root), None

    sparse_dir = data_root / "sparse" / "0"
    has_sparse = (sparse_dir / "images.bin").exists() or (sparse_dir / "images.txt").exists()
    if has_sparse:
        return str(data_root), None

    # Common stills-app layout: <splat>/stills and <splat>/dataset
    sibling_dataset = data_root.parent / "dataset"
    sibling_sparse = sibling_dataset / "sparse" / "0"
    sibling_has_sparse = (sibling_sparse / "images.bin").exists() or (sibling_sparse / "images.txt").exists()
    if sibling_has_sparse:
        return (
            str(sibling_dataset),
            f"[INFO] Auto-resolved COLMAP dataset path: '{sibling_dataset}' (from '{data_root}').",
        )

    raise ValueError(
        f"Selected config '{config_name}' expects COLMAP metadata under '{sparse_dir}'. "
        "Your path looks like raw extracted stills. Run COLMAP first (or use a dataset preset matching your data format)."
    )


def _colmap_path_debug(data_path: str) -> str:
    data_root = Path(data_path).expanduser()
    sparse_dir = data_root / "sparse" / "0"
    sibling_dataset = data_root.parent / "dataset"
    sibling_sparse = sibling_dataset / "sparse" / "0"
    lines = [
        "[DEBUG] COLMAP path diagnostics:",
        f"  - input data_path: {data_root}",
        f"  - input exists: {data_root.exists()}",
        f"  - input sparse dir: {sparse_dir}",
        f"  - input images.bin exists: {(sparse_dir / 'images.bin').exists()}",
        f"  - input images.txt exists: {(sparse_dir / 'images.txt').exists()}",
        f"  - sibling dataset dir: {sibling_dataset}",
        f"  - sibling dataset exists: {sibling_dataset.exists()}",
        f"  - sibling sparse dir: {sibling_sparse}",
        f"  - sibling images.bin exists: {(sibling_sparse / 'images.bin').exists()}",
        f"  - sibling images.txt exists: {(sibling_sparse / 'images.txt').exists()}",
    ]
    return "\n".join(lines)


def _reader_thread(proc: subprocess.Popen[str], log_queue: queue.Queue[str]) -> None:
    if proc.stdout is None:
        return
    for line in iter(proc.stdout.readline, ""):
        if not line:
            break
        log_queue.put(line)
    proc.stdout.close()


def _update_config_choices(mode: str):
    choices = MODE_CONFIG_CHOICES[mode]
    return gr.update(choices=choices, value=choices[0])


def start_training(
    repo_dir: str,
    data_path: str,
    experiment_name: str,
    mode: str,
    config_name: str,
    out_dir: str,
    resume_checkpoint: str,
    downsample: int,
    iterations: int,
    optimizer: str,
    export_usdz: bool,
    gpu_selection: str,
    conda_env: str,
    run_state: dict[str, Any],
):
    with PROCESS_LOCK:
        if run_state.get("running"):
            yield "A training process is already running.\n", run_state, gr.update(interactive=False), gr.update(
                interactive=True
            )
            return

        error = _validate_inputs(repo_dir, data_path, mode, config_name, out_dir, resume_checkpoint)
        if error:
            yield f"[ERROR] {error}\n", run_state, gr.update(interactive=True), gr.update(interactive=False)
            return

        if not experiment_name.strip():
            experiment_name = f"{mode.lower()}_run"
        resolved_data_path = data_path
        data_path_info: str | None = None
        try:
            resolved_data_path, data_path_info = _resolve_data_path_for_config(data_path, config_name)
        except ValueError as exc:
            debug_info = _colmap_path_debug(data_path)
            yield (
                f"[ERROR] {exc}\n{debug_info}\n",
                run_state,
                gr.update(interactive=True),
                gr.update(interactive=False),
            )
            return

        repo_root, _ = _repo_paths(repo_dir)
        command = _build_command(
            repo_dir=repo_dir,
            config_name=config_name,
            data_path=resolved_data_path,
            experiment_name=experiment_name,
            out_dir=out_dir,
            resume_checkpoint=resume_checkpoint,
            downsample=downsample,
            iterations=iterations,
            optimizer=optimizer,
            export_usdz=export_usdz,
            conda_env=conda_env,
        )
        command_preview = " ".join(shlex.quote(part) for part in command)
        selected_gpu, gpu_message = _resolve_gpu_selection(gpu_selection)
        launch_env = os.environ.copy()
        if selected_gpu is not None:
            launch_env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            launch_env["CUDA_VISIBLE_DEVICES"] = selected_gpu
        try:
            process = subprocess.Popen(
                command,
                cwd=str(repo_root),
                env=launch_env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
            )
        except Exception as exc:
            yield (
                f"[ERROR] Failed to launch training: {exc}\nCommand: {command_preview}\n",
                run_state,
                gr.update(interactive=True),
                gr.update(interactive=False),
            )
            return

        log_queue: queue.Queue[str] = queue.Queue()
        reader = threading.Thread(target=_reader_thread, args=(process, log_queue), daemon=True)
        reader.start()
        run_state["running"] = True
        run_state["process"] = process
        run_state["log_queue"] = log_queue
        run_state["reader"] = reader

    resolved_info = f"[INFO] Using data_path: {resolved_data_path}\n"
    debug_info = _colmap_path_debug(resolved_data_path) if any(token in config_name for token in ("colmap", "cusfm")) else ""
    info_prefix = f"{data_path_info}\n" if data_path_info else ""
    log_text = f"{info_prefix}{gpu_message}\n$ {command_preview}\n\n"
    log_text = f"{resolved_info}{debug_info}\n{log_text}"
    yield log_text, run_state, gr.update(interactive=False), gr.update(interactive=True)
    while True:
        try:
            line = log_queue.get(timeout=0.2)
            log_text = _append_log(log_text, line)
            yield log_text, run_state, gr.update(interactive=False), gr.update(interactive=True)
        except queue.Empty:
            if process.poll() is not None:
                while not log_queue.empty():
                    log_text = _append_log(log_text, log_queue.get_nowait())
                break

    log_text = _append_log(log_text, f"\n[INFO] Training finished with exit code {process.returncode}.\n")
    with PROCESS_LOCK:
        run_state["running"] = False
        run_state["process"] = None
        run_state["reader"] = None
        run_state["log_queue"] = None
    yield log_text, run_state, gr.update(interactive=True), gr.update(interactive=False)


def kill_training(run_state: dict[str, Any], current_log: str):
    with PROCESS_LOCK:
        process = run_state.get("process")
        if not run_state.get("running") or process is None:
            return _append_log(current_log, "[INFO] No running process to kill.\n"), run_state, gr.update(
                interactive=True
            ), gr.update(interactive=False)
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
        run_state["running"] = False
        run_state["process"] = None
        run_state["reader"] = None
        run_state["log_queue"] = None
    return _append_log(current_log, "[INFO] Kill signal sent. Process terminated.\n"), run_state, gr.update(
        interactive=True
    ), gr.update(interactive=False)


def save_preset(
    preset_name: str,
    repo_dir: str,
    data_path: str,
    experiment_name: str,
    mode: str,
    config_name: str,
    out_dir: str,
    resume_checkpoint: str,
    downsample: int,
    iterations: int,
    optimizer: str,
    export_usdz: bool,
    gpu_selection: str,
    conda_env: str,
    current_log: str,
):
    if not preset_name.strip():
        return _append_log(current_log, "[ERROR] Preset name is required.\n"), gr.update(choices=_preset_choices())
    presets = _load_presets()
    presets[preset_name.strip()] = {
        "repo_dir": repo_dir,
        "data_path": data_path,
        "experiment_name": experiment_name,
        "mode": mode,
        "config_name": config_name,
        "out_dir": out_dir,
        "resume_checkpoint": resume_checkpoint,
        "downsample": int(downsample),
        "iterations": int(iterations),
        "optimizer": optimizer,
        "export_usdz": bool(export_usdz),
        "gpu_selection": gpu_selection,
        "conda_env": conda_env,
    }
    _save_presets(presets)
    return _append_log(current_log, f"[INFO] Saved preset '{preset_name.strip()}'.\n"), gr.update(
        choices=_preset_choices(), value=preset_name.strip()
    )


def load_preset(selected_preset: str, current_log: str):
    presets = _load_presets()
    if not selected_preset or selected_preset not in presets:
        return (
            current_log,
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
        )
    preset = presets[selected_preset]
    mode = preset.get("mode", "3DGUT")
    mode_choices = MODE_CONFIG_CHOICES.get(mode, MODE_CONFIG_CHOICES["3DGUT"])
    config_value = preset.get("config_name", mode_choices[0])
    if config_value not in mode_choices:
        config_value = mode_choices[0]
    return (
        _append_log(current_log, f"[INFO] Loaded preset '{selected_preset}'.\n"),
        gr.update(value=preset.get("repo_dir", str(DEFAULT_REPO_DIR))),
        gr.update(value=preset.get("data_path", "")),
        gr.update(value=preset.get("experiment_name", "")),
        gr.update(value=mode),
        gr.update(choices=mode_choices, value=config_value),
        gr.update(value=preset.get("out_dir", "runs")),
        gr.update(value=preset.get("resume_checkpoint", "")),
        gr.update(value=int(preset.get("downsample", 2))),
        gr.update(value=int(preset.get("iterations", 30000))),
        gr.update(value=preset.get("optimizer", "selective_adam")),
        gr.update(value=bool(preset.get("export_usdz", False))),
        gr.update(value=preset.get("gpu_selection", "auto")),
        gr.update(value=preset.get("conda_env", "")),
    )


def on_video_selected_bounds(video_path_text: str, video_file: Any, current_log: str):
    # Use explicit local path as the source of truth to avoid corrupted temp uploads.
    if not video_path_text or not video_path_text.strip():
        upload_path = _extract_video_path(video_file)
        msg = "[INFO] Provide a local video path in the textbox (preferred and reliable)."
        if upload_path:
            msg += " Upload preview detected, but metadata probing from upload is skipped."
        return (
            _append_log(current_log, msg + "\n"),
            gr.update(value="No reliable local path selected yet."),
            gr.update(maximum=1.0, value=0.0),
            gr.update(maximum=1.0, value=1.0),
        )
    video_path = video_path_text.strip()
    path = Path(video_path).expanduser()
    if not path.exists():
        return (
            _append_log(current_log, f"[ERROR] Video file does not exist: {path}\n"),
            gr.update(value="No video selected."),
            gr.update(),
            gr.update(),
        )
    try:
        duration = _probe_video_duration_seconds(path)
    except Exception as exc:
        extra_hint = ""
        if "gradio" in str(path).lower():
            extra_hint = " Try using the direct local file path field instead of upload."
        return (
            _append_log(current_log, f"[ERROR] Could not inspect video duration: {exc}.{extra_hint}\n"),
            gr.update(value="Could not read video duration."),
            gr.update(),
            gr.update(),
        )
    return (
        _append_log(current_log, f"[INFO] Loaded video: {path}\n"),
        gr.update(value=f"Selected video: {path.name} | Duration: {duration:.2f} s"),
        gr.update(maximum=duration, value=0.0),
        gr.update(maximum=duration, value=duration),
    )


def sync_start_end(start_second: float, end_second: float):
    start = float(start_second)
    end = float(end_second)
    if start > end:
        end = start
    return gr.update(value=start), gr.update(value=end)


def sync_end_start(start_second: float, end_second: float):
    start = float(start_second)
    end = float(end_second)
    if end < start:
        start = end
    return gr.update(value=start), gr.update(value=end)


def set_start_from_marker(marker_second: float, end_second: float):
    marker = max(0.0, float(marker_second))
    end = float(end_second)
    start = marker
    if start > end:
        end = start
    return gr.update(value=start), gr.update(value=end)


def set_end_from_marker(start_second: float, marker_second: float):
    start = float(start_second)
    marker = max(0.0, float(marker_second))
    end = marker
    if end < start:
        start = end
    return gr.update(value=start), gr.update(value=end)


def on_video_selected_end_only(video_file: Any, current_log: str):
    video_path = _extract_video_path(video_file)
    if not video_path:
        return _append_log(current_log, "[INFO] Cleared video selection.\n"), gr.update(value="No video selected."), gr.update(
            maximum=0.0, value=0.0
        )
    path = Path(video_path).expanduser()
    if not path.exists():
        return _append_log(current_log, f"[ERROR] Video file does not exist: {path}\n"), gr.update(
            value="No video selected."
        ), gr.update()
    try:
        duration = _probe_video_duration_seconds(path)
    except Exception as exc:
        return _append_log(current_log, f"[ERROR] Could not inspect video duration: {exc}\n"), gr.update(
            value="Could not read video duration."
        ), gr.update()
    return _append_log(current_log, f"[INFO] Loaded video: {path}\n"), gr.update(
        value=f"Selected video: {path.name} | Duration: {duration:.2f} s"
    ), gr.update(maximum=duration, value=duration)


def generate_stills(
    video_path_text: str,
    video_file: Any,
    time_range: tuple[float, float] | list[float],
    images_per_second: float,
    stills_out_dir: str,
    current_log: str,
):
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        return _append_log(current_log, "[ERROR] ffmpeg/ffprobe not found in PATH.\n"), []
    if not video_path_text or not video_path_text.strip():
        return _append_log(current_log, "[ERROR] Please provide a local MP4 path in the video path textbox.\n"), []
    video_path = video_path_text.strip()
    source = Path(video_path).expanduser()
    if not source.exists() or source.suffix.lower() != ".mp4":
        return _append_log(current_log, "[ERROR] Invalid MP4 path.\n"), []
    if not time_range or len(time_range) != 2:
        return _append_log(current_log, "[ERROR] Invalid range selection.\n"), []
    try:
        duration = _probe_video_duration_seconds(source)
    except Exception as exc:
        return _append_log(current_log, f"[ERROR] Could not inspect video duration: {exc}\n"), []

    start = max(0.0, float(time_range[0]))
    end = min(float(time_range[1]), duration)
    if end <= start:
        return _append_log(current_log, "[ERROR] End second must be greater than start second.\n"), []
    if images_per_second <= 0:
        return _append_log(current_log, "[ERROR] images_per_second must be > 0.\n"), []

    target_root = Path(stills_out_dir).expanduser() if stills_out_dir.strip() else WRAPPER_DIR / "stills"
    target_dir = target_root / f"{source.stem}_stills"
    target_dir.mkdir(parents=True, exist_ok=True)
    output_pattern = target_dir / "frame_%06d.png"
    cmd = [
        "ffmpeg",
        "-y",
        "-ss",
        f"{start:.3f}",
        "-to",
        f"{end:.3f}",
        "-i",
        str(source),
        "-vf",
        f"fps={images_per_second}",
        str(output_pattern),
    ]
    preview = " ".join(shlex.quote(part) for part in cmd)
    log_text = _append_log(current_log, f"$ {preview}\n")
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.stderr:
        log_text = _append_log(log_text, result.stderr + "\n")
    if result.returncode != 0:
        return _append_log(log_text, f"[ERROR] Still extraction failed (exit code {result.returncode}).\n"), []
    images = sorted(str(p) for p in target_dir.glob("frame_*.png"))
    return _append_log(log_text, f"[INFO] Extracted {len(images)} stills to: {target_dir}\n"), images


def generate_stills_from_bounds(
    video_path_text: str,
    video_file: Any,
    start_second: float,
    end_second: float,
    images_per_second: float,
    stills_out_dir: str,
    current_log: str,
):
    return generate_stills(
        video_path_text,
        video_file,
        (start_second, end_second),
        images_per_second,
        stills_out_dir,
        current_log,
    )


def generate_stills_end_only(
    video_path_text: str,
    video_file: Any,
    end_second: float,
    images_per_second: float,
    stills_out_dir: str,
    current_log: str,
):
    return generate_stills(video_path_text, video_file, (0.0, end_second), images_per_second, stills_out_dir, current_log)


with gr.Blocks(title="3DGRUT Trainer UI (External Wrapper)") as demo:
    gr.Markdown("## 3DGRUT Web UI (External Wrapper)")
    run_state = gr.State({"running": False, "process": None, "reader": None, "log_queue": None})

    repo_dir = gr.Textbox(label="3DGRUT repo path", value=str(DEFAULT_REPO_DIR))
    with gr.Row():
        splat_base_dir = gr.Textbox(label="Splatter sessions base dir", value=str(DEFAULT_SPLAT_BASE_DIR), scale=2)
        splat_selector = gr.Dropdown(label="Splat session", choices=_splat_choices(str(DEFAULT_SPLAT_BASE_DIR)), value=None, scale=2)
        refresh_splats_button = gr.Button("Refresh Splats", scale=1)
        load_splat_button = gr.Button("Load Splat Session", scale=1)
    splat_status = gr.Markdown("No splat loaded.")
    with gr.Row():
        data_path = gr.Textbox(label="data_path", placeholder=r"C:\path\to\dataset", value="", scale=2)
        experiment_name = gr.Textbox(label="experiment_name", placeholder="lego_3dgut", value="", scale=1)
    mode = gr.Radio(choices=["3DGUT", "3DGRT"], value="3DGUT", label="Model Config Toggle")
    config_name = gr.Dropdown(
        label="Config preset",
        choices=MODE_CONFIG_CHOICES["3DGUT"],
        value=MODE_CONFIG_CHOICES["3DGUT"][0],
    )
    with gr.Row():
        out_dir = gr.Textbox(label="out_dir", value="runs", scale=1)
        resume_checkpoint = gr.Textbox(label="resume checkpoint (optional)", value="", scale=2)
    with gr.Row():
        downsample = gr.Slider(label="downsample", minimum=1, maximum=8, step=1, value=2)
        iterations = gr.Number(label="iterations", value=30000, precision=0)
        optimizer = gr.Dropdown(label="optimizer", choices=["selective_adam", "adam"], value="selective_adam")
        export_usdz = gr.Checkbox(label="export_usdz", value=False)
    with gr.Row():
        gpu_selection = gr.Dropdown(
            label="GPU selection",
            choices=_gpu_choices(),
            value="auto",
            allow_custom_value=False,
            scale=3,
        )
        refresh_gpus_button = gr.Button("Refresh GPUs", scale=1)
    gpu_status = gr.Markdown("GPU status: Auto will choose the card with the most free VRAM.")
    conda_env = gr.Textbox(label="Conda env name (optional)", value="")
    with gr.Row():
        start_button = gr.Button("Start Training", variant="primary")
        kill_button = gr.Button("Kill Process", variant="stop", interactive=False)
    with gr.Row():
        preset_name = gr.Textbox(label="Save As preset", value="", scale=1)
        save_preset_button = gr.Button("Save Preset")
        preset_selector = gr.Dropdown(label="Saved presets", choices=_preset_choices(), value=None, scale=1)
        load_preset_button = gr.Button("Load Preset")
    log_output = gr.Code(label="Execution Console", value="", language="shell", lines=24, interactive=False)

    gr.Markdown("Stills extraction moved to `stills_extractor_app.py` to keep this training wrapper focused.")

    start_button.click(
        fn=start_training,
        inputs=[
            repo_dir,
            data_path,
            experiment_name,
            mode,
            config_name,
            out_dir,
            resume_checkpoint,
            downsample,
            iterations,
            optimizer,
            export_usdz,
            gpu_selection,
            conda_env,
            run_state,
        ],
        outputs=[log_output, run_state, start_button, kill_button],
    )
    mode.change(fn=_update_config_choices, inputs=[mode], outputs=[config_name])
    save_preset_button.click(
        fn=save_preset,
        inputs=[
            preset_name,
            repo_dir,
            data_path,
            experiment_name,
            mode,
            config_name,
            out_dir,
            resume_checkpoint,
            downsample,
            iterations,
            optimizer,
            export_usdz,
            gpu_selection,
            conda_env,
            log_output,
        ],
        outputs=[log_output, preset_selector],
    )
    load_preset_button.click(
        fn=load_preset,
        inputs=[preset_selector, log_output],
        outputs=[
            log_output,
            repo_dir,
            data_path,
            experiment_name,
            mode,
            config_name,
            out_dir,
            resume_checkpoint,
            downsample,
            iterations,
            optimizer,
            export_usdz,
            gpu_selection,
            conda_env,
        ],
    )
    refresh_splats_button.click(
        fn=refresh_splats,
        inputs=[splat_base_dir, log_output],
        outputs=[log_output, splat_selector],
    )
    load_splat_button.click(
        fn=load_splat_session,
        inputs=[splat_base_dir, splat_selector, log_output],
        outputs=[log_output, data_path, experiment_name, out_dir, splat_status],
    )
    refresh_gpus_button.click(
        fn=refresh_gpu_list,
        inputs=[gpu_selection, log_output],
        outputs=[log_output, gpu_selection, gpu_status],
    )
    kill_button.click(
        fn=kill_training,
        inputs=[run_state, log_output],
        outputs=[log_output, run_state, start_button, kill_button],
    )


if __name__ == "__main__":
    demo.queue().launch()
