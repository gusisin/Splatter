from __future__ import annotations

import os
import subprocess
from pathlib import Path

import gradio as gr

import gui_wrapper
import stills_extractor_app

APP_DIR = Path(__file__).resolve().parent
DEFAULT_SPLAT_BASE_DIR = Path.home() / "Pictures" / "Splatter"
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}
VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".webm"}


def _discover_splats(base_dir: str) -> list[str]:
    base = Path(base_dir).expanduser()
    if not base.exists():
        return []
    return sorted([p.name for p in base.iterdir() if p.is_dir() and (p / "manifest.json").exists()])


def _run_dirs_for_splat(base_dir: str, splat_name: str) -> list[Path]:
    runs_root = Path(base_dir).expanduser() / splat_name / "runs"
    if not runs_root.exists():
        return []
    return sorted([p for p in runs_root.rglob("*") if p.is_dir()], key=lambda p: p.stat().st_mtime, reverse=True)


def _viewable_assets(run_dir: Path) -> list[Path]:
    candidates: list[Path] = []
    for p in run_dir.rglob("*"):
        if not p.is_file():
            continue
        ext = p.suffix.lower()
        if ext in IMAGE_EXTS or ext in VIDEO_EXTS:
            candidates.append(p)
    return sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True)


def _open_path(path: Path) -> str:
    if not path.exists():
        return f"[ERROR] Path does not exist: {path}"
    if os.name == "nt":
        os.startfile(str(path))  # type: ignore[attr-defined]
        return f"[INFO] Opened: {path}"
    subprocess.run(["xdg-open", str(path)], check=False)
    return f"[INFO] Opened: {path}"


def refresh_display_splats(base_dir: str):
    splats = _discover_splats(base_dir)
    first = splats[0] if splats else None
    msg = f"Found {len(splats)} splat session(s) in `{Path(base_dir).expanduser()}`."
    return gr.update(choices=splats, value=first), msg


def refresh_runs(base_dir: str, splat_name: str):
    if not splat_name:
        return gr.update(choices=[], value=None), "Select a splat first."
    run_dirs = _run_dirs_for_splat(base_dir, splat_name)
    labels = [str(p.relative_to(Path(base_dir).expanduser() / splat_name)) for p in run_dirs]
    first = labels[0] if labels else None
    if not labels:
        return gr.update(choices=[], value=None), f"No run folders found under `{splat_name}/runs`."
    return gr.update(choices=labels, value=first), f"Found {len(labels)} run folder(s) for `{splat_name}`."


def load_run_assets(base_dir: str, splat_name: str, run_label: str):
    if not splat_name or not run_label:
        return [], "Select a run folder to preview assets."
    run_dir = Path(base_dir).expanduser() / splat_name / run_label
    if not run_dir.exists():
        return [], f"[ERROR] Run folder not found: {run_dir}"
    assets = _viewable_assets(run_dir)
    gallery_items = [str(p) for p in assets[:100]]
    status = (
        f"Run: `{run_dir}`\n\n"
        f"- Found {len(assets)} preview asset(s)\n"
        f"- Showing up to 100 in gallery"
    )
    return gallery_items, status


def open_run_folder(base_dir: str, splat_name: str, run_label: str):
    if not splat_name or not run_label:
        return "[ERROR] Select a run folder first."
    run_dir = Path(base_dir).expanduser() / splat_name / run_label
    return _open_path(run_dir)


def open_splat_folder(base_dir: str, splat_name: str):
    if not splat_name:
        return "[ERROR] Select a splat first."
    splat_dir = Path(base_dir).expanduser() / splat_name
    return _open_path(splat_dir)


GRUT_REPO_DIR = APP_DIR / "tools" / "3dgrut"
GRUT_VENV_PYTHON = (
    GRUT_REPO_DIR / ".venv" / "Scripts" / "python.exe"
    if os.name == "nt"
    else GRUT_REPO_DIR / ".venv" / "bin" / "python"
)
GRUT_PLAYGROUND = GRUT_REPO_DIR / "playground.py"
SPLATTER_EXPORT_PLY = APP_DIR / "splatter_export_ply.py"


def _resolve_run_dir(base_dir: str, splat_name: str, run_label: str) -> Path | None:
    if not splat_name or not run_label:
        return None
    candidate = Path(base_dir).expanduser() / splat_name / run_label
    return candidate if candidate.is_dir() else None


def _find_latest_checkpoint(run_dir: Path) -> Path | None:
    """Pick the most useful .pt in (or under) a run folder.

    Order of preference:
      1. <run>/ckpt_last.pt  (3DGRUT writes this at end of training)
      2. The highest-numbered ckpt_<N>.pt found recursively (e.g. ours_30000/ckpt_30000.pt)
    """
    last = run_dir / "ckpt_last.pt"
    if last.is_file():
        return last
    numbered: list[tuple[int, Path]] = []
    for p in run_dir.rglob("ckpt_*.pt"):
        if not p.is_file():
            continue
        stem = p.stem  # e.g. "ckpt_30000"
        suffix = stem.split("_", 1)[1] if "_" in stem else ""
        try:
            n = int(suffix)
        except ValueError:
            continue
        numbered.append((n, p))
    if not numbered:
        return None
    numbered.sort(key=lambda t: t[0], reverse=True)
    return numbered[0][1]


def _subprocess_env_for_grut() -> dict[str, str]:
    """Mirror gui_wrapper: UTF-8 + MSVC bin + 3DGRUT venv Scripts on PATH."""
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    venv_scripts = GRUT_VENV_PYTHON.parent
    env["PATH"] = str(venv_scripts) + os.pathsep + env.get("PATH", "")
    msvc_dir = gui_wrapper._find_msvc_cl_path()
    if msvc_dir is not None:
        env["PATH"] = str(msvc_dir) + os.pathsep + env["PATH"]
    return env


def open_in_grut_viewer(base_dir: str, splat_name: str, run_label: str):
    if not GRUT_VENV_PYTHON.is_file():
        return f"[ERROR] 3DGRUT venv python not found at {GRUT_VENV_PYTHON}. Run install.ps1."
    if not GRUT_PLAYGROUND.is_file():
        return f"[ERROR] playground.py not found at {GRUT_PLAYGROUND}."
    run_dir = _resolve_run_dir(base_dir, splat_name, run_label)
    if run_dir is None:
        return "[ERROR] Select a splat and run folder first."
    ckpt = _find_latest_checkpoint(run_dir)
    if ckpt is None:
        return f"[ERROR] No ckpt_*.pt found under {run_dir}."

    cmd = [str(GRUT_VENV_PYTHON), str(GRUT_PLAYGROUND), "--gs_object", str(ckpt)]
    try:
        # Detached so the GUI window survives even if Splatter is restarted.
        creationflags = 0
        if os.name == "nt":
            # CREATE_NEW_PROCESS_GROUP | DETACHED_PROCESS so closing the parent
            # shell doesn't kill the playground window.
            creationflags = 0x00000200 | 0x00000008
        subprocess.Popen(
            cmd,
            cwd=str(GRUT_REPO_DIR),
            env=_subprocess_env_for_grut(),
            creationflags=creationflags,
            close_fds=True,
        )
    except Exception as exc:
        return f"[ERROR] Failed to launch playground: {exc}"
    return (
        f"[INFO] Launched 3DGRUT Viewer for {ckpt.name}.\n"
        "Look for a new native window titled 'Polyscope' (Alt+Tab if hidden).\n"
        "First launch may take ~30-90s to JIT compile kernels."
    )


def export_run_to_ply(base_dir: str, splat_name: str, run_label: str):
    """Subprocess-export the latest checkpoint of the selected run to PLY.

    Returns: (gr.update for the gr.Model3D viewer, status_text, gr.update for the download file).
    """
    no_ply = gr.update(value=None, visible=False)
    no_file = gr.update(value=None, visible=False)
    if not GRUT_VENV_PYTHON.is_file():
        return no_ply, f"[ERROR] 3DGRUT venv python not found at {GRUT_VENV_PYTHON}.", no_file
    if not SPLATTER_EXPORT_PLY.is_file():
        return no_ply, f"[ERROR] Missing helper script: {SPLATTER_EXPORT_PLY}.", no_file
    run_dir = _resolve_run_dir(base_dir, splat_name, run_label)
    if run_dir is None:
        return no_ply, "[ERROR] Select a splat and run folder first.", no_file
    ckpt = _find_latest_checkpoint(run_dir)
    if ckpt is None:
        return no_ply, f"[ERROR] No ckpt_*.pt found under {run_dir}.", no_file

    out_ply = run_dir / "point_cloud.ply"
    cmd = [
        str(GRUT_VENV_PYTHON),
        str(SPLATTER_EXPORT_PLY),
        "--checkpoint",
        str(ckpt),
        "--output",
        str(out_ply),
    ]
    try:
        result = subprocess.run(
            cmd,
            cwd=str(GRUT_REPO_DIR),
            env=_subprocess_env_for_grut(),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=600,
        )
    except subprocess.TimeoutExpired:
        return (
            no_ply,
            "[ERROR] PLY export timed out after 10 minutes. Checkpoint may be very large.",
            no_file,
        )
    except Exception as exc:
        return no_ply, f"[ERROR] PLY export failed to launch: {exc}", no_file

    log_tail = (result.stdout or "") + (result.stderr or "")
    if result.returncode != 0 or not out_ply.is_file():
        return (
            no_ply,
            f"[ERROR] PLY export exited with code {result.returncode}.\n\n{log_tail.strip()}",
            no_file,
        )

    size_mb = out_ply.stat().st_size / (1024 * 1024)
    status = (
        f"[INFO] Wrote {out_ply} ({size_mb:.1f} MB) from `{ckpt.name}`.\n\n"
        "Note: `gr.Model3D` renders PLY as opaque points, not true Gaussian splats. "
        "For the photoreal splat view use the *Open in 3DGRUT Viewer* button."
    )
    return (
        gr.update(value=str(out_ply), visible=True),
        status,
        gr.update(value=str(out_ply), visible=True),
    )


with gr.Blocks(title="Splatter Unified App") as demo:
    gr.Markdown("## Splatter")
    gr.Markdown(
        "Use one app for the full workflow: extract stills, build COLMAP, train splats, and view run outputs."
    )

    with gr.Tabs():
        with gr.Tab("Produce Stills"):
            stills_extractor_app.demo.render()

        with gr.Tab("Build COLMAP Dataset"):
            stills_extractor_app.colmap_demo.render()

        with gr.Tab("Train Splat"):
            gui_wrapper.demo.render()

        with gr.Tab("Display Splat"):
            gr.Markdown("### Display / View Splat")
            gr.Markdown(
                "Browse splat sessions and run folders, then preview generated images/videos.\n\n"
                "This is a practical v1 browser/launcher while an in-app realtime viewer is being built."
            )
            with gr.Row():
                display_base_dir = gr.Textbox(
                    label="Splatter sessions base dir",
                    value=str(DEFAULT_SPLAT_BASE_DIR),
                    scale=2,
                )
                display_refresh_splats_btn = gr.Button("Refresh Splats", scale=1)
                display_refresh_runs_btn = gr.Button("Refresh Runs", scale=1)
            with gr.Row():
                display_splat_selector = gr.Dropdown(label="Splat session", choices=[], value=None, scale=1)
                display_run_selector = gr.Dropdown(label="Run folder", choices=[], value=None, scale=2)
            with gr.Row():
                open_splat_btn = gr.Button("Open Splat Folder")
                open_run_btn = gr.Button("Open Run Folder")
            with gr.Row():
                grut_viewer_btn = gr.Button("Open in 3DGRUT Viewer", variant="primary")
                export_ply_btn = gr.Button("Export PLY (inline preview)")
            display_status = gr.Markdown("Select a splat and run folder to preview assets.")
            display_gallery = gr.Gallery(label="Preview assets", columns=5, height=360)

            ply_status = gr.Markdown("")
            ply_viewer = gr.Model3D(
                label="Splat point cloud (PLY)",
                clear_color=[0.05, 0.05, 0.08, 1.0],
                height=480,
                interactive=False,
                visible=False,
            )
            ply_download = gr.File(label="Download PLY", visible=False, interactive=False)

            display_open_status = gr.Code(label="Display console", value="", language="shell", interactive=False, lines=6)

            display_refresh_splats_btn.click(
                fn=refresh_display_splats,
                inputs=[display_base_dir],
                outputs=[display_splat_selector, display_status],
            )
            display_refresh_runs_btn.click(
                fn=refresh_runs,
                inputs=[display_base_dir, display_splat_selector],
                outputs=[display_run_selector, display_status],
            )
            display_splat_selector.change(
                fn=refresh_runs,
                inputs=[display_base_dir, display_splat_selector],
                outputs=[display_run_selector, display_status],
            )
            display_run_selector.change(
                fn=load_run_assets,
                inputs=[display_base_dir, display_splat_selector, display_run_selector],
                outputs=[display_gallery, display_status],
            )
            open_splat_btn.click(
                fn=open_splat_folder,
                inputs=[display_base_dir, display_splat_selector],
                outputs=[display_open_status],
            )
            open_run_btn.click(
                fn=open_run_folder,
                inputs=[display_base_dir, display_splat_selector, display_run_selector],
                outputs=[display_open_status],
            )
            grut_viewer_btn.click(
                fn=open_in_grut_viewer,
                inputs=[display_base_dir, display_splat_selector, display_run_selector],
                outputs=[display_open_status],
            )
            export_ply_btn.click(
                fn=export_run_to_ply,
                inputs=[display_base_dir, display_splat_selector, display_run_selector],
                outputs=[ply_viewer, ply_status, ply_download],
            )


if __name__ == "__main__":
    # Allow Gradio to serve generated stills/run outputs from user-selected
    # session directories (e.g. ~/Pictures/Splatter) and from the app folder.
    demo.queue().launch(
        server_name="127.0.0.1",
        server_port=8000,
        allowed_paths=[str(APP_DIR), str(Path.home())],
        css=stills_extractor_app.SPLATTER_CSS,
    )

