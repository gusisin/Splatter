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


with gr.Blocks(title="Splatter Unified App") as demo:
    gr.Markdown("## Splatter")
    gr.Markdown(
        "Use one app for the full workflow: extract stills + COLMAP, train splats, and view run outputs."
    )

    with gr.Tabs():
        with gr.Tab("Produce Stills and COLMAP Files"):
            stills_extractor_app.demo.render()

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
            display_status = gr.Markdown("Select a splat and run folder to preview assets.")
            display_gallery = gr.Gallery(label="Preview assets", columns=5, height=360)
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


if __name__ == "__main__":
    demo.queue().launch()

