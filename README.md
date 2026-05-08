# Splatter

Windows-first tooling around NVIDIA 3DGRUT for:

- Extracting still frames from videos
- Preparing COLMAP datasets
- Launching 3DGRUT training from a GUI wrapper
- Running the full workflow from a single unified app

This repository keeps `3dgrut` itself untouched and provides an orchestration layer in a separate project.

## Current Scope

- `stills_extractor_app.py`
  - Create "splat" sessions
  - Extract stills from one or more videos
  - Prepare COLMAP sparse reconstruction (`dataset/sparse/0`)
  - Persist session metadata to `manifest.json`
- `gui_wrapper.py`
  - Launch 3DGRUT training from a Gradio UI
  - Load session manifests and auto-resolve COLMAP dataset paths
  - Show training logs and manage runs
- `install.ps1`
  - One-shot bootstrap script for Windows users
  - Creates local venv, installs GUI dependencies, clones and bootstraps `3dgrut`
  - Validates that a compatible NVIDIA GPU is present
- `splatter_app.py`
  - Single app entrypoint with tabs for:
    - Produce stills + COLMAP files
    - Train splat
    - Display/View splat outputs (run browser + preview gallery + open-folder actions)

## Requirements

- Windows 10/11
- NVIDIA GPU (RTX-class recommended for 3DGRUT kernels)
- NVIDIA driver with CUDA support
- Python 3.10+
- Git
- PowerShell

Notes:

- COLMAP and ffmpeg are used in the stills/COLMAP workflow.
- `install.ps1` checks and warns for missing runtime components.

## GPU Compatibility

3DGRUT kernel builds in this workflow are intended for newer NVIDIA architectures.

| GPU class | Expected result |
| --- | --- |
| RTX 20/30/40/50 series | Supported path for training |
| GTX 10 series (for example GTX 1060) | Likely fails with `cudaErrorNoKernelImageForDevice` |
| No NVIDIA GPU | Not supported |

If your GPU is not in a supported class, install will stop early with guidance.

## First 5 Minutes

1. Open PowerShell in the repository root.
2. Run:
   ```powershell
   Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
   .\install.ps1
   ```
3. Activate the Splatter environment:
   ```powershell
   .\.venv-splatter\Scripts\Activate.ps1
   ```
4. Start the unified app:
   ```powershell
   python .\splatter_app.py
   ```
5. Use:
   - **Produce Stills and COLMAP Files** tab first
   - then **Train Splat** tab
   - then **Display Splat** tab to browse outputs

## Quick Start

From a PowerShell terminal in the repo root:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\install.ps1
```

After install, activate the Splatter venv and run the unified app:

```powershell
.\.venv-splatter\Scripts\Activate.ps1
python .\splatter_app.py
```

Optional advanced usage (run individual apps directly):

```powershell
.\.venv-splatter\Scripts\Activate.ps1
python .\stills_extractor_app.py
```

```powershell
.\.venv-splatter\Scripts\Activate.ps1
python .\gui_wrapper.py
```

## Typical Workflow

1. Open `splatter_app.py`
2. In **Produce Stills and COLMAP Files**, create a new splat session
3. Add source videos and generate stills
4. Run **Prepare COLMAP Dataset**
5. Switch to **Train Splat**
6. Load the splat session and start training
7. Switch to **Display Splat** to browse run outputs and preview generated assets

## Repository Layout

- `splatter_app.py` - unified app entrypoint (recommended)
- `gui_wrapper.py` - 3DGRUT training GUI wrapper
- `stills_extractor_app.py` - still extraction + COLMAP prep app
- `install.ps1` - Windows install/bootstrap script
- `requirements-gui.txt` - Python dependencies for GUI tooling
- `.gitignore` - local/runtime artifact exclusions

## Troubleshooting

- If training fails with COLMAP path errors, verify:
  - `.../dataset/images` exists
  - `.../dataset/sparse/0/images.bin` exists
- If training fails with CUDA/kernel compatibility errors:
  - Verify your GPU model and driver
  - Ensure your GPU architecture is supported by the 3DGRUT build/toolchain
- If ffmpeg/COLMAP commands are missing:
  - Install and ensure they are available on PATH (or use configured shims)

## Status

This project is actively evolving. Installation and runtime ergonomics are being improved with each iteration.
