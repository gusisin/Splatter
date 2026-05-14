"""Export a 3DGRUT checkpoint to a PLY point cloud.

This script is invoked by Splatter as a subprocess **inside the 3DGRUT venv**
(``tools/3dgrut/.venv/Scripts/python.exe``), because it depends on the full
3DGRUT stack (PyTorch, Hydra config, ``MixtureOfGaussians``). It should NOT be
imported into Splatter's UI venv — the dependencies aren't there.

Usage (from the Splatter wrapper):

    .../tools/3dgrut/.venv/Scripts/python.exe splatter_export_ply.py \
        --checkpoint <run>/ckpt_last.pt \
        --output <run>/point_cloud.ply
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

from threedgrut.export import PLYExporter
from threedgrut.model.model import MixtureOfGaussians


def export_checkpoint_to_ply(checkpoint_path: str, output_path: str) -> int:
    ckpt_path = Path(checkpoint_path)
    out_path = Path(output_path)
    if not ckpt_path.is_file():
        print(f"[ERROR] Checkpoint not found: {ckpt_path}", flush=True)
        return 2

    print(f"[INFO] Loading checkpoint: {ckpt_path}", flush=True)
    # weights_only=False is required for 3DGRUT checkpoints (they hold numpy arrays + a Hydra config).
    checkpoint = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    if "config" not in checkpoint:
        print("[ERROR] Checkpoint missing 'config' key; cannot rebuild the model.", flush=True)
        return 3

    conf = checkpoint["config"]
    model = MixtureOfGaussians(conf, scene_extent=checkpoint.get("scene_extent"))
    model.init_from_checkpoint(checkpoint, setup_optimizer=False)
    model.eval()
    num_gaussians = int(model.get_positions().shape[0])
    print(f"[INFO] Model loaded: {num_gaussians} gaussians", flush=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Writing PLY to: {out_path}", flush=True)
    PLYExporter().export(model=model, output_path=out_path, conf=conf)
    if not out_path.is_file() or out_path.stat().st_size == 0:
        print("[ERROR] PLY exporter wrote no bytes.", flush=True)
        return 4
    print(
        f"[INFO] Done. PLY size: {out_path.stat().st_size:,} bytes, gaussians: {num_gaussians}",
        flush=True,
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Export a 3DGRUT checkpoint to a PLY point cloud.")
    parser.add_argument("--checkpoint", required=True, help="Path to ckpt_*.pt produced by 3DGRUT training.")
    parser.add_argument("--output", required=True, help="Destination .ply path.")
    args = parser.parse_args()
    return export_checkpoint_to_ply(args.checkpoint, args.output)


if __name__ == "__main__":
    sys.exit(main())
