#!/usr/bin/env python3
"""
Local inference with Depth Anything (relative depth + optional features).

Runs entirely on your machine — no server needed.
- CPU: works, slower.
- CUDA GPU: fast.
- Apple Silicon (MPS): use device='mps', dtype=torch.float32 (bfloat16 not supported on MPS).
"""

import argparse
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")  # Apple Silicon
    return torch.device("cpu")


def load_model(model_id: str = "LiheYoung/depth-anything-small-hf", device=None):
    """Load Depth Anything from Hugging Face. First run downloads weights (~100–300 MB)."""
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation

    device = device or get_device()
    # MPS: use float32; CUDA can use bfloat16 for speed
    dtype = torch.float32
    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        dtype = torch.bfloat16

    print(f"Loading model {model_id} on {device} ({dtype})...")
    processor = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModelForDepthEstimation.from_pretrained(model_id)
    model = model.to(device).to(dtype).eval()
    return processor, model, device, dtype


def infer_rel_depth(processor, model, image: Image.Image, device, dtype):
    """Return relative depth map (numpy, same aspect as input)."""
    inputs = processor(images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    if dtype in (torch.bfloat16, torch.float16):
        inputs = {k: v.to(dtype) if v.is_floating_point() else v for k, v in inputs.items()}

    with torch.no_grad():
        out = model(**inputs)
    pred = out.predicted_depth

    h, w = image.size[1], image.size[0]
    pred = torch.nn.functional.interpolate(
        pred.unsqueeze(1),
        size=(h, w),
        mode="bicubic",
        align_corners=False,
    )
    return pred.squeeze().cpu().float().numpy()


def main():
    parser = argparse.ArgumentParser(description="Depth Anything inference (local)")
    parser.add_argument("image", type=str, help="Path to RGB image or directory of images")
    parser.add_argument("--out-dir", type=str, default="./outputs/depth_anything", help="Output directory")
    parser.add_argument(
        "--model",
        type=str,
        default="LiheYoung/depth-anything-small-hf",
        choices=[
            "LiheYoung/depth-anything-small-hf",
            "LiheYoung/depth-anything-base-hf",
            "LiheYoung/depth-anything-large-hf",
        ],
        help="Model size: small (fast), base, large (best quality)",
    )
    parser.add_argument("--save-npy", action="store_true", help="Save raw depth as .npy")
    args = parser.parse_args()

    processor, model, device, dtype = load_model(args.model)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = []
    if os.path.isfile(args.image):
        paths = [args.image]
    elif os.path.isdir(args.image):
        paths = [
            os.path.join(args.image, f)
            for f in os.listdir(args.image)
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))
        ]
        paths.sort()
    if not paths:
        print("No images found.")
        return

    print(f"Processing {len(paths)} image(s) on {device}...")
    for i, path in enumerate(paths):
        image = Image.open(path).convert("RGB")
        rel_depth = infer_rel_depth(processor, model, image, device, dtype)
        name = Path(path).stem
        # Normalize for visualization (relative depth is scale/shift invariant)
        d_min, d_max = rel_depth.min(), rel_depth.max()
        if d_max > d_min:
            vis = (rel_depth - d_min) / (d_max - d_min)
        else:
            vis = np.zeros_like(rel_depth)
        vis = (vis * 255).astype(np.uint8)
        Image.fromarray(vis).save(out_dir / f"{name}_depth_vis.png")
        if args.save_npy:
            np.save(out_dir / f"{name}_depth_rel.npy", rel_depth)
        print(f"  {path} -> {out_dir / (name + '_depth_vis.png')}")

    print("Done. Depth Anything runs locally — no server required.")


if __name__ == "__main__":
    main()
