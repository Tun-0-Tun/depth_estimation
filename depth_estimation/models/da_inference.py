import numpy as np
import torch
from PIL import Image


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_da_model(
    model_id: str = "depth-anything/Depth-Anything-V2-Small-hf",
    device: torch.device | None = None,
):
    """
    Load Depth Anything v2 from Hugging Face and return (processor, model, device, dtype).
    """
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation

    device = device or get_device()
    dtype = torch.float32
    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        dtype = torch.bfloat16

    print(f"Loading Depth Anything V2 {model_id} on {device} ({dtype})...")
    processor = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModelForDepthEstimation.from_pretrained(model_id)
    model = model.to(device).to(dtype).eval()
    return processor, model, device, dtype


def infer_depth(
    processor,
    model,
    image: Image.Image,
    device: torch.device,
    dtype: torch.dtype,
) -> np.ndarray:
    """
    Run Depth Anything V2 on a single RGB image and return depth as float32 numpy array.
    """
    inputs = processor(images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    if dtype in (torch.bfloat16, torch.float16):
        inputs = {k: v.to(dtype) if v.is_floating_point() else v for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
    pred = outputs.predicted_depth  # (1, H', W')

    h, w = image.size[1], image.size[0]
    pred = torch.nn.functional.interpolate(
        pred.unsqueeze(1),
        size=(h, w),
        mode="bicubic",
        align_corners=False,
    )
    depth = pred.squeeze().cpu().float().numpy()
    return depth


def depth_to_vis(depth: np.ndarray) -> Image.Image:
    """
    Normalize depth map to 0–255 uint8 for quick visualization.
    """
    d_min, d_max = float(depth.min()), float(depth.max())
    if d_max > d_min:
        vis = (depth - d_min) / (d_max - d_min)
    else:
        vis = np.zeros_like(depth)
    vis = (vis * 255).astype(np.uint8)
    return Image.fromarray(vis)

