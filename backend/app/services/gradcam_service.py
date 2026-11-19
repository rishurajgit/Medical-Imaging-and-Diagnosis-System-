"""
Real Gradient-Weighted Class Activation Mapping (Grad-CAM) for PyTorch DenseNet-121.
Registers forward and full backward hooks on the final convolutional feature block
(`features.norm5`), computes global-average-pooled gradient weights, applies ReLU,
normalizes activation intensity to [0, 1], resizes to original scan dimensions,
and exports a transparent RGBA thermal heatmap PNG plus blended overlay PNG.

Note: Grad-CAM highlights convolutional layer sensitivity for a target class and does
NOT establish clinical causation.
"""

from pathlib import Path
from typing import Dict, Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from app.ml.classification.model import ChestXRayDenseNet121


def generate_gradcam_artifacts(
    model: ChestXRayDenseNet121,
    input_tensor: torch.Tensor,
    original_rgb: np.ndarray,
    target_class_index: int,
    device: torch.device,
    heatmap_output_path: Path,
    blended_output_path: Optional[Path] = None,
    alpha: float = 0.45,
) -> Dict:
    activations = []
    gradients = []

    target_layer = model.get_gradcam_target_layer()

    def forward_hook(_module, _inp, output):
        activations.append(output.detach())

    def backward_hook(_module, _grad_in, grad_out):
        gradients.append(grad_out[0].detach())

    fwd_handle = target_layer.register_forward_hook(forward_hook)
    bwd_handle = target_layer.register_full_backward_hook(backward_hook)

    try:
        model.eval()
        model.zero_grad(set_to_none=True)
        tensor_in = input_tensor.to(device).requires_grad_(True)
        logits = model(tensor_in)
        score = logits[0, target_class_index]
        score.backward()

        if not activations or not gradients:
            raise RuntimeError("Grad-CAM hooks did not capture feature activations or gradients.")

        act = activations[0]  # [1, C, h, w]
        grad = gradients[0]   # [1, C, h, w]
        weights = grad.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * act).sum(dim=1, keepdim=True))
        cam_np = cam.squeeze().cpu().numpy()

        cam_min, cam_max = float(cam_np.min()), float(cam_np.max())
        if cam_max - cam_min > 1e-8:
            cam_norm = (cam_np - cam_min) / (cam_max - cam_min)
        else:
            cam_norm = np.zeros_like(cam_np, dtype=np.float32)

        orig_h, orig_w = original_rgb.shape[:2]
        cam_resized = cv2.resize(cam_norm.astype(np.float32), (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)
        cam_resized = np.clip(cam_resized, 0.0, 1.0)

        # Convert normalized CAM into JET RGB and transparent RGBA PNG
        cam_uint8 = (cam_resized * 255.0).astype(np.uint8)
        bgr_heat = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)
        rgb_heat = cv2.cvtColor(bgr_heat, cv2.COLOR_BGR2RGB)

        alpha_channel = np.where(cam_resized < 0.12, 0, (cam_resized * 230.0)).astype(np.uint8)
        rgba_heat = np.dstack([rgb_heat, alpha_channel])

        heatmap_output_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(rgba_heat, mode="RGBA").save(heatmap_output_path, format="PNG")

        if blended_output_path is not None:
            blended_output_path.parent.mkdir(parents=True, exist_ok=True)
            blended = (
                original_rgb.astype(np.float32) * (1.0 - alpha * cam_resized[..., None])
                + rgb_heat.astype(np.float32) * (alpha * cam_resized[..., None])
            )
            blended_uint8 = np.clip(blended, 0, 255).astype(np.uint8)
            Image.fromarray(blended_uint8, mode="RGB").save(blended_output_path, format="PNG")

        peak_y, peak_x = np.unravel_index(int(np.argmax(cam_resized)), cam_resized.shape)
        return {
            "heatmap_path": str(heatmap_output_path),
            "blended_path": str(blended_output_path) if blended_output_path else None,
            "peak_normalized_xy": [round(float(peak_x / orig_w), 4), round(float(peak_y / orig_h), 4)],
            "max_activation": round(float(cam_max), 6),
            "disclaimer": "Grad-CAM highlights model gradient sensitivity and does not establish clinical causation.",
        }
    finally:
        fwd_handle.remove()
        bwd_handle.remove()
