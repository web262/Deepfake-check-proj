from __future__ import annotations
import numpy as np
import torch
import torch.nn.functional as F
import cv2
from PIL import Image

class GradCAM:
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self.h1 = target_layer.register_forward_hook(self._forward_hook)
        self.h2 = target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, inp, out):
        self.activations = out.detach()

    def _backward_hook(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def close(self):
        self.h1.remove()
        self.h2.remove()

    def __call__(self, x: torch.Tensor) -> np.ndarray:
        """
        Returns CAM as numpy array in [0,1], shape (H, W)
        """
        self.model.zero_grad(set_to_none=True)
        logit = self.model(x)  # (B,1)
        score = logit[:, 0].sum()
        score.backward(retain_graph=True)

        A = self.activations  # (B,C,h,w)
        G = self.gradients    # (B,C,h,w)
        weights = G.mean(dim=(2, 3), keepdim=True)  # (B,C,1,1)
        cam = (weights * A).sum(dim=1, keepdim=True)  # (B,1,h,w)
        cam = F.relu(cam)

        cam = cam.squeeze(0).squeeze(0)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        return cam.cpu().numpy()

def overlay_heatmap(pil_img: Image.Image, cam01: np.ndarray, alpha: float = 0.45) -> Image.Image:
    """
    cam01: [0,1] heatmap (h,w) from model feature map space
    """
    img = np.array(pil_img.convert("RGB"))
    H, W = img.shape[:2]
    cam = cv2.resize(cam01, (W, H))
    heat = (255 * cam).astype(np.uint8)
    heat = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    out = (img * (1 - alpha) + heat * alpha).astype(np.uint8)
    return Image.fromarray(out)
