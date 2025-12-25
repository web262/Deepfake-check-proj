from __future__ import annotations
import os
import torch
from collections import OrderedDict
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

class DeepfakeDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        in_features = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(in_features, 1)  # binary logit

    def forward(self, x):
        return self.model(x)

def build_preprocess(image_size: int = 224):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

@torch.inference_mode()
def predict_image(model: DeepfakeDetector, pil_img: Image.Image, device: str = "cpu"):
    model.eval()
    tfm = build_preprocess(224)
    x = tfm(pil_img.convert("RGB")).unsqueeze(0).to(device)
    logit = model(x).squeeze(0).squeeze(0)
    prob_fake = torch.sigmoid(logit).item()
    return prob_fake, logit.item()

def _remap_state_dict_keys(state: dict, add_prefix: str = "", remove_prefix: str = "") -> dict:
    new_state = OrderedDict()
    for k, v in state.items():
        nk = k
        if remove_prefix and nk.startswith(remove_prefix):
            nk = nk[len(remove_prefix):]
        if add_prefix and not nk.startswith(add_prefix):
            nk = add_prefix + nk
        new_state[nk] = v
    return new_state

def load_detector(weights_path: str, device="cpu"):
    model = DeepfakeDetector()  # keep your existing init
    ckpt = torch.load(weights_path, map_location=device)

    # Some checkpoints are saved as {"state_dict": ...}
    state = ckpt.get("state_dict", ckpt)

    # Try normal load first
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError:
        # If weights are missing "model." prefix, add it
        state2 = _remap_state_dict_keys(state, add_prefix="model.")
        try:
            model.load_state_dict(state2, strict=True)
        except RuntimeError:
            # If weights have "model." but model doesn't, remove it
            state3 = _remap_state_dict_keys(state, remove_prefix="model.")
            model.load_state_dict(state3, strict=False)

    model.to(device)
    model.eval()
    return model
