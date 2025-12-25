import streamlit as st
import torch
from PIL import Image

from deepfake_check.detector import load_detector, predict_image, build_preprocess
from deepfake_check.explain import GradCAM, overlay_heatmap
from deepfake_check.vision import crop_largest_face

st.set_page_config(page_title="DeepFake Check", layout="wide")

st.title("DeepFake Check — Verify Before You Share")
st.caption("Upload an image → get a deepfake risk score + visual explanation (heatmap) + verification guidance.")

# Sidebar
st.sidebar.header("Settings")
use_face_crop = st.sidebar.checkbox("Auto-crop largest face", value=True)
alpha = st.sidebar.slider("Heatmap strength", 0.15, 0.75, 0.45, 0.05)
weights_path = st.sidebar.text_input("Weights file", value="weights/deepfake_efficientnet_b0.pth")
device = "cuda" if torch.cuda.is_available() else "cpu"
st.sidebar.write(f"Device: **{device}**")

@st.cache_resource
def get_model(path: str, device: str):
    return load_detector(path, device=device)

model = get_model(weights_path, device)

uploaded = st.file_uploader("Upload an image (JPG/PNG)", type=["jpg", "jpeg", "png"])
if not uploaded:
    st.info("Tip: upload a face image for best results.")
    st.stop()

img = Image.open(uploaded).convert("RGB")
work_img = crop_largest_face(img) if use_face_crop else img

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Input")
    st.image(work_img, use_column_width=True, caption="Face-cropped" if use_face_crop else "Original")

prob_fake, logit = predict_image(model, work_img, device=device)
risk = int(round(prob_fake * 100))

# Grad-CAM target layer for EfficientNet-B0 (torchvision)
target_layer = model.model.features[-1]
tfm = build_preprocess(224)
x = tfm(work_img).unsqueeze(0).to(device)
cam = GradCAM(model, target_layer)
cam_map = cam(x)
cam.close()
heat_img = overlay_heatmap(work_img, cam_map, alpha=alpha)

label = "Low" if risk < 35 else "Medium" if risk < 70 else "High"

with col2:
    st.subheader("Result")
    st.metric("Deepfake Risk", f"{risk}/100", help="Risk score is a probability estimate, not a final verdict.")
    st.write(f"**Risk level:** {label}")
    st.image(heat_img, use_column_width=True, caption="Explainability heatmap (where the model focused)")
    st.caption("Use this as a verification aid. Always check the original source/context.")

st.divider()
st.subheader("Why it might be suspicious (Reason Cards)")

reasons = []
if label == "High":
    reasons = [
        "Possible blending artifacts around facial boundaries (edges, hairline, jaw).",
        "Texture inconsistency (skin/eyes) compared with surrounding regions.",
        "Lighting/shadow mismatch across face regions.",
    ]
elif label == "Medium":
    reasons = [
        "Some artifact-like patterns detected; could also be compression or low resolution.",
        "Minor inconsistencies in texture/edges; verify source and context.",
    ]
else:
    reasons = [
        "No strong manipulation cues detected, but this does not guarantee authenticity.",
        "If the content is sensitive/viral, verify source before sharing."
    ]

for r in reasons:
    st.write(f"- {r}")

st.subheader("Verify before sharing (Checklist)")
st.write("""
- **Check the original uploader/source** (is it a trusted account or official channel?)
- **Reverse image search** to find the earliest appearance.
- **Look for context** (full clip, full conversation, uncut footage).
- If it could harm someone: **don’t repost** until verified.
""")

st.caption("Note: This tool flags content risk; it does not identify people or make accusations.")
