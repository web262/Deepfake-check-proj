import cv2
import numpy as np
from PIL import Image

def pil_to_bgr(pil_img: Image.Image) -> np.ndarray:
    rgb = np.array(pil_img.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

def bgr_to_pil(bgr: np.ndarray) -> Image.Image:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)

def crop_largest_face(pil_img: Image.Image, pad: float = 0.25) -> Image.Image:
    """
    Crops the largest detected face. If none found, returns original.
    pad: fraction of face box size to expand.
    """
    bgr = pil_to_bgr(pil_img)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)

    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
    )

    if len(faces) == 0:
        return pil_img

    # pick largest face
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])

    # pad the box
    px = int(w * pad)
    py = int(h * pad)

    x1 = max(0, x - px)
    y1 = max(0, y - py)
    x2 = min(bgr.shape[1], x + w + px)
    y2 = min(bgr.shape[0], y + h + py)

    crop = bgr[y1:y2, x1:x2]
    return bgr_to_pil(crop)
