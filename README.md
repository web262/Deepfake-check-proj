# DeepFake Check — Verify Before You Share

A lightweight **Streamlit** web app that estimates **deepfake risk** for face images and provides an **explainable heatmap** (Grad-CAM style) to highlight regions that influenced the model’s decision.

> Goal: Help users verify suspicious images quickly, with transparency  not just a “fake/real” label.

---

##  Features

- **Upload & classify (JPG/PNG)**  
  Get a **Deepfake Risk score (0–100)** + risk label (**Low / Medium / High**).

- **Explainability Heatmap (Grad-CAM style)**  
  Visual overlay to show *where the model is focusing* (eyes, mouth edges, skin texture, blending boundaries, etc.).

- **Auto-crop largest face**  
  Option to automatically crop the most prominent face for better inference stability.

- **Heatmap strength control**  
  Slider to increase/decrease overlay intensity for clearer visual interpretation.

- **Configurable model weights path**  
  Swap weights easily via sidebar path input.

---

##  How it works (high level)

1. The app loads an **EfficientNet-based** deepfake detector model.
2. You upload an image → optional **face auto-crop**.
3. The model outputs a **probability** which is mapped to a **0–100 risk score**.
4. Grad-CAM generates a **heatmap** explaining which image regions most influenced the prediction.

---

## 📦 Project Structure

<img width="504" height="440" alt="image" src="https://github.com/user-attachments/assets/1acbbc7a-f62e-441b-9620-c708c7d0608d" />


---

## 🚀 Run Locally

### 1) Install dependencies
```bash
pip install -r requirements.txt

streamlit run app.py

Then open the URL shown in your terminal (usually http://localhost:8501).

🏋️‍♂️ Optional: Retrain / Fine-tune

If you want to retrain the model using your own dataset:

1) Prepare dataset

Place images here:

data/real/ → real face images

data/fake/ → AI-generated / manipulated face images

2) Train
python train.py

3) Use new weights

The script will output a new weights file. Move it into:

weights/your_new_weights.pth

Then update the weights path in the app (sidebar) if needed.

⚠️ Notes & Limitations

This app provides a risk estimate, not a guaranteed verdict.

Results can be affected by:

low-resolution images

heavy compression

extreme angles / occlusions

non-face images (best results are face-centric)
