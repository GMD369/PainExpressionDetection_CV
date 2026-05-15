"""
Week 2 — Single-image pain level predictor.
Loads the saved model and returns a label + confidence scores.

Usage:
  python predict.py path/to/image.png
  python predict.py  (opens a file dialog)
"""

import sys, pickle
from pathlib import Path

import numpy as np
from PIL import Image

import torch
import torchvision.models as models
import torchvision.transforms as T

ROOT      = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "week2" / "output"
MODEL_PATH = MODEL_DIR / "model.pkl"

PAIN_LABELS = ["no_pain", "mild_pain", "moderate_pain", "severe_pain", "extreme_pain"]
TRANSFORM   = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}.\n"
            "Run classify.py first to train and save the model."
        )
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


def build_extractor():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    model.fc = torch.nn.Identity()
    return model.to(device).eval(), device


def predict(image_path: str) -> dict:
    saved   = load_model()
    clf     = saved["clf"]
    scaler  = saved["scaler"]
    le      = saved["le"]

    extractor, device = build_extractor()

    img   = Image.open(image_path).convert("RGB")
    tensor = TRANSFORM(img).unsqueeze(0).to(device)

    with torch.no_grad():
        feat = extractor(tensor).cpu().numpy()

    feat_scaled = scaler.transform(feat)
    pred_idx    = clf.predict(feat_scaled)[0]
    proba       = clf.predict_proba(feat_scaled)[0]

    label  = le.inverse_transform([pred_idx])[0]
    scores = {le.classes_[i]: round(float(p), 4) for i, p in enumerate(proba)}

    return {"predicted": label, "confidence": scores}


if __name__ == "__main__":
    if len(sys.argv) > 1:
        img_path = sys.argv[1]
    else:
        try:
            import tkinter as tk
            from tkinter import filedialog
            tk.Tk().withdraw()
            img_path = filedialog.askopenfilename(
                title="Select an image",
                filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp *.tiff")]
            )
            if not img_path:
                print("No file selected."); raise SystemExit
        except Exception:
            print("Usage: python predict.py <image_path>"); raise SystemExit

    result = predict(img_path)
    print(f"\nImage     : {img_path}")
    print(f"Predicted : {result['predicted'].upper()}")
    print("\nConfidence scores:")
    for lbl, score in sorted(result["confidence"].items(),
                               key=lambda x: -x[1]):
        bar = "█" * int(score * 30)
        print(f"  {lbl:<22} {score:.4f}  {bar}")
