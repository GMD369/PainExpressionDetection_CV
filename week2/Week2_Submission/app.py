"""
Pain Expression Predictor — Flask Web Interface
Model: MobileNetV2 fine-tuned on Dynamic_faces + Emotional_faces
Classes: No_Pain | Mild | Moderate | Severe
"""

import io, base64
from pathlib import Path
from flask import Flask, render_template, request, jsonify
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms
from torchvision.models import MobileNet_V2_Weights

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024   # 16 MB max upload

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "pain_model_weights" / "best_model.pth"

CLASSES = ["No_Pain", "Mild", "Moderate", "Severe"]
COLORS  = {
    "No_Pain":  "#2ecc71",
    "Mild":     "#f1c40f",
    "Moderate": "#e67e22",
    "Severe":   "#e74c3c",
}

TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

# ── Load model once at startup ─────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model():
    m = models.mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
    m.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(m.classifier[1].in_features, len(CLASSES))
    )
    state = torch.load(MODEL_PATH, map_location=device)
    m.load_state_dict(state)
    m.to(device).eval()
    return m

print(f"Loading model from {MODEL_PATH} …")
model = load_model()
print(f"Model ready on {device}")


# ── Routes ─────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    try:
        img = Image.open(file.stream).convert("RGB")
    except Exception:
        return jsonify({"error": "Invalid image file"}), 400

    # Run inference
    tensor = TRANSFORM(img).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(tensor)
        probs  = torch.softmax(logits, dim=1)[0].cpu().tolist()

    predicted = CLASSES[int(torch.argmax(torch.tensor(probs)))]
    scores    = {cls: round(p * 100, 2) for cls, p in zip(CLASSES, probs)}

    # Encode image as base64 to display back in browser
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    img_b64 = base64.b64encode(buf.getvalue()).decode()

    return jsonify({
        "predicted": predicted,
        "color":     COLORS[predicted],
        "scores":    scores,
        "image":     img_b64,
    })


if __name__ == "__main__":
    app.run(debug=True, port=5001)
