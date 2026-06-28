"""
Week 2 — Pain Expression Classification
Datasets:
  • images/extracted_frames/<PainLevel>/*.jpg  (Dynamic_faces — extracted by extract_frames.py)
  • images/Emotional_faces/Emotional_faces/<subject>/<emotion>.jpg

Emotion → Pain Level mapping:
  neutral / happiness → No_Pain
  sadness             → Mild
  fear / surprise     → Moderate
  anger / disgust     → Severe

Pipeline: Load → ResNet50 features → SVM (5-fold CV) → Metrics + Plots
"""

import os, json, pickle
from pathlib import Path
from collections import Counter

import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from PIL import Image
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, f1_score,
)
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import torch
import torchvision.models as models
import torchvision.transforms as T

# ── Paths & constants ──────────────────────────────────────────────────────────
ROOT    = Path(__file__).resolve().parent.parent
OUT     = ROOT / "week2" / "output"
OUT.mkdir(parents=True, exist_ok=True)

LABELS     = ["No_Pain", "Mild", "Moderate", "Severe"]
COLORS_MAP = {
    "No_Pain":  "#2ecc71",
    "Mild":     "#f1c40f",
    "Moderate": "#e67e22",
    "Severe":   "#e74c3c",
}
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

# Emotional_faces: filename → pain level
EMOTION_TO_PAIN = {
    "neutral":   "No_Pain",
    "happiness": "No_Pain",
    "hapiness":  "No_Pain",   # dataset typo
    "sadness":   "Mild",
    "fear":      "Moderate",
    "surprise":  "Moderate",
    "suprise":   "Moderate",  # dataset typo
    "surpris":   "Moderate",  # dataset typo
    "anger":     "Severe",
    "disgust":   "Severe",
    "disgest":   "Severe",    # dataset typo
}


# ── 1. Dataset loading ─────────────────────────────────────────────────────────
def load_extracted_frames() -> tuple[list, list]:
    """Load frames extracted from Dynamic_faces videos (images/extracted_frames/)."""
    frames_root = ROOT / "images" / "extracted_frames"
    paths, labels = [], []

    if not frames_root.exists():
        print("  WARNING: extracted_frames/ not found — run extract_frames.py first")
        return paths, labels

    for label in LABELS:
        label_dir = frames_root / label
        if not label_dir.exists():
            continue
        imgs = sorted([p for p in label_dir.iterdir() if p.suffix.lower() in IMG_EXTS])
        paths.extend(imgs)
        labels.extend([label] * len(imgs))
        print(f"  [Dynamic]  {label:<12} {len(imgs)} frames")

    return paths, labels


def load_emotional_faces() -> tuple[list, list]:
    """Load static images from Emotional_faces, mapping emotion filename → pain level."""
    emo_root = ROOT / "images" / "Emotional_faces" / "Emotional_faces"
    paths, labels = [], []

    if not emo_root.exists():
        print("  WARNING: Emotional_faces/ not found")
        return paths, labels

    for subject_dir in sorted(emo_root.iterdir()):
        if not subject_dir.is_dir():
            continue
        for img_path in sorted(subject_dir.iterdir()):
            if img_path.suffix.lower() not in IMG_EXTS:
                continue
            emotion    = img_path.stem.lower()
            pain_level = EMOTION_TO_PAIN.get(emotion)
            if pain_level is None:
                continue
            paths.append(img_path)
            labels.append(pain_level)

    # print per-class summary
    counts = Counter(labels)
    for label in LABELS:
        print(f"  [Emotional]{label:<12} {counts.get(label, 0)} images")

    return paths, labels


def load_dataset() -> tuple[list, list]:
    dyn_paths,  dyn_labels  = load_extracted_frames()
    emo_paths,  emo_labels  = load_emotional_faces()

    paths  = dyn_paths  + emo_paths
    labels = dyn_labels + emo_labels

    print(f"  {'TOTAL':<22} {len(paths)} images")
    return paths, labels


# ── 2. Feature extraction ──────────────────────────────────────────────────────
TRANSFORM = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

def extract_features(paths: list, batch_size: int = 64) -> np.ndarray:
    """ResNet50 avgpool output → 2048-d feature vector per image."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device : {device}")

    model    = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    model.fc = torch.nn.Identity()
    model    = model.to(device).eval()

    all_feats = []
    with torch.no_grad():
        for start in range(0, len(paths), batch_size):
            batch = torch.stack([
                TRANSFORM(Image.open(p).convert("RGB"))
                for p in paths[start : start + batch_size]
            ]).to(device)
            all_feats.append(model(batch).cpu().numpy())
            done = min(start + batch_size, len(paths))
            print(f"  {done:>5}/{len(paths)}  ({100*done/len(paths):.1f}%)", end="\r")

    print()
    return np.vstack(all_feats)


def load_or_extract(paths: list, labels: list) -> tuple[np.ndarray, list]:
    feat_cache = OUT / "features.npy"
    lbl_cache  = OUT / "labels.npy"

    if feat_cache.exists() and lbl_cache.exists():
        print("  Loading cached features…")
        features = np.load(feat_cache)
        labels   = list(np.load(lbl_cache))
    else:
        features = extract_features(paths)
        np.save(feat_cache, features)
        np.save(lbl_cache,  np.array(labels))
        print(f"  Cached → {feat_cache}")

    print(f"  Feature matrix : {features.shape}")
    return features, labels


# ── 3. Label distribution plot ─────────────────────────────────────────────────
def plot_label_distribution(labels: list):
    counts = [Counter(labels)[l] for l in LABELS]
    colors = [COLORS_MAP[l] for l in LABELS]

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(LABELS, counts, color=colors, edgecolor="#111", linewidth=0.6)
    for bar, cnt in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                str(cnt), ha="center", va="bottom", fontsize=10)
    ax.set_title("Dataset Label Distribution", fontsize=13, weight="bold")
    ax.set_ylabel("Number of images")
    ax.set_ylim(0, max(counts) * 1.18)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "label_distribution.png", dpi=150)
    plt.close(fig)
    print(f"  Saved → {OUT / 'label_distribution.png'}")


# ── 4. Train + evaluate ────────────────────────────────────────────────────────
def train_and_evaluate(features: np.ndarray, labels: list) -> dict:
    le = LabelEncoder().fit(LABELS)
    y  = le.transform(labels)

    scaler = StandardScaler()
    X      = scaler.fit_transform(features)

    # class_weight='balanced' handles the Severe class being over-represented
    clf = SVC(kernel="rbf", C=10, gamma="scale",
              probability=True, class_weight="balanced", random_state=42)

    print("  Running 5-fold stratified cross-validation…")
    cv     = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    y_pred = cross_val_predict(clf, X, y, cv=cv, n_jobs=-1)

    acc    = accuracy_score(y, y_pred)
    f1_w   = f1_score(y, y_pred, average="weighted")
    report = classification_report(y, y_pred, target_names=le.classes_)

    print(f"\n  Accuracy          : {acc:.4f}")
    print(f"  Weighted F1 Score : {f1_w:.4f}")
    print("\n" + report)

    clf.fit(X, y)
    return clf, scaler, le, {
        "accuracy":    round(float(acc), 4),
        "f1_weighted": round(float(f1_w), 4),
        "y":           y,
        "y_pred":      y_pred,
        "classes":     list(le.classes_),
        "report":      report,
    }


# ── 5. Confusion matrix ────────────────────────────────────────────────────────
def plot_confusion_matrix(y, y_pred, classes: list):
    cm   = confusion_matrix(y, y_pred)
    norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, data, title, fmt in zip(
        axes,
        [cm,   norm],
        ["Confusion Matrix (counts)", "Confusion Matrix (normalised)"],
        ["d",  ".2f"],
    ):
        sns.heatmap(data, annot=True, fmt=fmt, cmap="Purples",
                    xticklabels=classes, yticklabels=classes, ax=ax,
                    linewidths=0.4, linecolor="#333")
        ax.set_xlabel("Predicted", fontsize=10)
        ax.set_ylabel("True",      fontsize=10)
        ax.set_title(title,        fontsize=11, weight="bold")

    fig.tight_layout()
    fig.savefig(OUT / "confusion_matrix.png", dpi=150)
    plt.close(fig)
    print(f"  Saved → {OUT / 'confusion_matrix.png'}")


# ── 6. t-SNE visualisation ─────────────────────────────────────────────────────
def plot_tsne(features: np.ndarray, labels: list, sample_n: int = 2000):
    if len(labels) > sample_n:
        rng  = np.random.default_rng(42)
        idx  = rng.choice(len(labels), size=sample_n, replace=False)
        feat = features[idx]
        lbls = [labels[i] for i in idx]
    else:
        feat, lbls = features, labels

    print(f"  PCA 2048→50 on {len(lbls)} samples…")
    reduced = PCA(n_components=50, random_state=42).fit_transform(feat)

    print("  t-SNE 50→2…")
    emb = TSNE(n_components=2, perplexity=40, random_state=42,
               n_jobs=-1).fit_transform(reduced)

    fig, ax = plt.subplots(figsize=(9, 7))
    for lbl in LABELS:
        mask = np.array(lbls) == lbl
        ax.scatter(emb[mask, 0], emb[mask, 1],
                   c=COLORS_MAP[lbl], label=lbl,
                   s=10, alpha=0.6, edgecolors="none")
    ax.legend(fontsize=10, markerscale=2.5)
    ax.set_title("t-SNE of ResNet50 Features", fontsize=13, weight="bold")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(OUT / "tsne.png", dpi=150)
    plt.close(fig)
    print(f"  Saved → {OUT / 'tsne.png'}")


# ── 7. Save model & metrics ────────────────────────────────────────────────────
def save_outputs(clf, scaler, le, metrics: dict):
    with open(OUT / "model.pkl", "wb") as f:
        pickle.dump({"clf": clf, "scaler": scaler, "le": le}, f)

    with open(OUT / "metrics.json", "w") as f:
        json.dump({"accuracy": metrics["accuracy"],
                   "f1_weighted": metrics["f1_weighted"]}, f, indent=2)

    with open(OUT / "classification_report.txt", "w") as f:
        f.write(metrics["report"])

    print(f"  model.pkl, metrics.json, classification_report.txt → {OUT}")


# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print(" Week 2 — Pain Expression Classification")
    print(" Sources: Dynamic_faces (frames) + Emotional_faces")
    print("=" * 55)

    print("\n[1/6] Loading dataset…")
    paths, labels = load_dataset()

    if not paths:
        print("\nERROR: No images loaded.")
        print("Run extract_frames.py first to extract frames from Dynamic_faces videos.")
        raise SystemExit(1)

    print("\n[2/6] Label distribution…")
    plot_label_distribution(labels)

    print("\n[3/6] Extracting ResNet50 features…")
    features, labels = load_or_extract(paths, labels)

    print("\n[4/6] Training SVM (5-fold CV)…")
    clf, scaler, le, metrics = train_and_evaluate(features, labels)

    print("\n[5/6] Confusion matrix…")
    plot_confusion_matrix(metrics["y"], metrics["y_pred"], metrics["classes"])

    print("\n[6/6] t-SNE visualisation…")
    plot_tsne(features, labels)

    print("\n[Saving outputs]")
    save_outputs(clf, scaler, le, metrics)

    print("\n" + "=" * 55)
    print(f" Accuracy  : {metrics['accuracy']:.4f}")
    print(f" F1 (wtd)  : {metrics['f1_weighted']:.4f}")
    print(f" All files : {OUT}")
    print("=" * 55)
