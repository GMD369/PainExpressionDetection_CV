"""
Week 2 — Pain Expression Classification
Pipeline: Load dataset → ResNet50 features → SVM (5-fold CV) → Metrics + Plots
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

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT   = Path(__file__).resolve().parent.parent          # PainExpressionDetection_CV/
IMAGES = ROOT / "images"
OUT    = ROOT / "week2" / "output"
OUT.mkdir(parents=True, exist_ok=True)

PAIN_LABELS = ["no_pain", "mild_pain", "moderate_pain", "severe_pain", "extreme_pain"]
COLORS_MAP  = dict(zip(PAIN_LABELS, ["#2ecc71","#f1c40f","#e67e22","#e74c3c","#8e44ad"]))


# ── 1. Dataset loading ─────────────────────────────────────────────────────────
def _temporal_label(idx: int, total: int) -> str:
    pos = idx / total
    if pos < 0.20: return "mild_pain"
    if pos < 0.45: return "moderate_pain"
    if pos < 0.70: return "severe_pain"
    return "extreme_pain"


def load_dataset() -> tuple[list, list]:
    """
    Assigns labels by folder type and temporal frame position — the same
    strategy used in select_20.py.  Each 'aff' (affected/pain) sequence has
    frames divided into 4 pain levels; 'unaff' frames are all no_pain.
    """
    paths, labels = [], []
    for folder in sorted(os.listdir(IMAGES)):
        folder_abs = IMAGES / folder
        if not folder_abs.is_dir():
            continue
        is_unaff = "unaff" in folder
        imgs = sorted(folder_abs.glob("*.png"))
        n    = len(imgs)
        for i, img in enumerate(imgs):
            lbl = "no_pain" if is_unaff else _temporal_label(i, n)
            paths.append(img)
            labels.append(lbl)

    dist = dict(sorted(Counter(labels).items()))
    print(f"  Total images : {len(paths)}")
    for lbl, cnt in dist.items():
        print(f"  {lbl:<22} {cnt}")
    return paths, labels


# ── 2. Feature extraction ──────────────────────────────────────────────────────
TRANSFORM = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

def extract_features(paths: list, batch_size: int = 64) -> np.ndarray:
    """ResNet50 (avgpool output = 2048-d) as a fixed feature extractor."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device : {device}")

    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    model.fc = torch.nn.Identity()          # remove classification head
    model    = model.to(device).eval()

    all_feats = []
    with torch.no_grad():
        for start in range(0, len(paths), batch_size):
            batch_paths = paths[start : start + batch_size]
            imgs = torch.stack([
                TRANSFORM(Image.open(p).convert("RGB"))
                for p in batch_paths
            ]).to(device)
            feats = model(imgs).cpu().numpy()
            all_feats.append(feats)
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
        np.save(lbl_cache, np.array(labels))
        print(f"  Cached → {feat_cache}")

    print(f"  Feature matrix : {features.shape}")
    return features, labels


# ── 3. Label distribution plot ─────────────────────────────────────────────────
def plot_label_distribution(labels: list):
    counts = [Counter(labels)[l] for l in PAIN_LABELS]
    colors = [COLORS_MAP[l] for l in PAIN_LABELS]
    names  = [l.replace("_", "\n") for l in PAIN_LABELS]

    fig, ax = plt.subplots(figsize=(9, 4))
    bars = ax.bar(names, counts, color=colors, edgecolor="#111", linewidth=0.6)
    for bar, cnt in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                str(cnt), ha="center", va="bottom", fontsize=9)
    ax.set_title("Dataset Label Distribution", fontsize=13, weight="bold")
    ax.set_ylabel("Number of images")
    ax.set_ylim(0, max(counts) * 1.15)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "label_distribution.png", dpi=150)
    plt.close(fig)
    print(f"  Saved → {OUT / 'label_distribution.png'}")


# ── 4. Train + evaluate (SVM, 5-fold CV) ──────────────────────────────────────
def train_and_evaluate(features: np.ndarray, labels: list) -> dict:
    le      = LabelEncoder().fit(PAIN_LABELS)   # fixed order
    y       = le.transform(labels)
    scaler  = StandardScaler()
    X       = scaler.fit_transform(features)

    clf = SVC(kernel="rbf", C=10, gamma="scale",
              probability=True, random_state=42, class_weight="balanced")

    print("  Running 5-fold stratified cross-validation…")
    cv     = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    y_pred = cross_val_predict(clf, X, y, cv=cv, n_jobs=-1)

    acc    = accuracy_score(y, y_pred)
    f1_w   = f1_score(y, y_pred, average="weighted")
    report = classification_report(y, y_pred, target_names=le.classes_)

    print(f"\n  Accuracy          : {acc:.4f}")
    print(f"  Weighted F1 Score : {f1_w:.4f}")
    print("\n" + report)

    # Train final model on ALL data for saving
    clf.fit(X, y)
    return clf, scaler, le, {
        "accuracy":     round(float(acc), 4),
        "f1_weighted":  round(float(f1_w), 4),
        "y":            y,
        "y_pred":       y_pred,
        "classes":      list(le.classes_),
        "report":       report,
    }


# ── 5. Confusion matrix ────────────────────────────────────────────────────────
def plot_confusion_matrix(y, y_pred, classes: list):
    cm   = confusion_matrix(y, y_pred)
    norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)  # row-normalised

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    labels = [c.replace("_", "\n") for c in classes]

    for ax, data, title, fmt in zip(
        axes,
        [cm, norm],
        ["Confusion Matrix (counts)", "Confusion Matrix (normalised)"],
        ["d", ".2f"],
    ):
        sns.heatmap(data, annot=True, fmt=fmt, cmap="Purples",
                    xticklabels=labels, yticklabels=labels, ax=ax,
                    linewidths=0.4, linecolor="#333")
        ax.set_xlabel("Predicted", fontsize=10)
        ax.set_ylabel("True",      fontsize=10)
        ax.set_title(title,        fontsize=11, weight="bold")

    fig.tight_layout()
    fig.savefig(OUT / "confusion_matrix.png", dpi=150)
    plt.close(fig)
    print(f"  Saved → {OUT / 'confusion_matrix.png'}")


# ── 6. t-SNE feature visualisation ────────────────────────────────────────────
def plot_tsne(features: np.ndarray, labels: list, sample_n: int = 1500):
    # Sub-sample for speed if dataset is large
    if len(labels) > sample_n:
        rng  = np.random.default_rng(42)
        idx  = rng.choice(len(labels), size=sample_n, replace=False)
        feat = features[idx]
        lbls = [labels[i] for i in idx]
    else:
        feat, lbls = features, labels

    print(f"  PCA 2048→50 on {len(lbls)} samples…")
    pca    = PCA(n_components=50, random_state=42)
    reduced = pca.fit_transform(feat)

    print("  t-SNE 50→2 (this takes ~2 min)…")
    tsne   = TSNE(n_components=2, perplexity=30, random_state=42, n_jobs=-1)
    emb    = tsne.fit_transform(reduced)

    fig, ax = plt.subplots(figsize=(9, 7))
    for lbl in PAIN_LABELS:
        mask = np.array(lbls) == lbl
        ax.scatter(emb[mask, 0], emb[mask, 1],
                   c=COLORS_MAP[lbl], label=lbl.replace("_", " "),
                   s=12, alpha=0.65, edgecolors="none")
    ax.legend(fontsize=9, markerscale=2)
    ax.set_title("t-SNE of ResNet50 Features", fontsize=13, weight="bold")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(OUT / "tsne.png", dpi=150)
    plt.close(fig)
    print(f"  Saved → {OUT / 'tsne.png'}")


# ── 7. Save model and metrics ──────────────────────────────────────────────────
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
    print("=" * 55)

    print("\n[1/6] Loading dataset…")
    paths, labels = load_dataset()

    print("\n[2/6] Label distribution…")
    plot_label_distribution(labels)

    print("\n[3/6] Extracting ResNet50 features…")
    features, labels = load_or_extract(paths, labels)

    print("\n[4/6] Training SVM (5-fold CV)…")
    clf, scaler, le, metrics = train_and_evaluate(features, labels)

    print("\n[5/6] Plotting confusion matrix…")
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
