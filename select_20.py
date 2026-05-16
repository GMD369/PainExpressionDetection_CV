"""
Select 20 best-quality sample images from the pain expression dataset.
Dataset structure: images/Label/Label/*.jpg  (4 classes, 2000 images each)
Picks 5 sharpest images per class = 20 total.
"""

import os, shutil, json
import cv2
import numpy as np

ROOT    = r"F:\Semester 6\Computer Vision\Project\PainExpressionDetection_CV"
BASE    = os.path.join(ROOT, "images")
OUTPUT  = os.path.join(ROOT, "sample_20")
os.makedirs(OUTPUT, exist_ok=True)

LABELS  = ["No_Pain", "Mild", "Moderate", "Severe"]
N_EACH  = 5          # images per class  →  5 × 4 = 20 total


# ── Sharpness score (Laplacian variance — higher = clearer) ───────────────────
def sharpness(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    return cv2.Laplacian(img, cv2.CV_64F).var() if img is not None else 0


# ── Pick N sharpest images from a folder ──────────────────────────────────────
def pick_best(folder, n):
    exts  = {".jpg", ".jpeg", ".png", ".bmp"}
    imgs  = [
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if os.path.splitext(f)[1].lower() in exts
    ]
    scored = sorted([(sharpness(p), p) for p in imgs], reverse=True)
    return [p for _, p in scored[:n]]


# ── Select and copy ───────────────────────────────────────────────────────────
print("=== Selecting 20 sample images ===\n")
manifest = []
total    = 0

for label in LABELS:
    # Dataset stores images one level deeper: images/Label/Label/
    img_dir   = os.path.join(BASE, label, label)
    out_dir   = os.path.join(OUTPUT, label)
    os.makedirs(out_dir, exist_ok=True)

    best = pick_best(img_dir, N_EACH)

    for i, src in enumerate(best):
        ext   = os.path.splitext(src)[1]
        fname = f"{label}_{i+1:02d}{ext}"
        dst   = os.path.join(out_dir, fname)
        shutil.copy2(src, dst)
        sharp = sharpness(src)
        manifest.append({
            "label":     label,
            "file":      fname,
            "source":    src,
            "sharpness": round(sharp, 2),
        })
        print(f"  [{label}] {fname}  (sharpness={sharp:.1f})")
        total += 1

# ── Save manifest ─────────────────────────────────────────────────────────────
with open(os.path.join(OUTPUT, "manifest.json"), "w") as f:
    json.dump(manifest, f, indent=2)

print(f"\nTotal selected : {total}")
print("\n--- Summary ---")
for label in LABELS:
    cnt = sum(1 for m in manifest if m["label"] == label)
    print(f"  {label:<12} {cnt} images")
print(f"\nOutput → {OUTPUT}")
print("Done. Upload the sample_20 folder into the web annotator.")
