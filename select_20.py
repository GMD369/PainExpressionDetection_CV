"""
Auto-select 20 best quality sample images from UNBC-McMaster style dataset.
Pain levels assigned by folder type (aff/unaff) and frame position.
"""

import os
import shutil
import cv2
import numpy as np

BASE       = r"F:\Semester 6\Computer Vision\Project\Dataset\images"
OUTPUT     = r"F:\Semester 6\Computer Vision\Project\Dataset\sample_20"
os.makedirs(OUTPUT, exist_ok=True)

# ── Sharpness score (higher = clearer image) ──────────────────────────────────
def sharpness(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return 0
    return cv2.Laplacian(img, cv2.CV_64F).var()

# ── Get all images from a folder, sorted by frame number ─────────────────────
def get_images(folder):
    imgs = sorted([
        os.path.join(BASE, folder, f)
        for f in os.listdir(os.path.join(BASE, folder))
        if f.lower().endswith(".png")
    ])
    return imgs

# ── Pick best N images from a specific frame range ───────────────────────────
def pick_best(images, n, start_pct, end_pct):
    total = len(images)
    start = int(total * start_pct)
    end   = int(total * end_pct)
    subset = images[start:end]
    if not subset:
        subset = images
    scored = [(sharpness(p), p) for p in subset]
    scored.sort(reverse=True)
    return [p for _, p in scored[:n]]

# ── Folder classification ─────────────────────────────────────────────────────
all_folders = os.listdir(BASE)
aff_folders   = [f for f in all_folders if f.endswith("aff") and "unaff" not in f]
unaff_folders = [f for f in all_folders if "unaff" in f]

print(f"Affected folders   : {aff_folders}")
print(f"Unaffected folders : {unaff_folders}")

# ── Pain level selection plan ─────────────────────────────────────────────────
#
#  no_pain       → unaff folders, any frame (pick 4)
#  mild_pain     → aff folders, first 20% of frames (pain just starting)
#  moderate_pain → aff folders, 20-45% of frames
#  severe_pain   → aff folders, 45-70% of frames (peak pain)
#  extreme_pain  → aff folders, 70-100% of frames (sustained pain)
#
selected = {
    "no_pain":       [],
    "mild_pain":     [],
    "moderate_pain": [],
    "severe_pain":   [],
    "extreme_pain":  [],
}

# No Pain — pick 4 from unaffected folders
unaff_images = []
for f in unaff_folders:
    unaff_images.extend(get_images(f))
selected["no_pain"] = pick_best(unaff_images, 4, 0.2, 0.8)

# Pain levels — pick from affected folders across frame ranges
aff_images = []
for f in aff_folders:
    aff_images.extend(get_images(f))
aff_images = sorted(aff_images)  # sort all affected frames together

selected["mild_pain"]     = pick_best(aff_images, 4, 0.00, 0.20)
selected["moderate_pain"] = pick_best(aff_images, 4, 0.20, 0.45)
selected["severe_pain"]   = pick_best(aff_images, 4, 0.45, 0.70)
selected["extreme_pain"]  = pick_best(aff_images, 4, 0.70, 1.00)

# ── Copy selected images to sample_20 folder ─────────────────────────────────
print("\n--- Copying 20 selected images ---")
manifest = []
total = 0

for label, paths in selected.items():
    label_dir = os.path.join(OUTPUT, label)
    os.makedirs(label_dir, exist_ok=True)
    for i, src in enumerate(paths):
        fname    = f"{label}_{i+1:02d}{os.path.splitext(src)[1]}"
        dst      = os.path.join(label_dir, fname)
        shutil.copy2(src, dst)
        sharp    = sharpness(src)
        manifest.append({"label": label, "file": fname, "source": src, "sharpness": round(sharp, 2)})
        print(f"  [{label}] {fname}  (sharpness={sharp:.1f})")
        total += 1

print(f"\nTotal images selected: {total}")

# ── Save manifest ─────────────────────────────────────────────────────────────
import json
manifest_path = os.path.join(OUTPUT, "manifest.json")
with open(manifest_path, "w") as f:
    json.dump(manifest, f, indent=2)
print(f"Manifest saved -> {manifest_path}")

# Summary
print("\n--- Sample 20 Summary ---")
for label, paths in selected.items():
    print(f"  {label:<20} {len(paths)} images")
print(f"\nOutput folder: {OUTPUT}")
print("Done. Open your annotator and load sample_20 folder.")
