"""
GrabCut Segmentation Demo
Runs on local Windows machine — no Colab, no GPU needed.
Takes face images from Week4 prediction_samples and produces
proper face-contour segmentation masks.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
SAMPLES_DIR = Path(__file__).parent / 'Week4_Results' / 'prediction_samples'
OUT_DIR     = Path(__file__).parent / 'grabcut_output'
OUT_DIR.mkdir(exist_ok=True)

CLASSES = ['No_Pain', 'Mild', 'Moderate', 'Severe']
COLORS  = {
    'No_Pain':  (46,  204, 113),
    'Mild':     (241, 196,  15),
    'Moderate': (230, 126,  34),
    'Severe':   (231,  76,  60),
}

# Map filenames to pain class based on emotion
EMOTION_MAP = {
    'happiness': 'No_Pain', 'neutral':  'No_Pain',
    'sadness':   'Mild',
    'fear':      'Moderate', 'surprise': 'Moderate',
    'anger':     'Severe',   'disgust':  'Severe',
}

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

def get_pain_class(filename):
    name = filename.lower()
    for emotion, pain in EMOTION_MAP.items():
        if emotion in name:
            return pain
    return 'No_Pain'

def grabcut_segment(img):
    """Run GrabCut using Haar Cascade face box as hint."""
    h, w = img.shape[:2]
    gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 3, minSize=(20, 20))

    if len(faces) > 0:
        fx, fy, fw, fh = faces[0]
        x1, y1 = max(0, fx), max(0, fy)
        x2, y2 = min(w, fx+fw), min(h, fy+fh)
    else:
        pad = int(min(h, w) * 0.08)
        x1, y1, x2, y2 = pad, pad, w-pad, h-pad

    rect = (x1, y1, x2-x1, y2-y1)

    mask  = np.zeros((h, w), np.uint8)
    bgd   = np.zeros((1, 65), np.float64)
    fgd   = np.zeros((1, 65), np.float64)

    cv2.grabCut(img, mask, rect, bgd, fgd, 5, cv2.GC_INIT_WITH_RECT)

    # Pixels marked as foreground or probable foreground
    fg_mask = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 1, 0).astype('uint8')

    # Get contours of segmented region
    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    return fg_mask, contours, (x1, y1, x2, y2)


def visualize(img_bgr, fg_mask, contours, bbox, pain_class):
    """Draw segmentation mask + contour + bbox comparison."""
    color = COLORS[pain_class]
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # ── Left: original with just bounding box (what Week3 did) ──
    left = img_rgb.copy()
    x1, y1, x2, y2 = bbox
    cv2.rectangle(left, (x1, y1), (x2, y2), color, 2)
    label = f'BBox only — {pain_class}'
    cv2.putText(left, label, (x1, max(y1-6, 12)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # ── Right: GrabCut segmentation mask (Week4) ──
    right = img_rgb.copy()
    overlay = np.zeros_like(right)
    overlay[fg_mask == 1] = color
    right = cv2.addWeighted(right, 0.55, overlay, 0.45, 0)

    # Draw precise contour outline
    cv2.drawContours(right, contours, -1, color, 2)

    # Label
    cv2.putText(right, f'GrabCut Seg — {pain_class}', (x1, max(y1-6, 12)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return left, right


# ── Process images ─────────────────────────────────────────────────────────────
images = sorted(SAMPLES_DIR.glob('*.jpg'))
print(f'Found {len(images)} images in prediction_samples/')

results = []   # (pain_class, left, right)

for img_path in images:
    img = cv2.imread(str(img_path))
    if img is None:
        continue

    pain_class  = get_pain_class(img_path.stem)
    fg_mask, contours, bbox = grabcut_segment(img)
    left, right = visualize(img, fg_mask, contours, bbox, pain_class)

    results.append((pain_class, left, right, img_path.stem))
    print(f'  {img_path.name:<30} -> {pain_class}')

# ── Plot: Bounding Box vs GrabCut Segmentation ────────────────────────────────
n    = len(results)
cols = 4
rows = (n + cols//2 - 1) // (cols//2)   # each image takes 2 columns (left+right)

fig, axes = plt.subplots(n, 2, figsize=(12, n * 3.5))
fig.patch.set_facecolor('#0f1117')

if n == 1:
    axes = [axes]

for i, (pain_class, left, right, name) in enumerate(results):
    color_hex = '#{:02x}{:02x}{:02x}'.format(*COLORS[pain_class])

    axes[i][0].imshow(left)
    axes[i][0].axis('off')
    axes[i][0].set_title(f'{name}\nBounding Box Only', color='white', fontsize=8)

    axes[i][1].imshow(right)
    axes[i][1].axis('off')
    axes[i][1].set_title(f'{name}\nGrabCut Segmentation — {pain_class}',
                         color=color_hex, fontsize=8, fontweight='bold')

plt.suptitle('Bounding Box (Week 3) vs GrabCut Segmentation (Week 4)',
             color='white', fontsize=13, fontweight='bold')
plt.tight_layout()

out_path = OUT_DIR / 'grabcut_comparison.png'
plt.savefig(out_path, dpi=120, bbox_inches='tight', facecolor='#0f1117')
plt.show()
print(f'\nSaved: {out_path}')

# ── Also save individual segmented images ─────────────────────────────────────
for pain_class, left, right, name in results:
    save_path = OUT_DIR / f'{name}_segmented.jpg'
    cv2.imwrite(str(save_path), cv2.cvtColor(right, cv2.COLOR_RGB2BGR))

print(f'Individual images saved to: {OUT_DIR}')
