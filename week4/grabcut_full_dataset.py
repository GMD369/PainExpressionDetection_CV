"""
GrabCut Full Dataset Processor — runs locally on Windows CPU
No Colab, no GPU, no session timeout.

Steps:
  1. Put pain_dataset_new.zip in the same folder as this script
     (download it from Google Drive)
  2. Run:  python grabcut_full_dataset.py
  3. Script produces:  grabcut_dataset.zip
  4. Upload grabcut_dataset.zip to Google Drive
  5. Update DRIVE_ZIP in Week4 notebook to 'grabcut_dataset.zip'
  6. Retrain — model will now learn real face contours
"""

import zipfile, shutil, os, json, time
from pathlib import Path
from collections import Counter
import cv2
import numpy as np

# ── Config ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR  = Path(__file__).parent
ZIP_IN      = SCRIPT_DIR / 'pain_dataset_new.zip'
WORK_DIR    = SCRIPT_DIR / 'grabcut_work'
OUT_ZIP     = SCRIPT_DIR / 'grabcut_dataset.zip'
LOG_FILE    = SCRIPT_DIR / 'grabcut_log.json'

CLASSES     = ['No_Pain', 'Mild', 'Moderate', 'Severe']
CLASS_TO_ID = {c: i for i, c in enumerate(CLASSES)}
IMG_EXTS    = {'.jpg', '.jpeg', '.png', '.bmp'}

EMOTION_TO_PAIN = {
    'neutral':'No_Pain', 'happiness':'No_Pain', 'hapiness':'No_Pain',
    'sadness':'Mild',
    'fear':'Moderate', 'surprise':'Moderate', 'suprise':'Moderate', 'surpris':'Moderate',
    'anger':'Severe', 'disgust':'Severe', 'disgest':'Severe',
}

TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15

import random
random.seed(42)

# ── Face detector ──────────────────────────────────────────────────────────────
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

def grabcut_polygon(image_path):
    """
    Returns normalized polygon string for YOLO segmentation label.
    Uses GrabCut to find the precise face contour.
    Falls back to bbox polygon if GrabCut fails.
    """
    img = cv2.imread(str(image_path))
    if img is None:
        return None

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

    # ── GrabCut ────────────────────────────────────────────────────────────────
    try:
        mask = np.zeros((h, w), np.uint8)
        bgd  = np.zeros((1, 65), np.float64)
        fgd  = np.zeros((1, 65), np.float64)
        rect = (x1, y1, x2-x1, y2-y1)
        cv2.grabCut(img, mask, rect, bgd, fgd, 5, cv2.GC_INIT_WITH_RECT)

        fg = np.where(
            (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 1, 0
        ).astype('uint8')

        contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            # Take the largest contour (the face)
            cnt = max(contours, key=cv2.contourArea)

            # Simplify contour to reduce number of points
            epsilon = 0.01 * cv2.arcLength(cnt, True)
            approx  = cv2.approxPolyDP(cnt, epsilon, True)

            if len(approx) >= 3:
                pts = approx.reshape(-1, 2)
                poly = ' '.join(f'{px/w:.6f} {py/h:.6f}' for px, py in pts)
                return poly, 'grabcut'

    except Exception:
        pass

    # ── Fallback: bbox polygon ─────────────────────────────────────────────────
    pts = [(x1/w, y1/h), (x2/w, y1/h), (x2/w, y2/h), (x1/w, y2/h)]
    poly = ' '.join(f'{x:.6f} {y:.6f}' for x, y in pts)
    return poly, 'bbox_fallback'


# ── Check input ZIP ────────────────────────────────────────────────────────────
if not ZIP_IN.exists():
    print('='*60)
    print('ERROR: pain_dataset_new.zip not found.')
    print(f'Expected at: {ZIP_IN}')
    print()
    print('Steps:')
    print('  1. Open Google Drive in browser')
    print('  2. Download pain_dataset_new.zip')
    print(f'  3. Move it to: {SCRIPT_DIR}')
    print('  4. Run this script again')
    print('='*60)
    exit(1)

# ── Extract dataset ────────────────────────────────────────────────────────────
print('='*60)
print('GrabCut Full Dataset Processor')
print('='*60)

if WORK_DIR.exists():
    shutil.rmtree(WORK_DIR)
WORK_DIR.mkdir()

print(f'\n[1/5] Extracting {ZIP_IN.name} ...')
with zipfile.ZipFile(ZIP_IN, 'r') as z:
    z.extractall(WORK_DIR / 'raw')
print('      Done.')

# ── Collect all images ─────────────────────────────────────────────────────────
FRAMES_DIR = WORK_DIR / 'raw' / 'images' / 'extracted_frames'
EMO_DIR    = WORK_DIR / 'raw' / 'images' / 'Emotional_faces' / 'Emotional_faces'

all_samples = []

if FRAMES_DIR.exists():
    for label in CLASSES:
        d = FRAMES_DIR / label
        if d.exists():
            for p in d.iterdir():
                if p.suffix.lower() in IMG_EXTS:
                    all_samples.append((p, label))

if EMO_DIR.exists():
    for subj in EMO_DIR.iterdir():
        if not subj.is_dir(): continue
        for img in subj.iterdir():
            if img.suffix.lower() not in IMG_EXTS: continue
            pain = EMOTION_TO_PAIN.get(img.stem.lower())
            if pain:
                all_samples.append((img, pain))

counts = Counter(lbl for _, lbl in all_samples)
print(f'\n[2/5] Collected {len(all_samples)} images:')
for cls in CLASSES:
    print(f'      {cls:<12} {counts[cls]}')

# ── Build YOLO dataset with GrabCut labels ─────────────────────────────────────
YOLO_DIR = WORK_DIR / 'yolo_grabcut'
for split in ['train', 'val', 'test']:
    (YOLO_DIR / 'images' / split).mkdir(parents=True)
    (YOLO_DIR / 'labels' / split).mkdir(parents=True)

print(f'\n[3/5] Running GrabCut on {len(all_samples)} images ...')
print('      This takes ~15-25 minutes on CPU. Please wait.\n')

class_samples = {cls: [] for cls in CLASSES}
for path, label in all_samples:
    class_samples[label].append(path)

stats = {'grabcut': 0, 'bbox_fallback': 0, 'skipped': 0}
split_counts = Counter()
start = time.time()
done  = 0

for cls, paths in class_samples.items():
    random.shuffle(paths)
    n       = len(paths)
    n_train = int(n * TRAIN_RATIO)
    n_val   = int(n * VAL_RATIO)
    splits  = (
        [('train', p) for p in paths[:n_train]] +
        [('val',   p) for p in paths[n_train:n_train+n_val]] +
        [('test',  p) for p in paths[n_train+n_val:]]
    )
    class_id = CLASS_TO_ID[cls]

    for split, src in splits:
        result = grabcut_polygon(src)
        if result is None:
            stats['skipped'] += 1
            done += 1
            continue

        poly, method = result
        stats[method] += 1

        dst_img = YOLO_DIR / 'images' / split / src.name
        if dst_img.exists():
            dst_img = dst_img.with_stem(dst_img.stem + f'_{cls[:3]}')
        shutil.copy2(src, dst_img)

        lbl = YOLO_DIR / 'labels' / split / (dst_img.stem + '.txt')
        lbl.write_text(f'{class_id} {poly}\n')
        split_counts[f'{split}/{cls}'] += 1

        done += 1
        elapsed = time.time() - start
        rate    = done / elapsed if elapsed > 0 else 0
        remaining = (len(all_samples) - done) / rate if rate > 0 else 0

        if done % 50 == 0 or done == len(all_samples):
            print(f'      [{done:>4}/{len(all_samples)}]  '
                  f'GrabCut:{stats["grabcut"]}  '
                  f'Fallback:{stats["bbox_fallback"]}  '
                  f'ETA: {remaining/60:.1f} min')

print(f'\n      Finished in {(time.time()-start)/60:.1f} minutes.')
print(f'      GrabCut masks  : {stats["grabcut"]}')
print(f'      Bbox fallback  : {stats["bbox_fallback"]}')
print(f'      Skipped        : {stats["skipped"]}')

# ── data.yaml ─────────────────────────────────────────────────────────────────
print('\n[4/5] Writing data.yaml ...')
yaml_text = (
    f'path: /content/yolo_grabcut_dataset\n'
    f'train: images/train\n'
    f'val:   images/val\n'
    f'test:  images/test\n\n'
    f'nc: {len(CLASSES)}\n'
    f'names: {CLASSES}\n'
)
(YOLO_DIR / 'data.yaml').write_text(yaml_text)

for split in ['train', 'val', 'test']:
    total = sum(v for k, v in split_counts.items() if k.startswith(split))
    print(f'      {split:<6} {total}')
    for cls in CLASSES:
        print(f'             {cls:<12} {split_counts[f"{split}/{cls}"]}')

# ── Save log ───────────────────────────────────────────────────────────────────
log = {
    'total_images': len(all_samples),
    'grabcut_masks': stats['grabcut'],
    'bbox_fallback': stats['bbox_fallback'],
    'skipped': stats['skipped'],
    'split_counts': dict(split_counts),
}
LOG_FILE.write_text(json.dumps(log, indent=2))

# ── ZIP output ─────────────────────────────────────────────────────────────────
print(f'\n[5/5] Creating {OUT_ZIP.name} ...')
if OUT_ZIP.exists():
    OUT_ZIP.unlink()

shutil.make_archive(str(OUT_ZIP.with_suffix('')), 'zip', WORK_DIR, 'yolo_grabcut')
print(f'      Saved: {OUT_ZIP}')

print('\n' + '='*60)
print('DONE!')
print('='*60)
print(f'Output ZIP  : {OUT_ZIP}')
print(f'Log         : {LOG_FILE}')
print()
print('Next steps:')
print('  1. Upload grabcut_dataset.zip to Google Drive')
print('  2. In Week4 notebook Cell 3, change:')
print("       DRIVE_ZIP = '/content/drive/MyDrive/grabcut_dataset.zip'")
print('  3. Also change:')
print("       YOLO_DIR  = Path('/content/yolo_grabcut_dataset')")
print('  4. Run All — model will train on real GrabCut face masks')
print('='*60)
