Week 4 — Pain Expression Segmentation
=======================================
Subject : Computer Vision (Semester 6)
Topic   : Pain Expression Detection — Instance Segmentation
Model   : YOLOv8n-seg
Deadline: 02-06-2026

===========================================================================
APPROACH
===========================================================================

Goal:
  Detect and segment the facial region in each image, then classify the
  pain level as one of four classes:
    No_Pain | Mild | Moderate | Severe

Pain Level Mapping (Emotion -> Pain):
  No Pain  : neutral, happiness
  Mild     : sadness
  Moderate : fear, surprise
  Severe   : anger, disgust

Dataset:
  SZU-EmoDage dataset — 120 subjects
  2,288 images total (video frames + static emotion images)
  Source: https://osf.io/7a5fs/

Step-by-Step Approach:

  1. FACE DETECTION (Haar Cascade)
     Used OpenCV's built-in Haar Cascade classifier
     (haarcascade_frontalface_default.xml) to locate the face in
     each image and get a bounding box (x, y, w, h).
     No extra installation required — built into OpenCV.

  2. SEGMENTATION LABEL GENERATION (GrabCut)
     Used OpenCV GrabCut algorithm to extract the precise face
     contour from the detected bounding box region.
     GrabCut separates foreground (face) from background pixels.
     The resulting contour was simplified using approxPolyDP and
     saved as a normalized polygon in YOLO segmentation format:
       class_id  x1 y1  x2 y2  x3 y3 ... (normalized 0-1)
     Total: 2,288 images processed, 0 fallbacks, 0 skipped.

  3. DATASET SPLIT
     Train : 70%  (1,598 images)
     Val   : 15%  (343 images)
     Test  : 15%  (347 images)
     Split done per-class to maintain class balance.

  4. MODEL TRAINING (YOLOv8n-seg)
     Base model  : yolov8n-seg.pt (pre-trained on COCO — Transfer Learning)
     Epochs      : 50
     Image size  : 224 x 224
     Batch size  : 16
     Early stop  : patience=10 (stopped if no improvement for 10 epochs)
     Optimizer   : Auto (AdamW)
     Platform    : Google Colab (T4 GPU)

  5. EVALUATION
     Evaluated on the held-out test split using:
       - Mask mAP@0.5       (main segmentation metric)
       - Mask mAP@0.5:0.95  (strict multi-threshold metric)
       - Mask Precision / Recall
       - Per-class Mask AP@0.5

===========================================================================
TOOLS USED
===========================================================================

  Tool              Version    Purpose
  ----------------  ---------  --------------------------------------------
  Python            3.10       Programming language
  OpenCV (cv2)      4.x        Haar Cascade face detection + GrabCut
  Ultralytics YOLO  8.x        YOLOv8n-seg model training and inference
  PyTorch           2.x        Deep learning backend for YOLO
  NumPy             1.x        Array operations
  Matplotlib        3.x        Visualization and result plots
  Google Colab      -          Cloud GPU training (T4 GPU)
  Google Drive      -          Dataset storage and results backup

===========================================================================
RESULTS
===========================================================================

  Metric                  Score
  ----------------------  --------
  Mask mAP@0.5            98.79%     <- main metric
  Mask mAP@0.5:0.95       93.56%
  Box  mAP@0.5            98.79%
  Mask Precision          97.51%
  Mask Recall             97.27%

  Per-class Mask AP@0.5:
    No_Pain    98.44%
    Mild       98.59%
    Moderate   99.49%
    Severe     98.64%

===========================================================================
FOLDER CONTENTS
===========================================================================

  Week4_YOLOv8_Seg.ipynb
    Main Google Colab notebook — full pipeline from data loading
    to training, evaluation, and saving results.

  results/
    seg_results.json          - All metrics in JSON format
    sample_masks.png          - Sample training images with masks
    seg_prediction_grid.png   - Grid of test set predictions
    confusion_matrix.png      - Confusion matrix (raw counts)
    confusion_matrix_normalized.png  - Confusion matrix (percentages)
    MaskPR_curve.png          - Mask Precision-Recall curve per class
    MaskF1_curve.png          - Mask F1 score vs confidence threshold
    results.png               - Training loss + mAP curves over 50 epochs

  predicted_images/
    20 test images with YOLOv8-seg segmentation masks overlaid.
    Colored by pain class:
      Blue  = No_Pain  |  Cyan  = Mild
      White = Moderate |  Green = Severe

  grabcut_segmentation/
    Sample images showing GrabCut face contour segmentation.
    Demonstrates pixel-level face mask extraction used to generate
    training labels. Run locally on Windows CPU (2,288 images
    processed in ~20 minutes — no GPU required).

===========================================================================
HOW TO RUN
===========================================================================

  1. Upload Week4_YOLOv8_Seg.ipynb to Google Colab
  2. Upload pain_dataset_new.zip to Google Drive
  3. Run Cell 1 (Mount Drive) manually — click auth link
  4. Press Ctrl+F9 to run all remaining cells
  5. Results save automatically to Drive/Week4_Results/
