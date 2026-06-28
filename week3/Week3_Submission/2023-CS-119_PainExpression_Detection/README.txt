Week 3 — Pain Expression Object Detection
==========================================
Subject : Computer Vision (Semester 6)
Topic   : Pain Expression Detection — Object Detection with YOLOv8
Model   : YOLOv8n

===========================================================================
APPROACH
===========================================================================

Goal:
  Detect the facial region in each image using a bounding box and
  classify the pain level as one of four classes:
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

  1. DATASET PREPARATION
     Extracted the dataset ZIP from Google Drive.
     Collected images from two sources:
       - extracted_frames/   : video frames labeled by folder (pain class)
       - Emotional_faces/    : static images labeled by filename (emotion)
     Emotion-to-pain mapping applied to Emotional_faces filenames.

  2. FACE DETECTION + LABEL GENERATION (Haar Cascade)
     Used OpenCV's built-in Haar Cascade classifier
     (haarcascade_frontalface_default.xml) to locate the face
     in each image and get bounding box (x, y, w, h).
     Bounding box converted to YOLO format:
       class_id  cx  cy  w  h  (all normalized 0-1)
     If no face detected: full image used as fallback bounding box.

  3. DATASET SPLIT
     Train : 70%  (1,598 images)
     Val   : 15%  (343 images)
     Test  : 15%  (347 images)
     Split done per-class to maintain class balance.

  4. MODEL TRAINING (YOLOv8n)
     Base model  : yolov8n.pt (pre-trained on COCO — Transfer Learning)
     Epochs      : 50
     Image size  : 224 x 224
     Batch size  : 16
     Early stop  : patience=10
     Optimizer   : Auto (AdamW)
     Platform    : Google Colab (T4 GPU)

  5. EVALUATION
     Evaluated on held-out test split using:
       mAP@0.5        — detection correct if IoU >= 0.5
       mAP@0.5:0.95   — averaged across IoU 0.5 to 0.95
       Precision / Recall

===========================================================================
TOOLS USED
===========================================================================

  Tool              Version    Purpose
  ----------------  ---------  --------------------------------------------
  Python            3.10       Programming language
  OpenCV (cv2)      4.x        Haar Cascade face detection
  Ultralytics YOLO  8.x        YOLOv8n model training and inference
  PyTorch           2.x        Deep learning backend for YOLO
  NumPy             1.x        Array operations
  Matplotlib        3.x        Visualization and result plots
  Google Colab      -          Cloud GPU training (T4 GPU)
  Google Drive      -          Dataset storage and results backup

===========================================================================
RESULTS
===========================================================================

  Metric              Score
  ------------------  --------
  mAP@0.5             99.09%     <- main detection metric
  mAP@0.5:0.95        98.45%
  Precision           97.30%
  Recall              97.51%

  Per-class AP@0.5:
    No_Pain    98.39%
    Mild       99.14%
    Moderate   99.46%
    Severe     99.37%

===========================================================================
FOLDER CONTENTS
===========================================================================

  Week3_YOLOv8_Pain.ipynb
    Main Google Colab notebook — full pipeline from data loading
    to training, evaluation, and saving results.

  results/
    map_results.json              - All metrics in JSON format
    training_curves.png           - Loss + mAP curves over 50 epochs
    prediction_grid.png           - Grid of test set detections
    sample_annotations.png        - Sample training images with boxes
    confusion_matrix.png          - Confusion matrix (raw counts)
    confusion_matrix_normalized.png
    BoxPR_curve.png               - Precision-Recall curve per class
    BoxF1_curve.png               - F1 score vs confidence threshold

  detected_images/
    20 test images with YOLOv8 detection bounding boxes overlaid.
    Each image shows the predicted pain class + confidence score.
    Color coded by class:
      Blue  = No_Pain  |  Cyan  = Mild
      White = Moderate |  Green = Severe

===========================================================================
HOW TO RUN
===========================================================================

  1. Upload Week3_YOLOv8_Pain.ipynb to Google Colab
  2. Upload pain_dataset_new.zip to Google Drive
  3. Run Cell 1 (Mount Drive) manually — click auth link
  4. Press Ctrl+F9 to run all remaining cells
  5. Results save automatically to Drive/Week3_Results/
