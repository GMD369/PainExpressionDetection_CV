Week 2 — Medical Image Annotation and Classification
=====================================================
Subject : Computer Vision (Semester 6)
Topic   : Pain Expression Detection from Facial Images

---------------------------------------------------------------------------
FOLDER CONTENTS
---------------------------------------------------------------------------

1. dataset_link.txt
   - Full dataset name, DOI, download URL, and citation
   - Dataset: SZU-EmoDage (Scientific Data, Nature 2023)

2. MobileNet_Colab_pain.ipynb
   - Main classification notebook (Google Colab)
   - Model: MobileNetV2 fine-tuned on 2,288 images
   - Test Accuracy : 93.02%
   - Weighted F1   : 93.07%
   - Best Val Acc  : 95.35%

3. classify.py
   - Baseline classification script
   - Model: ResNet50 features (2048-d) + SVM classifier
   - Class-balanced SVM with cross-validation

4. annotated_dataset/
   - 20 hand-annotated sample images (5 per pain level)
   - Pain levels: No_Pain, Mild, Moderate, Severe
   - Each subfolder contains images + annotations.csv
   - Annotation format: bounding box + polygon per image

5. results/
   - confusion_matrix.png    (MobileNetV2)
   - training_curves.png     (loss & accuracy per epoch)
   - classification_report.txt (per-class precision/recall/F1)
   - test_metrics.json       (accuracy, F1, precision, recall)

6. research_paper/
   - Place the downloaded PDF of the SZU-EmoDage paper here
   - Download from: https://www.nature.com/articles/s41597-023-02701-2

---------------------------------------------------------------------------
HOW TO RUN
---------------------------------------------------------------------------

Colab Notebook:
  1. Upload MobileNet_Colab_pain.ipynb to Google Colab
  2. Mount Google Drive
  3. Upload dataset ZIP to Drive (path in notebook cell 2)
  4. Run all cells

Local SVM baseline:
  pip install -r requirements.txt
  python classify.py

---------------------------------------------------------------------------
PAIN LEVEL MAPPING (Emotion -> Pain)
---------------------------------------------------------------------------
  No Pain  : neutral, happiness expressions
  Mild     : sadness expression
  Moderate : fear, surprise expressions
  Severe   : anger, disgust expressions

This mapping is supported by facial Action Unit (AU) research:
  - Severe pain activates AU4 (brow lowerer) + AU7 (lid tightener)
    which overlap strongly with anger/disgust AUs.
  - Fear/surprise activate AU1+AU2 (brow raise) matching moderate
    pain expressions.
