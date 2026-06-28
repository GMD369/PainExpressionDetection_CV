$root    = "F:\Semester 6\Computer Vision\Project\PainExpressionDetection_CV\week4"
$staging = "$root\Week4_Submission\Pain_Expression_Week4"
$zipOut  = "$root\Week4_Submission\Pain_Expression_Week4.zip"

if (Test-Path $staging) { Remove-Item $staging -Recurse -Force }
New-Item -ItemType Directory -Path "$staging\results"             | Out-Null
New-Item -ItemType Directory -Path "$staging\predicted_images"    | Out-Null
New-Item -ItemType Directory -Path "$staging\grabcut_segmentation"| Out-Null

# README
Copy-Item "$root\Week4_Submission\README.txt" "$staging\"

# Notebook
Copy-Item "$root\Week4_YOLOv8_Seg.ipynb" "$staging\"

# Results
Copy-Item "$root\Week4_Results\seg_results.json"          "$staging\results\"
Copy-Item "$root\Week4_Results\sample_masks.png"          "$staging\results\"
Copy-Item "$root\Week4_Results\seg_prediction_grid.png"   "$staging\results\"
Copy-Item "$root\Week4_Results\yolo_seg_results\confusion_matrix.png"            "$staging\results\"
Copy-Item "$root\Week4_Results\yolo_seg_results\confusion_matrix_normalized.png" "$staging\results\"
Copy-Item "$root\Week4_Results\yolo_seg_results\MaskPR_curve.png"                "$staging\results\"
Copy-Item "$root\Week4_Results\yolo_seg_results\MaskF1_curve.png"                "$staging\results\"
Copy-Item "$root\Week4_Results\yolo_seg_results\results.png"                     "$staging\results\"

# Predicted images (YOLOv8-seg output on test set)
Copy-Item "$root\Week4_Results\prediction_samples\*" "$staging\predicted_images\"

# GrabCut segmentation demo
Copy-Item "$root\grabcut_output\*_grabcut.jpg" "$staging\grabcut_segmentation\"

# Build ZIP
if (Test-Path $zipOut) { Remove-Item $zipOut -Force }
Compress-Archive -Path $staging -DestinationPath $zipOut

Write-Host ""
Write-Host "Done! Submission ZIP created:" -ForegroundColor Green
Write-Host "  $zipOut" -ForegroundColor Cyan
Write-Host ""
$size = [math]::Round((Get-Item $zipOut).Length / 1MB, 2)
Write-Host "  Size: $size MB" -ForegroundColor Yellow
