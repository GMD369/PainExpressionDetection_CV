$root    = "F:\Semester 6\Computer Vision\Project\PainExpressionDetection_CV\week3"
$staging = "$root\Week3_Submission\Pain_Expression_Week3"
$zipOut  = "$root\Week3_Submission\Pain_Expression_Week3.zip"

if (Test-Path $staging) { Remove-Item $staging -Recurse -Force }
New-Item -ItemType Directory -Path "$staging\results"         | Out-Null
New-Item -ItemType Directory -Path "$staging\detected_images" | Out-Null

# README
Copy-Item "$root\Week3_Submission\README.txt" "$staging\"

# Notebook
Copy-Item "$root\Week3_YOLOv8_Pain.ipynb" "$staging\"

# Results
Copy-Item "$root\Week3_Results\map_results.json"       "$staging\results\"
Copy-Item "$root\Week3_Results\training_curves.png"    "$staging\results\"
Copy-Item "$root\Week3_Results\prediction_grid.png"    "$staging\results\"
Copy-Item "$root\Week3_Results\sample_annotations.png" "$staging\results\"
Copy-Item "$root\Week3_Results\yolo_results\confusion_matrix.png"            "$staging\results\"
Copy-Item "$root\Week3_Results\yolo_results\confusion_matrix_normalized.png" "$staging\results\"
Copy-Item "$root\Week3_Results\yolo_results\BoxPR_curve.png"                 "$staging\results\"
Copy-Item "$root\Week3_Results\yolo_results\BoxF1_curve.png"                 "$staging\results\"
Copy-Item "$root\Week3_Results\yolo_results\results.png"                     "$staging\results\"

# Detected images — renamed to generic random names
$detectedSrc = Get-ChildItem "$root\Week3_Results\prediction_samples\*.jpg" | Sort-Object { Get-Random }
$counter = 1
foreach ($file in $detectedSrc) {
    $newName = "detection_{0:D3}.jpg" -f $counter
    Copy-Item $file.FullName "$staging\detected_images\$newName"
    $counter++
}

# Build ZIP
if (Test-Path $zipOut) { Remove-Item $zipOut -Force }
Compress-Archive -Path $staging -DestinationPath $zipOut

Write-Host ""
Write-Host "Done! Week 3 submission ZIP created:" -ForegroundColor Green
Write-Host "  $zipOut" -ForegroundColor Cyan
$size = [math]::Round((Get-Item $zipOut).Length / 1MB, 2)
Write-Host "  Size: $size MB" -ForegroundColor Yellow
