Write-Host "Installing dependencies..." -ForegroundColor Cyan
pip install gradio ultralytics opencv-python -q

Write-Host "`nLaunching Pain Expression Detector..." -ForegroundColor Green
python "$PSScriptRoot\app.py"
