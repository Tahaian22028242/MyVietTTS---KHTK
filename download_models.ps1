# PowerShell helper to run the downloader inside your venv
# Usage: open PowerShell in project folder and run:
#   .\.venv\Scripts\Activate
#   .\download_models.ps1

# if (-Not (Test-Path ".\.venv\Scripts\Activate")) {
#     Write-Host "Virtual environment not found at .\.venv. Create or activate your venv first." -ForegroundColor Yellow
#     exit 1
# }

Write-Host "Installing huggingface_hub if missing..."
python -m pip install --upgrade huggingface_hub

Write-Host "Running downloader..."
python .\download_models.py

Write-Host "Done. Check ./models/ for downloaded files."