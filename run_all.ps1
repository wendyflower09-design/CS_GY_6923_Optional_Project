# ==========================================
# Run Part 2, Part 3, Part 4
# ==========================================

Write-Host "==================================="
Write-Host "Running Part 2: Transformer Scaling"
Write-Host "==================================="

python transformer_scaling.py

if ($LASTEXITCODE -ne 0) {
    throw "Part 2 failed. Stop execution."
}


Write-Host ""
Write-Host "==================================="
Write-Host "Running Part 3: uP Scaling"
Write-Host "==================================="

python uP_scaling.py

if ($LASTEXITCODE -ne 0) {
    throw "Part 3 failed. Stop execution."
}


Write-Host ""
Write-Host "==================================="
Write-Host "Running Part 4: Best Model Training"
Write-Host "==================================="

python best_model_train.py

if ($LASTEXITCODE -ne 0) {
    throw "Part 4 failed."
}


Write-Host ""
Write-Host "==================================="
Write-Host "All Parts Finished Successfully"
Write-Host "==================================="