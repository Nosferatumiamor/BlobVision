# BlobVision
$Root = Split-Path $PSScriptRoot -Parent
$Python = Join-Path $Root "venv\Scripts\python.exe"
Set-Location $PSScriptRoot

function Stop-BlobVisionInstances {
    $stopped = 0
    Get-CimInstance Win32_Process -Filter "Name='python.exe'" -ErrorAction SilentlyContinue |
        Where-Object { $_.CommandLine -like '*blobvision_ui.py*' -or $_.CommandLine -like '*blobvision_app.py*' } |
        ForEach-Object {
            Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue
            $stopped++
        }
    if ($stopped -gt 0) {
        Write-Host "Ancienne instance BlobVision fermee ($stopped processus)."
        Start-Sleep -Seconds 1
    }
}

Stop-BlobVisionInstances

$env:GRADIO_SSR_MODE = "false"
$env:BLOBVISION_SKIP_WARMUP = "1"
$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"
# Force SDXL into CPU-offload mode even on GPUs with "enough" VRAM to
# auto-select full-GPU (see resolve_sketch_full_gpu). Full-GPU keeps SDXL's
# ~6.5GB block resident, which forces a slow explicit park/unpark against
# VQGAN on every redux/style-preset generation and leaves so little headroom
# that VQGAN's own optimization loop (Adam + backprop through CLIP) measured
# 5-9x slower right at the edge of OOM. CPU offload keeps SDXL's weights in
# system RAM and only touches GPU per denoising step, measured at full VQGAN
# speed with several GB of headroom to spare — a small per-step SDXL cost
# that's far cheaper than the alternative given BlobVision always needs SDXL
# to coexist with VQGAN/DeepDream/Style Transfer, never standalone.
$env:BLOBVISION_SKETCH_FULL_GPU = "0"
if (-not $env:NO_PROXY) { $env:NO_PROXY = "127.0.0.1,localhost,<local>" }
& $Python -u (Join-Path $PSScriptRoot "python\blobvision_app.py") @args
exit $LASTEXITCODE
