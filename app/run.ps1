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
if (-not $env:NO_PROXY) { $env:NO_PROXY = "127.0.0.1,localhost,<local>" }
& $Python -u (Join-Path $PSScriptRoot "python\blobvision_app.py") @args
exit $LASTEXITCODE
