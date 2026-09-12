# Bootstraps a working Python environment from scratch on a machine that has
# never run BlobVision before -- no system Python, no venv, nothing. This is
# what makes the app installable on someone else's machine instead of only
# working here where a virtualenv-created venv/ already exists (that venv's
# own python.exe is NOT relocatable -- see tauri/src-tauri/src/main.rs's
# repo_root() comments for the exact reason this session already ran into).
#
# Produces a FLAT-layout venv/ (python.exe and pythonw.exe directly under
# venv/, not venv/Scripts/) using Python's own official "embeddable package"
# distribution, which is specifically built to be relocatable -- unlike a
# regular venv/virtualenv. Scripts/ still ends up existing underneath it
# (pip and other console-script entry points land there once installed),
# but the interpreter itself does not, which is why main.rs checks BOTH
# venv/python.exe (this layout) and venv/Scripts/python.exe (the existing
# dev machine's virtualenv layout) rather than assuming one or the other.
#
# Run manually to test in isolation: powershell -File bootstrap_venv.ps1 -TargetDir <path>
# The real caller (main.rs) points -TargetDir at the real venv/ when it's missing.

param(
    [string]$TargetDir = (Join-Path (Split-Path -Parent (Split-Path -Parent $PSScriptRoot)) "venv"),
    [string]$PythonVersion = "3.10.11",
    # torch/torchaudio share one version number; torchvision uses its own
    # (0.26.0 alongside torch 2.11.0 here) -- these three came straight off
    # the real working venv's own `pip freeze`, not guessed.
    [string]$TorchVersion = "2.11.0",
    [string]$TorchvisionVersion = "0.26.0",
    [string]$CudaTag = "cu128"
)

$ErrorActionPreference = "Stop"

function Say($msg) {
    Write-Host "[bootstrap] $msg"
}

if ((Test-Path (Join-Path $TargetDir "python.exe")) -or (Test-Path (Join-Path $TargetDir "Scripts\python.exe"))) {
    Say "Python environment already present at $TargetDir -- nothing to do."
    exit 0
}

$scriptRoot = $PSScriptRoot
$repoRoot = Split-Path -Parent (Split-Path -Parent $scriptRoot)
$requirementsPath = Join-Path $repoRoot "app\requirements-frozen.txt"
if (-not (Test-Path $requirementsPath)) {
    throw "Missing $requirementsPath -- can't bootstrap without the frozen package list."
}

$tempDir = Join-Path $env:TEMP ("blobvision-bootstrap-" + [guid]::NewGuid().ToString("N"))
New-Item -ItemType Directory -Path $tempDir -Force | Out-Null
try {
    $embedZipUrl = "https://www.python.org/ftp/python/$PythonVersion/python-$PythonVersion-embed-amd64.zip"
    $embedZipPath = Join-Path $tempDir "python-embed.zip"
    Say "Downloading Python $PythonVersion (embeddable, ~10 MB) from python.org..."
    Invoke-WebRequest -Uri $embedZipUrl -OutFile $embedZipPath -UseBasicParsing

    Say "Extracting to $TargetDir ..."
    New-Item -ItemType Directory -Path $TargetDir -Force | Out-Null
    Expand-Archive -Path $embedZipPath -DestinationPath $TargetDir -Force

    # The embeddable package ships with site-packages import disabled by
    # default (python*._pth has "import site" commented out) -- pip and any
    # installed package would otherwise be invisible to the interpreter.
    $pthFile = Get-ChildItem -Path $TargetDir -Filter "python*._pth" | Select-Object -First 1
    if (-not $pthFile) {
        throw "Could not find the embeddable package's ._pth file under $TargetDir"
    }
    Say "Enabling site-packages in $($pthFile.Name)..."
    (Get-Content $pthFile.FullName) -replace '^#\s*import site', 'import site' | Set-Content $pthFile.FullName

    $pythonExe = Join-Path $TargetDir "python.exe"

    Say "Downloading get-pip.py..."
    $getPipPath = Join-Path $tempDir "get-pip.py"
    Invoke-WebRequest -Uri "https://bootstrap.pypa.io/get-pip.py" -OutFile $getPipPath -UseBasicParsing

    Say "Installing pip..."
    & $pythonExe $getPipPath --no-warn-script-location
    if ($LASTEXITCODE -ne 0) { throw "get-pip.py failed (exit $LASTEXITCODE)" }

    # torch/torchvision/torchaudio aren't in requirements-frozen.txt on
    # purpose -- they need PyTorch's own CUDA-specific index, not plain
    # PyPI, and the right build depends on what this app already assumes
    # about the target machine (an NVIDIA GPU -- see resolve_sketch_full_gpu
    # in blobvision_engine.py, which has no real CPU-only path anyway).
    Say "Installing PyTorch $TorchVersion ($CudaTag build) -- this is the big one, several GB..."
    & $pythonExe -m pip install --no-warn-script-location `
        "torch==$TorchVersion+$CudaTag" "torchvision==$TorchvisionVersion+$CudaTag" "torchaudio==$TorchVersion+$CudaTag" `
        --index-url "https://download.pytorch.org/whl/$CudaTag"
    if ($LASTEXITCODE -ne 0) { throw "PyTorch install failed (exit $LASTEXITCODE)" }

    Say "Installing the rest of the frozen requirements (~150 packages)..."
    & $pythonExe -m pip install --no-warn-script-location -r $requirementsPath
    if ($LASTEXITCODE -ne 0) { throw "requirements-frozen.txt install failed (exit $LASTEXITCODE)" }

    Say "Bootstrap complete: $TargetDir is a working Python environment."
}
finally {
    Remove-Item -Recurse -Force $tempDir -ErrorAction SilentlyContinue
}
