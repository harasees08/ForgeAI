param(
    [switch]$WithOllama,
    [string]$OllamaModel = "qwen3:4b"
)

$ErrorActionPreference = "Stop"
$project = Split-Path -Parent $PSScriptRoot

function Invoke-Checked {
    param(
        [string]$Executable,
        [string[]]$Arguments
    )
    & $Executable @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$Executable exited with code $LASTEXITCODE."
    }
}

Push-Location $project
try {
    $venvPython = Join-Path $project ".venv\Scripts\python.exe"
    if (-not (Test-Path -LiteralPath $venvPython)) {
        $existingPython = Join-Path $env:USERPROFILE "AppData\Local\Python\pythoncore-3.14-64\python.exe"
        if (Test-Path -LiteralPath $existingPython) {
            Invoke-Checked $existingPython @("-m", "venv", ".venv")
        } else {
            Invoke-Checked "py" @("-3.14", "-m", "venv", ".venv")
        }
    }

    $pythonCheck = 'import platform, sys; sys.exit(sys.version_info[:2] != (3, 14) or platform.machine().lower() != "amd64" or sys.prefix == sys.base_prefix)'
    Invoke-Checked $venvPython @("-c", $pythonCheck)
    Invoke-Checked $venvPython @(
        "-m", "pip", "install", "--index-url", "https://download.pytorch.org/whl/cu128", "torch==2.11.0+cu128"
    )
    Invoke-Checked $venvPython @("-m", "pip", "install", "--index-url", "https://pypi.org/simple", "-r", "requirements.txt")
    Invoke-Checked $venvPython @("-m", "pip", "check")
    Invoke-Checked $venvPython @("-m", "pytest", "-q")

    $checks = @("python", "cuda-driver", "telemetry", "pdf", "torch", "torch-cuda", "embeddings")
    if (Test-Path -LiteralPath ".git") {
        $checks += "git"
    }
    if ($WithOllama) {
        Invoke-Checked "ollama" @("pull", $OllamaModel)
        $checks += @("ollama-inventory", "ollama")
    }
    Invoke-Checked $venvPython (@(
        "-m", "scripts.smoke", "--output", "experiments/raw_results/phase0-smoke.json",
        "--ollama-model", $OllamaModel
    ) + $checks)
} finally {
    Pop-Location
}
