#Requires -Version 5.1

param(
  [switch]$SkipVenv,
  [switch]$SkipTests,
  [switch]$SkipDockerPull,
  [switch]$SkipDockerBuild,
  [switch]$SkipGpuCheck,
  [switch]$SkipVllmPull,
  [string[]]$ExtraImages = @()
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

function Write-Step {
  param([string]$Message)

  Write-Host ""
  Write-Host "==> $Message" -ForegroundColor Cyan
}

function Invoke-Checked {
  param(
    [string]$FilePath,
    [string[]]$Arguments
  )

  Write-Host "+ $FilePath $($Arguments -join ' ')" -ForegroundColor DarkGray
  & $FilePath @Arguments
  if ($LASTEXITCODE -ne 0) {
    throw "Command failed with exit code $LASTEXITCODE`: $FilePath $($Arguments -join ' ')"
  }
}

$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

Write-Step "Preparing local environment"
if (-not (Test-Path -LiteralPath ".env")) {
  if (-not (Test-Path -LiteralPath ".env.example")) {
    throw ".env.example was not found"
  }

  Copy-Item -LiteralPath ".env.example" -Destination ".env"
  Write-Host "Created .env from .env.example"
} else {
  Write-Host ".env already exists"
}

if (-not $SkipVenv) {
  Write-Step "Installing Python development dependencies"
  if (-not (Test-Path -LiteralPath ".\.venv\Scripts\python.exe")) {
    Invoke-Checked "python" @("-m", "venv", ".venv")
  }

  Invoke-Checked ".\.venv\Scripts\python.exe" @("-m", "pip", "install", "--upgrade", "pip")
  Invoke-Checked ".\.venv\Scripts\python.exe" @("-m", "pip", "install", "-e", ".[dev]")
}

if (-not $SkipTests) {
  Write-Step "Running Python tests"
  if (-not (Test-Path -LiteralPath ".\.venv\Scripts\python.exe")) {
    throw ".venv is missing. Run without -SkipVenv or install dependencies manually."
  }

  Invoke-Checked ".\.venv\Scripts\python.exe" @("-m", "pytest")
}

if (-not $SkipDockerPull) {
  Write-Step "Pulling runtime images"
  $Images = @(
    "postgres:16-alpine",
    "redis:7-alpine",
    "docker.litellm.ai/berriai/litellm:main-latest",
    "prom/prometheus:v2.54.1",
    "grafana/grafana-oss:11.2.0",
    "nvidia/cuda:12.8.0-base-ubuntu22.04"
  )

  if (-not $SkipVllmPull) {
    $Images += "vllm/vllm-openai:latest"
  }

  $Images += $ExtraImages

  foreach ($Image in $Images) {
    Invoke-Checked "docker" @("pull", $Image)
  }
}

if (-not $SkipDockerBuild) {
  Write-Step "Building Compose services"
  Invoke-Checked "docker" @("compose", "build")
  Invoke-Checked "docker" @("compose", "--profile", "test", "build", "fake-backend")
}

if (-not $SkipGpuCheck) {
  Write-Step "Verifying NVIDIA GPU passthrough in Docker"
  Invoke-Checked "docker" @("run", "--rm", "--gpus", "all", "nvidia/cuda:12.8.0-base-ubuntu22.04", "nvidia-smi")
}

Write-Step "Done"
Write-Host "Next: set a real model artifact and volumes in config/orchestrator.yaml, then enable LIFECYCLE_DRY_RUN=false when you are ready for real vLLM launches."
