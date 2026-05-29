#Requires -Version 5.1

param(
  [string]$BaseUrl = "http://localhost:4100",
  [string]$LifecycleUrl = "http://localhost:4300",
  [string]$Model = "mistralai/ministral-3-3b",
  [string]$Prompt = "Return exactly: ok",
  [int]$MaxTokens = 16,
  [double]$EstimatedVramGb = 6,
  [double]$SafetyMarginGb = 1,
  [int]$MaxParallel = 1,
  [int]$MaxQueuedRequests = 4,
  [int]$IdleTtlSeconds = 900,
  [int]$TimeoutSeconds = 240
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$Payload = @{
  model = $Model
  messages = @(
    @{
      role = "user"
      content = $Prompt
    }
  )
  temperature = 0
  max_tokens = $MaxTokens
  orchestration = @{
    gpu = "auto"
    warmup_enabled = $false
    estimated_vram_gb = $EstimatedVramGb
    safety_margin_gb = $SafetyMarginGb
    max_parallel = $MaxParallel
    max_queued_requests = $MaxQueuedRequests
    idle_ttl_seconds = $IdleTtlSeconds
  }
}

$Body = $Payload | ConvertTo-Json -Depth 10
$ChatUrl = "$($BaseUrl.TrimEnd('/'))/v1/chat/completions"

Write-Host "POST $ChatUrl" -ForegroundColor Cyan
Write-Host "Model: $Model"

$Response = Invoke-RestMethod `
  -Method Post `
  -Uri $ChatUrl `
  -ContentType "application/json" `
  -Body $Body `
  -TimeoutSec $TimeoutSeconds

Write-Host ""
Write-Host "Response:" -ForegroundColor Cyan
$Response | ConvertTo-Json -Depth 10

Write-Host ""
Write-Host "Registry:" -ForegroundColor Cyan
Invoke-RestMethod "$($LifecycleUrl.TrimEnd('/'))/registry" | ConvertTo-Json -Depth 10

Write-Host ""
Write-Host "Queue status:" -ForegroundColor Cyan
Invoke-RestMethod "$($BaseUrl.TrimEnd('/'))/status" | ConvertTo-Json -Depth 10
