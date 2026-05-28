[CmdletBinding()]
param(
    [string]$BaseUrl = $env:ORCHESTRATOR_BASE_URL_PUBLIC,
    [string]$ApiKey = $env:LITELLM_MASTER_KEY,
    [string]$Model = $env:PUBLIC_MODEL_NAME
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Import-DotEnv {
    param([string]$Path)

    if (-not (Test-Path -LiteralPath $Path)) {
        return
    }

    foreach ($line in Get-Content -LiteralPath $Path) {
        $trimmed = $line.Trim()
        if (-not $trimmed -or $trimmed.StartsWith("#")) {
            continue
        }

        $separatorIndex = $trimmed.IndexOf("=")
        if ($separatorIndex -lt 1) {
            continue
        }

        $name = $trimmed.Substring(0, $separatorIndex).Trim()
        $value = $trimmed.Substring($separatorIndex + 1).Trim().Trim('"').Trim("'")

        if (-not [Environment]::GetEnvironmentVariable($name, "Process")) {
            [Environment]::SetEnvironmentVariable($name, $value, "Process")
        }
    }
}

$root = Resolve-Path (Join-Path $PSScriptRoot "..")
Import-DotEnv -Path (Join-Path $root ".env")

if ([string]::IsNullOrWhiteSpace($BaseUrl)) {
    $BaseUrl = $env:ORCHESTRATOR_BASE_URL_PUBLIC
}
if ([string]::IsNullOrWhiteSpace($BaseUrl)) {
    $BaseUrl = "http://localhost:4100"
}

if ([string]::IsNullOrWhiteSpace($ApiKey)) {
    $ApiKey = $env:LITELLM_MASTER_KEY
}
if ([string]::IsNullOrWhiteSpace($ApiKey)) {
    $ApiKey = "sk-change-this-local-key"
}

if ([string]::IsNullOrWhiteSpace($Model)) {
    $Model = $env:PUBLIC_MODEL_NAME
}
if ([string]::IsNullOrWhiteSpace($Model)) {
    $Model = "local-main"
}

$uri = "$($BaseUrl.TrimEnd('/'))/v1/chat/completions"
$body = @{
    model = $Model
    messages = @(
        @{
            role = "user"
            content = "Return exactly: ok"
        }
    )
    temperature = 0
    max_tokens = 8
} | ConvertTo-Json -Depth 8

$headers = @{
    Authorization = "Bearer $ApiKey"
}

$response = Invoke-RestMethod -Method Post -Uri $uri -Headers $headers -ContentType "application/json" -Body $body
$text = $response.choices[0].message.content

Write-Host "Smoke test passed"
Write-Host "Model: $($response.model)"
Write-Host "Text: $text"
