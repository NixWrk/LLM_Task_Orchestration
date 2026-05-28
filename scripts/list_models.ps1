[CmdletBinding()]
param(
    [string]$BaseUrl = "http://localhost:1234/v1",
    [string]$ApiKey = "lm-studio"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$headers = @{}
if (-not [string]::IsNullOrWhiteSpace($ApiKey)) {
    $headers.Authorization = "Bearer $ApiKey"
}

$uri = "$($BaseUrl.TrimEnd('/'))/models"
$response = Invoke-RestMethod -Method Get -Uri $uri -Headers $headers
$response.data | Select-Object id, object, owned_by | Format-Table -AutoSize
