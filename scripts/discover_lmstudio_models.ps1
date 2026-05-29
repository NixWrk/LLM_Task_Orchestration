#Requires -Version 5.1

param(
  [string]$BaseUrl = $(if ($env:LMSTUDIO_OPENAI_BASE_URL) { $env:LMSTUDIO_OPENAI_BASE_URL } else { "http://localhost:1234/v1" }),
  [string]$ApiKey = $(if ($env:LMSTUDIO_API_KEY) { $env:LMSTUDIO_API_KEY } else { "lm-studio" }),
  [string[]]$ModelRoots = @(),
  [string]$OutputPath = "",
  [switch]$SkipHttp,
  [switch]$SkipLmsCli,
  [switch]$SkipFilesystem
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$ModelsById = @{}
$Notes = New-Object System.Collections.Generic.List[string]

function Add-Model {
  param(
    [string]$Id,
    [string]$Source,
    [string]$Path = ""
  )

  if ([string]::IsNullOrWhiteSpace($Id)) {
    return
  }

  $CleanId = $Id.Trim()
  if (-not $ModelsById.ContainsKey($CleanId)) {
    $ModelsById[$CleanId] = [pscustomobject]@{
      id = $CleanId
      sources = @()
      paths = @()
    }
  }

  if ($ModelsById[$CleanId].sources -notcontains $Source) {
    $ModelsById[$CleanId].sources += $Source
  }

  if ($Path -and $ModelsById[$CleanId].paths -notcontains $Path) {
    $ModelsById[$CleanId].paths += $Path
  }
}

function Add-IdsFromObject {
  param(
    [object]$Value,
    [string]$Source
  )

  if ($null -eq $Value) {
    return
  }

  if ($Value -is [System.Array]) {
    foreach ($Item in $Value) {
      Add-IdsFromObject -Value $Item -Source $Source
    }
    return
  }

  if ($Value -is [string]) {
    return
  }

  $Properties = $Value.PSObject.Properties
  foreach ($Name in @("id", "modelKey", "key", "identifier")) {
    $Property = $Properties[$Name]
    if ($null -ne $Property -and -not [string]::IsNullOrWhiteSpace([string]$Property.Value)) {
      Add-Model -Id ([string]$Property.Value) -Source $Source
    }
  }

  foreach ($Property in $Properties) {
    $PropertyValue = $Property.Value
    if (
      $PropertyValue -is [System.Array] -or
      ($null -ne $PropertyValue -and -not ($PropertyValue -is [string]) -and @($PropertyValue.PSObject.Properties).Count -gt 0)
    ) {
      Add-IdsFromObject -Value $PropertyValue -Source $Source
    }
  }
}

function Get-DefaultModelRoots {
  $Candidates = @(
    "$env:USERPROFILE\.lmstudio\models",
    "$env:USERPROFILE\.cache\lm-studio\models",
    "$env:USERPROFILE\.cache\lmstudio\models",
    "$env:LOCALAPPDATA\LM Studio\models",
    "$env:APPDATA\LM Studio\models"
  )

  return $Candidates | Where-Object { $_ -and (Test-Path -LiteralPath $_) } | Select-Object -Unique
}

function Get-RelativeModelId {
  param(
    [string]$Root,
    [string]$Path,
    [switch]$DropExtension
  )

  $RootPath = (Resolve-Path -LiteralPath $Root).Path.TrimEnd("\", "/")
  $FullPath = (Resolve-Path -LiteralPath $Path).Path
  $Relative = $FullPath.Substring($RootPath.Length).TrimStart("\", "/")
  if ($DropExtension) {
    $Relative = [System.IO.Path]::ChangeExtension($Relative, $null)
  }
  return $Relative.Replace("\", "/")
}

if (-not $SkipHttp) {
  try {
    $Uri = "$($BaseUrl.TrimEnd('/'))/models"
    $Headers = @{ Authorization = "Bearer $ApiKey" }
    $Payload = Invoke-RestMethod -Method Get -Uri $Uri -Headers $Headers -TimeoutSec 5
    foreach ($Item in @($Payload.data)) {
      Add-Model -Id ([string]$Item.id) -Source "openai-api"
    }
  } catch {
    $Notes.Add("openai-api unavailable: $($_.Exception.Message)")
  }
}

if (-not $SkipLmsCli) {
  try {
    $RawOutput = & lms ls --json 2>&1
    if ($LASTEXITCODE -ne 0) {
      $Notes.Add("lms cli unavailable: $($RawOutput -join ' ')")
    } else {
      $JsonText = $RawOutput -join "`n"
      if (-not [string]::IsNullOrWhiteSpace($JsonText)) {
        $Payload = $JsonText | ConvertFrom-Json
        Add-IdsFromObject -Value $Payload -Source "lms-cli"
      }
    }
  } catch {
    $Notes.Add("lms cli unavailable: $($_.Exception.Message)")
  }
}

if (-not $SkipFilesystem) {
  $Roots = @($ModelRoots) + @(Get-DefaultModelRoots)
  foreach ($Root in ($Roots | Where-Object { $_ } | Select-Object -Unique)) {
    if (-not (Test-Path -LiteralPath $Root)) {
      $Notes.Add("model root not found: $Root")
      continue
    }

    $Files = Get-ChildItem -LiteralPath $Root -Recurse -File -Include *.gguf,config.json -ErrorAction SilentlyContinue
    foreach ($File in $Files) {
      if ($File.Name -ieq "config.json") {
        $Id = Get-RelativeModelId -Root $Root -Path $File.DirectoryName
        Add-Model -Id $Id -Source "filesystem" -Path $File.DirectoryName
      } else {
        $Id = Get-RelativeModelId -Root $Root -Path $File.FullName -DropExtension
        Add-Model -Id $Id -Source "filesystem" -Path $File.FullName
      }
    }
  }
}

$Models = @(@($ModelsById.Values) | Sort-Object id)
$Result = [ordered]@{
  generated_at = (Get-Date).ToUniversalTime().ToString("o")
  base_url = $BaseUrl
  model_count = $Models.Count
  models = $Models
  notes = @($Notes)
}

$Json = $Result | ConvertTo-Json -Depth 10

Write-Host "Discovered $($Models.Count) LM Studio model ids." -ForegroundColor Cyan
if ($Models.Count -gt 0) {
  $Models | Select-Object id, sources | Format-Table -AutoSize
}
if ($Notes.Count -gt 0) {
  Write-Host ""
  Write-Host "Notes:" -ForegroundColor Yellow
  foreach ($Note in $Notes) {
    Write-Host "- $Note"
  }
}

if ($OutputPath) {
  $Parent = Split-Path -Parent $OutputPath
  if ($Parent -and -not (Test-Path -LiteralPath $Parent)) {
    New-Item -ItemType Directory -Path $Parent | Out-Null
  }
  Set-Content -LiteralPath $OutputPath -Value $Json -Encoding UTF8
  Write-Host ""
  Write-Host "Wrote $OutputPath"
} else {
  Write-Host ""
  Write-Output $Json
}
