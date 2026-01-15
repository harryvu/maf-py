[CmdletBinding()]
param(
  [Parameter(Mandatory = $false)]
  [string] $Location = "westus2",

  [Parameter(Mandatory = $false)]
  [string] $ParametersFile = "infra/parameters/dev.bicepparam",

  [Parameter(Mandatory = $false)]
  [string] $TemplateFile = "infra/main.bicep",

  [Parameter(Mandatory = $false)]
  [string] $DeploymentName = "maf-py-appservice-$(Get-Date -Format 'yyyyMMdd-HHmmss')",

  [Parameter(Mandatory = $false)]
  [switch] $WhatIf
)

$ErrorActionPreference = 'Stop'

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path

function Assert-CommandExists {
  param([Parameter(Mandatory = $true)][string] $Name)
  if (-not (Get-Command $Name -ErrorAction SilentlyContinue)) {
    throw "Required command not found: $Name"
  }
}

Assert-CommandExists -Name 'az'

Write-Host "Using template: $TemplateFile" -ForegroundColor Cyan
Write-Host "Using parameters: $ParametersFile" -ForegroundColor Cyan
Write-Host "Deployment name: $DeploymentName" -ForegroundColor Cyan
Write-Host "Location: $Location" -ForegroundColor Cyan
Write-Host "Repo root: $RepoRoot" -ForegroundColor Cyan

function Resolve-RepoPath {
  param([Parameter(Mandatory = $true)][string] $Path)

  if ($Path -match '^\s*$') {
    throw 'Path is empty'
  }

  if (Split-Path -Path $Path -IsAbsolute) {
    return (Resolve-Path $Path).Path
  }

  return (Resolve-Path (Join-Path $RepoRoot $Path)).Path
}

function Resolve-DeploymentParametersArg {
  param(
    [Parameter(Mandatory = $true)]
    [string] $ParametersFilePath
  )

  $ParametersFilePath = Resolve-RepoPath -Path $ParametersFilePath

  if (-not (Test-Path $ParametersFilePath)) {
    throw "Parameters file not found: $ParametersFilePath"
  }

  if ($ParametersFilePath -like '*.bicepparam') {
    $tempOut = Join-Path -Path $env:TEMP -ChildPath ("maf-py-params-" + (Get-Date -Format 'yyyyMMdd-HHmmss') + ".parameters.json")
    Write-Host "Compiling .bicepparam to ARM JSON: $tempOut" -ForegroundColor Cyan

    az bicep build-params --file $ParametersFilePath --outfile $tempOut | Out-Null
    return "@$tempOut"
  }

  # For .json parameters files, az expects the @ prefix to load from disk.
  if ($ParametersFilePath -like '*.json') {
    return "@$ParametersFilePath"
  }

  # Fallback: pass through as-is (allows inline `name=value` forms).
  return $ParametersFilePath
}

# Ensure we're logged in
try {
  $null = az account show --output none
} catch {
  Write-Host "Azure CLI not logged in; running az login..." -ForegroundColor Yellow
  az login | Out-Null
}

# Deploy at subscription scope so we can create the resource group too.
# Note: --location here is required for subscription-scope deployments.
Write-Host "Starting subscription deployment..." -ForegroundColor Cyan

$TemplateFile = Resolve-RepoPath -Path $TemplateFile
$paramsArg = Resolve-DeploymentParametersArg -ParametersFilePath $ParametersFile

$deployArgs = @(
  'deployment', 'sub', 'create',
  '--name', $DeploymentName,
  '--location', $Location,
  '--template-file', $TemplateFile,
  '--parameters', $paramsArg
)

if ($WhatIf) {
  $deployArgs += '--what-if'
}

az @deployArgs

Write-Host "Deployment complete." -ForegroundColor Green
Write-Host "Outputs:" -ForegroundColor Green
az deployment sub show --name $DeploymentName --query properties.outputs --output json
