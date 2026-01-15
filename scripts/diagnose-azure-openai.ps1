[CmdletBinding()]
param(
	[Parameter(Mandatory = $false)]
	[string]$Endpoint,

	[Parameter(Mandatory = $false)]
	[string]$ApiKey,

	[Parameter(Mandatory = $false)]
	[string]$Deployment,

	[Parameter(Mandatory = $false)]
	[string[]]$ApiVersions = @(
		'2024-10-21',
		'2024-08-01-preview',
		'2024-06-01',
		'2024-02-15-preview',
		'2023-12-01-preview',
		'2023-07-01-preview',
		'2023-05-15'
	)
)

$ErrorActionPreference = 'Stop'

function Resolve-Setting([string]$Value, [string[]]$EnvNames) {
	if ($Value -and $Value.Trim().Length -gt 0) { return $Value }
	foreach ($n in $EnvNames) {
		$ev = [Environment]::GetEnvironmentVariable($n)
		if ($ev -and $ev.Trim().Length -gt 0) { return $ev }
	}
	return ''
}

function Require-NonEmpty([string]$Value, [string]$Name, [string]$Hint) {
	if (-not $Value -or $Value.Trim().Length -eq 0) {
		throw "Missing required value: $Name. $Hint"
	}
}

function Normalize-Endpoint([string]$RawEndpoint) {
	$e = $RawEndpoint.Trim().TrimEnd('/')
	$lower = $e.ToLowerInvariant()
	if ($lower.Contains('/api/projects/')) {
		throw 'This looks like an Azure AI Foundry project endpoint (contains "/api/projects/"). Use the Azure OpenAI resource endpoint (Keys & Endpoint) instead.'
	}
	# Convert resource endpoint to Azure OpenAI base URL expected by REST.
	if ($lower.EndsWith('/openai')) { return $e }
	return "$e/openai"
}

function Invoke-AOAIGet([string]$Url, [string]$Key) {
	$headers = @{ 'api-key' = $Key }
	return Invoke-RestMethod -Method Get -Uri $Url -Headers $headers -TimeoutSec 20
}

$Endpoint = Resolve-Setting $Endpoint @('AZURE_OPENAI_ENDPOINT', 'AZURE_OPENAI_BASE_URL')
$ApiKey = Resolve-Setting $ApiKey @('AZURE_OPENAI_API_KEY')
$Deployment = Resolve-Setting $Deployment @('AZURE_OPENAI_DEPLOYMENT', 'AZURE_OPENAI_DEPLOYMENT_NAME', 'AZURE_OPENAI_CHAT_DEPLOYMENT_NAME')

Require-NonEmpty $Endpoint 'Endpoint' 'Pass -Endpoint or set env AZURE_OPENAI_ENDPOINT (resource endpoint, not Foundry project URL).'
Require-NonEmpty $ApiKey 'ApiKey' 'Pass -ApiKey or set env AZURE_OPENAI_API_KEY.'

$base = Normalize-Endpoint $Endpoint

Write-Host "Testing Azure OpenAI endpoint:" -ForegroundColor Cyan
Write-Host "  Base URL: $base" -ForegroundColor Cyan

$workingVersion = $null
foreach ($v in $ApiVersions) {
	try {
		# /models is the lightest call that still validates api-version.
		$modelsUrl = "$base/models?api-version=$v"
		$null = Invoke-AOAIGet -Url $modelsUrl -Key $ApiKey
		$workingVersion = $v
		Write-Host "Supported api-version found: $v" -ForegroundColor Green
		break
	} catch {
		$msg = $_.Exception.Message
		if ($msg -match 'API version not supported') {
			Write-Host "Not supported: $v" -ForegroundColor DarkYellow
			continue
		}
		Write-Host "Failed for ${v}: $msg" -ForegroundColor Red
		continue
	}
}

if (-not $workingVersion) {
	throw 'Could not find a supported api-version from the default list. Open Azure AI Foundry / Azure OpenAI code samples for your deployment and copy the api-version shown there.'
}

try {
	$deploymentsUrl = "$base/deployments?api-version=$workingVersion"
	$deployments = Invoke-AOAIGet -Url $deploymentsUrl -Key $ApiKey
	$names = @()
	if ($deployments -and $deployments.data) {
		foreach ($d in $deployments.data) {
			if ($d -and $d.id) { $names += [string]$d.id }
		}
	}
	if ($names.Count -gt 0) {
		Write-Host "Deployments visible to this key (showing up to 25):" -ForegroundColor Cyan
		$names | Select-Object -First 25 | ForEach-Object { Write-Host "  - $_" }
	}

	if ($Deployment) {
		if ($names -contains $Deployment) {
			Write-Host "Deployment '$Deployment' is present." -ForegroundColor Green
		} else {
			Write-Host "Deployment '$Deployment' was NOT found in this resource." -ForegroundColor Yellow
			Write-Host "Check that AZURE_OPENAI_DEPLOYMENT matches one of the deployment ids above." -ForegroundColor Yellow
		}
	}
} catch {
	Write-Host "Could not list deployments: $($_.Exception.Message)" -ForegroundColor Yellow
	Write-Host 'This can happen if the key lacks permissions or the endpoint is not an Azure OpenAI resource endpoint.' -ForegroundColor Yellow
}

Write-Host ''
Write-Host 'Recommended App Service settings:' -ForegroundColor Cyan
Write-Host "  AZURE_OPENAI_ENDPOINT=$Endpoint"
Write-Host "  AZURE_OPENAI_API_VERSION=$workingVersion"
if ($Deployment) {
	Write-Host "  AZURE_OPENAI_DEPLOYMENT=$Deployment"
} else {
	Write-Host '  AZURE_OPENAI_DEPLOYMENT=<your deployment name>'
}
Write-Host '  AZURE_OPENAI_API_KEY=<secret>'
