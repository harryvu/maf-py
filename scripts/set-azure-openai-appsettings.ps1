[CmdletBinding(SupportsShouldProcess = $true, ConfirmImpact = 'Medium')]
param(
	[Parameter(Mandatory = $false)]
	[string]$ResourceGroup,

	[Parameter(Mandatory = $false)]
	[string]$WebAppName,

	[Parameter(Mandatory = $false)]
	[string]$Endpoint,

	[Parameter(Mandatory = $false)]
	[string]$Deployment,

	[Parameter(Mandatory = $false)]
	[string]$ApiVersion = '2024-06-01',

	[Parameter(Mandatory = $false)]
	[string]$ApiKey
)

$ErrorActionPreference = 'Stop'

function Get-EnvValue([string]$Name) {
	try {
		$item = Get-Item -Path ("Env:{0}" -f $Name) -ErrorAction SilentlyContinue
		if ($null -ne $item -and $item.Value -match '\S') { return $item.Value }
	} catch {
		# ignore
	}
	return ''
}

function Resolve-Setting([string]$Value, [string[]]$EnvNames) {
	if ($Value -match '\S') { return $Value }
	foreach ($n in $EnvNames) {
		$ev = Get-EnvValue $n
		if ($ev -match '\S') { return $ev }
	}
	return ''
}

function Require-NonEmpty([string]$Value, [string]$Name, [string]$Hint) {
	if (-not ($Value -match '\S')) {
		throw "Missing required value: $Name. $Hint"
	}
}

$ResourceGroup = Resolve-Setting $ResourceGroup @('AZURE_RESOURCE_GROUP', 'AZURE_WEBAPP_RESOURCE_GROUP')
$WebAppName = Resolve-Setting $WebAppName @('AZURE_WEBAPP_NAME')

$Endpoint = Resolve-Setting $Endpoint @('AZURE_OPENAI_ENDPOINT')
$Deployment = Resolve-Setting $Deployment @('AZURE_OPENAI_DEPLOYMENT', 'AZURE_OPENAI_DEPLOYMENT_NAME', 'AZURE_OPENAI_CHAT_DEPLOYMENT_NAME')
$ApiVersion = Resolve-Setting $ApiVersion @('AZURE_OPENAI_API_VERSION')
$ApiKey = Resolve-Setting $ApiKey @('AZURE_OPENAI_API_KEY')

Require-NonEmpty $ResourceGroup 'ResourceGroup' 'Pass -ResourceGroup or set env AZURE_RESOURCE_GROUP.'
Require-NonEmpty $WebAppName 'WebAppName' 'Pass -WebAppName or set env AZURE_WEBAPP_NAME.'
Require-NonEmpty $Endpoint 'Endpoint' 'Pass -Endpoint or set env AZURE_OPENAI_ENDPOINT (e.g. https://<resource>.openai.azure.com/).'
Require-NonEmpty $Deployment 'Deployment' 'Pass -Deployment or set env AZURE_OPENAI_DEPLOYMENT.'
Require-NonEmpty $ApiKey 'ApiKey' 'Pass -ApiKey or set env AZURE_OPENAI_API_KEY.'
Require-NonEmpty $ApiVersion 'ApiVersion' 'Pass -ApiVersion or set env AZURE_OPENAI_API_VERSION (e.g. 2024-06-01).'

$az = Get-Command az -ErrorAction SilentlyContinue
if (-not $az) {
	throw 'Azure CLI (az) not found. Install Azure CLI, then run: az login'
}

$settings = @(
	"AZURE_OPENAI_ENDPOINT=$Endpoint",
	"AZURE_OPENAI_DEPLOYMENT=$Deployment",
	"AZURE_OPENAI_API_VERSION=$ApiVersion",
	"AZURE_OPENAI_API_KEY=$ApiKey"
)

if ($PSCmdlet.ShouldProcess("$WebAppName in $ResourceGroup", 'Set Azure OpenAI app settings')) {
	# Avoid echoing secrets by suppressing JSON output.
	& az webapp config appsettings set `
		--resource-group $ResourceGroup `
		--name $WebAppName `
		--settings @settings `
		--only-show-errors `
		--output none
}

Write-Host "Configured Azure OpenAI settings on web app '$WebAppName'." -ForegroundColor Green
Write-Host 'If the app is still running, restart it:'
Write-Host "  az webapp restart --resource-group $ResourceGroup --name $WebAppName"