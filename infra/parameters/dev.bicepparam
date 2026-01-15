using '../main.bicep'

param location = 'westus2'
param resourceGroupName = 'maf-py-dev-rg'
param appServicePlanName = 'maf-py-dev-plan'
param webAppName = 'maf-py-dev-web'

// Basic B1 is a good default for dev. Use F1 for free tier (if available) or S1+ for Always On.
param skuName = 'B1'
param skuTier = 'Basic'
param skuCapacity = 1

// Next.js standalone startup
param linuxFxVersion = 'NODE|20-lts'
param appCommandLine = 'node server.js'

param appSettings = {
  // If you want to enable Real LLM mode, set these in Azure App Settings (recommended via Key Vault in prod).
  // AZURE_OPENAI_API_KEY: '...'
  // AZURE_OPENAI_RESOURCE_NAME: '...'
  // AZURE_OPENAI_DEPLOYMENT: '...'
  AZURE_OPENAI_API_VERSION: 'preview'
}
