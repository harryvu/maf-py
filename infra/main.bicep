targetScope = 'subscription'

@description('Azure region for the deployment and the resource group.')
param location string

@description('Name of the resource group to create/use for the App Service resources.')
param resourceGroupName string

@description('Name of the App Service Plan (Linux).')
param appServicePlanName string

@description('Name of the App Service Web App.')
param webAppName string

@description('SKU name for the App Service Plan (e.g., F1, B1, S1, P0v3).')
param skuName string = 'B1'

@description('SKU tier for the App Service Plan (e.g., Free, Basic, Standard, PremiumV3).')
param skuTier string = 'Basic'

@description('Instance count for the App Service Plan.')
@minValue(1)
param skuCapacity int = 1

@description('Node runtime for Linux App Service (e.g., NODE|20-lts).')
param linuxFxVersion string = 'NODE|20-lts'

@description('Startup command for the Web App. For Next.js standalone deployments, use "node server.js".')
param appCommandLine string = 'node server.js'

@description('Additional app settings to add to the Web App (key/value pairs).')
param appSettings object = {}

resource rg 'Microsoft.Resources/resourceGroups@2022-09-01' = {
  name: resourceGroupName
  location: location
}

module app 'app.bicep' = {
  name: 'appServiceDeployment'
  scope: rg
  params: {
    location: location
    appServicePlanName: appServicePlanName
    webAppName: webAppName
    skuName: skuName
    skuTier: skuTier
    skuCapacity: skuCapacity
    linuxFxVersion: linuxFxVersion
    appCommandLine: appCommandLine
    appSettings: appSettings
  }
}

output deployedResourceGroupName string = rg.name
output deployedWebAppName string = webAppName
output webAppDefaultHostName string = app.outputs.defaultHostName
