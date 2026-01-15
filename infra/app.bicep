targetScope = 'resourceGroup'

@description('Azure region for the App Service resources.')
param location string

@description('Name of the App Service Plan (Linux).')
param appServicePlanName string

@description('Name of the App Service Web App.')
param webAppName string

@description('SKU name for the App Service Plan (e.g., F1, B1, S1, P0v3).')
param skuName string

@description('SKU tier for the App Service Plan (e.g., Free, Basic, Standard, PremiumV3).')
param skuTier string

@description('Instance count for the App Service Plan.')
@minValue(1)
param skuCapacity int

@description('Node runtime for Linux App Service (e.g., NODE|20-lts).')
param linuxFxVersion string

@description('Startup command for the Web App. For Next.js standalone deployments, use "node server.js".')
param appCommandLine string

@description('Additional app settings to add to the Web App (key/value pairs).')
param appSettings object

var defaultAppSettings = {
  NODE_ENV: 'production'
}

resource plan 'Microsoft.Web/serverfarms@2022-09-01' = {
  name: appServicePlanName
  location: location
  sku: {
    name: skuName
    tier: skuTier
    capacity: skuCapacity
  }
  properties: {
    reserved: true
  }
}

resource web 'Microsoft.Web/sites@2022-09-01' = {
  name: webAppName
  location: location
  kind: 'app,linux'
  identity: {
    type: 'SystemAssigned'
  }
  properties: {
    httpsOnly: true
    serverFarmId: plan.id
    clientAffinityEnabled: false
    siteConfig: {
      linuxFxVersion: linuxFxVersion
      appCommandLine: appCommandLine
      ftpsState: 'Disabled'
      minTlsVersion: '1.2'
      scmMinTlsVersion: '1.2'
      http20Enabled: true
    }
  }
}

resource appSettingsConfig 'Microsoft.Web/sites/config@2022-09-01' = {
  name: 'appsettings'
  parent: web
  properties: union(defaultAppSettings, appSettings)
}

output defaultHostName string = web.properties.defaultHostName
