# Azure Infrastructure (Bicep)

This folder provisions Azure resources for the Next.js app using Bicep:

- Resource Group
- App Service Plan (Linux)
- App Service Web App (Linux)

The app is deployed separately via GitHub Actions.

## Files

- `infra/main.bicep`: subscription-scope entrypoint (creates the RG and calls `app.bicep`)
- `infra/app.bicep`: resource-group-scope resources (plan + web app + app settings)
- `infra/parameters/dev.bicepparam`: example parameter file

## Deploy from your machine (PowerShell)

From repo root:

- Run: `./scripts/deploy-azure-appservice.ps1`

From `scripts/`:

- Run: `./deploy-azure-appservice.ps1`

To override defaults:

- `./scripts/deploy-azure-appservice.ps1 -Location westus2 -ParametersFile infra/parameters/dev.bicepparam`

Note: the script compiles `.bicepparam` files to ARM JSON parameters automatically before running the deployment.

## After provisioning

- The template sets the Web App Startup Command to `node server.js` (Linux). Until you deploy the app, the site may show **Application Error** because `server.js` does not exist yet.
- After the first successful deployment, browsing the site should work normally.
- Configure any app settings you need (e.g. Azure OpenAI env vars) under App Service → Configuration.

### Configure Azure OpenAI (Real LLM mode)

Real LLM mode requires these App Service settings:

- `AZURE_OPENAI_ENDPOINT` (e.g. `https://<resource>.openai.azure.com/`)
- `AZURE_OPENAI_DEPLOYMENT` (your chat deployment name)
- `AZURE_OPENAI_API_VERSION` (e.g. `2024-06-01`)
- `AZURE_OPENAI_API_KEY` (secret)

Helper script (recommended for dev):

- `./scripts/set-azure-openai-appsettings.ps1 -ResourceGroup maf-py-dev-rg -WebAppName maf-py-dev-web -Endpoint https://... -Deployment ... -ApiVersion 2024-06-01 -ApiKey ...`
