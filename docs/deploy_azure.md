# Deploying to Azure Container Apps

Documented steps to build the Docker image, push it to Azure Container
Registry (ACR), and run it as an Azure Container App. These commands are a
reference — run them yourself with your own subscription; nothing here is
executed automatically.

## Prerequisites

Azure CLI (`az`) logged in (`az login`), Docker installed, and this repo
checked out.

## 1. Create a resource group and registry

```bash
az group create --name moderation-rg --location westeurope

az acr create --resource-group moderation-rg \
  --name <yourregistry> --sku Basic --admin-enabled true
```

## 2. Build and push the image

Build in the cloud (no local Docker needed) with ACR Tasks:

```bash
az acr build --registry <yourregistry> \
  --image moderation-service:v1 \
  --file docker/Dockerfile .
```

Or build locally and push:

```bash
docker build -f docker/Dockerfile -t <yourregistry>.azurecr.io/moderation-service:v1 .
az acr login --name <yourregistry>
docker push <yourregistry>.azurecr.io/moderation-service:v1
```

## 3. Create the Container Apps environment and app

```bash
az extension add --name containerapp --upgrade

az containerapp env create \
  --name moderation-env \
  --resource-group moderation-rg \
  --location westeurope

az containerapp create \
  --name moderation-service \
  --resource-group moderation-rg \
  --environment moderation-env \
  --image <yourregistry>.azurecr.io/moderation-service:v1 \
  --registry-server <yourregistry>.azurecr.io \
  --target-port 8000 \
  --ingress external \
  --cpu 1.0 --memory 2.0Gi \
  --min-replicas 0 --max-replicas 2
```

`--min-replicas 0` scales to zero when idle, which keeps a demo deployment
close to free.

## 4. Configuration

Offline demo mode (default): no env vars needed — stub classifier + mock LLM.

Real mode: set env vars (and mount/copy the checkpoint into the image or an
Azure Files share):

```bash
az containerapp update \
  --name moderation-service \
  --resource-group moderation-rg \
  --set-env-vars MODEL_DIR=/model MODEL_NAME=roberta-base \
                 LLM_BASE_URL=<endpoint>/v1 LLM_MODEL=<model>

# Secrets should go through Container Apps secrets, not plain env vars:
az containerapp secret set --name moderation-service \
  --resource-group moderation-rg --secrets llm-api-key=<key>
az containerapp update --name moderation-service \
  --resource-group moderation-rg \
  --set-env-vars LLM_API_KEY=secretref:llm-api-key
```

## 5. Verify

```bash
APP_URL=$(az containerapp show --name moderation-service \
  --resource-group moderation-rg \
  --query properties.configuration.ingress.fqdn -o tsv)

curl "https://$APP_URL/health"
curl -X POST "https://$APP_URL/moderate" \
  -H "Content-Type: application/json" \
  -d '{"text": "I really enjoyed the community picnic today."}'
```

## 6. Tear down

```bash
az group delete --name moderation-rg --yes --no-wait
```
