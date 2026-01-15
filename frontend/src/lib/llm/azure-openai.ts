import { createAzure } from '@ai-sdk/azure';
import type { LanguageModelV3 } from '@ai-sdk/provider';

export interface AzureOpenAIEnv {
  AZURE_OPENAI_RESOURCE_NAME?: string;
  AZURE_OPENAI_BASE_URL?: string;
  AZURE_OPENAI_ENDPOINT?: string;
  AZURE_OPENAI_API_KEY?: string;
  AZURE_OPENAI_API_VERSION?: string;
  AZURE_OPENAI_DEPLOYMENT?: string;
  AZURE_OPENAI_DEPLOYMENT_NAME?: string;
  AZURE_OPENAI_CHAT_DEPLOYMENT_NAME?: string;
}

export interface AzureOpenAIConfig {
  resourceName?: string;
  baseURL?: string;
  apiKey: string;
  apiVersion: string;
  deploymentId: string;
}

export function getAzureOpenAIConfig(
  env: AzureOpenAIEnv = process.env as unknown as AzureOpenAIEnv
): AzureOpenAIConfig {
  const apiKey = env.AZURE_OPENAI_API_KEY;
  if (!apiKey) {
    throw new Error(
      'Azure OpenAI is not configured: missing AZURE_OPENAI_API_KEY.'
    );
  }

  const apiVersion = env.AZURE_OPENAI_API_VERSION ?? '2024-06-01';
  if (apiVersion.trim().toLowerCase() === 'preview') {
    throw new Error(
      'Azure OpenAI is not configured: AZURE_OPENAI_API_VERSION="preview" is not a valid API version. Use a dated API version (e.g. "2024-06-01").'
    );
  }

  const endpoint = env.AZURE_OPENAI_ENDPOINT;
  const normalizedEndpoint = endpoint ? endpoint.replace(/\/+$/, '') : undefined;
  const normalizedEndpointBaseUrl = normalizedEndpoint
    ? normalizedEndpoint.endsWith('/openai')
      ? normalizedEndpoint
      : `${normalizedEndpoint}/openai`
    : undefined;

  const baseURL = env.AZURE_OPENAI_BASE_URL ?? normalizedEndpointBaseUrl;
  const resourceName = env.AZURE_OPENAI_RESOURCE_NAME;
  if (!baseURL && !resourceName) {
    throw new Error(
      'Azure OpenAI is not configured: set AZURE_OPENAI_BASE_URL, AZURE_OPENAI_ENDPOINT, or AZURE_OPENAI_RESOURCE_NAME.'
    );
  }

  const deploymentId =
    env.AZURE_OPENAI_DEPLOYMENT ??
    env.AZURE_OPENAI_DEPLOYMENT_NAME ??
    env.AZURE_OPENAI_CHAT_DEPLOYMENT_NAME;
  if (!deploymentId) {
    throw new Error(
      'Azure OpenAI is not configured: missing AZURE_OPENAI_DEPLOYMENT (or AZURE_OPENAI_DEPLOYMENT_NAME / AZURE_OPENAI_CHAT_DEPLOYMENT_NAME).'
    );
  }

  return {
    apiKey,
    baseURL,
    resourceName,
    apiVersion,
    deploymentId,
  };
}

export function getAzureChatModel(
  env: AzureOpenAIEnv = process.env as unknown as AzureOpenAIEnv
): LanguageModelV3 {
  const cfg = getAzureOpenAIConfig(env);

  const azure = createAzure({
    apiKey: cfg.apiKey,
    apiVersion: cfg.apiVersion,
    baseURL: cfg.baseURL,
    resourceName: cfg.resourceName,
  });

  return azure.chat(cfg.deploymentId);
}
