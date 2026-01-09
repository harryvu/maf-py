# API Contracts: LLM Risk Playground

**Feature**: 001-llm-risk-playground  
**Date**: 2026-01-09  
**Status**: Complete

## Overview

This feature uses Next.js Server Actions instead of traditional REST APIs. The contracts below define the Server Action interfaces and their expected behavior.

## Server Actions

### 1. submitRefundRequest

Primary action for processing refund requests through the AI agent.

**Location**: `src/app/actions/agent.ts`

```typescript
'use server'

/**
 * Submit a refund request to the AI agent for processing.
 * 
 * @param request - The refund request details
 * @param settings - Current educational mode settings
 * @returns AgentResponse with text, refund result, and security analysis
 * @throws Error if Azure OpenAI fails (non-content-filter errors)
 */
export async function submitRefundRequest(
  request: RefundRequest,
  settings: EducationalSettings
): Promise<AgentResponse>
```

**Input Schema**:
```typescript
interface RefundRequest {
  orderId: string;      // Required, non-empty
  amount: number;       // Required, positive number
  userMessage: string;  // Required, max 2000 chars
}

interface EducationalSettings {
  simulateMode: boolean;    // Default: true
  guardEnabled: boolean;    // Default: true
  adminBypassDemo: boolean; // Default: false
}
```

**Output Schema**:
```typescript
interface AgentResponse {
  text: string;
  refundIssued: boolean;
  refundResult?: RefundResult;
  securityAnalysis: SecurityAnalysis;
  mode: 'simulate' | 'live';
  timestamp: Date;
}
```

**Behavior Matrix**:

| Condition | Guards ON | Guards OFF |
|-----------|-----------|------------|
| Normal request | Process normally | Process normally |
| Injection pattern detected | Block + explain | Allow through (vulnerable) |
| Admin bypass claim + flag ON | Block | Allow (demonstrate vulnerability) |
| Admin bypass claim + flag OFF | Block | Block |

**Error Handling**:
```typescript
// Content filter error (Azure blocks the request)
{
  text: "Your request was blocked by the Azure OpenAI content filter...",
  refundIssued: false,
  securityAnalysis: { detected: true, severity: 'critical', ... },
  mode: 'live',
  timestamp: Date
}

// Network/timeout error
{
  text: "The agent service failed to complete the request...",
  refundIssued: false,
  securityAnalysis: { detected: false, severity: 'none', ... },
  mode: 'live',
  timestamp: Date
}
```

---

### 2. analyzeSecurityPatterns

Standalone security analysis without agent execution.

**Location**: `src/app/actions/security.ts`

```typescript
'use server'

/**
 * Analyze user input for security patterns without executing agent.
 * Used for real-time feedback as user types.
 * 
 * @param input - User message to analyze
 * @returns SecurityAnalysis with detected patterns
 */
export async function analyzeSecurityPatterns(
  input: string
): Promise<SecurityAnalysis>
```

**Input Schema**:
```typescript
input: string  // Max 2000 characters
```

**Output Schema**:
```typescript
interface SecurityAnalysis {
  detected: boolean;
  patterns: DetectedPattern[];
  severity: 'none' | 'low' | 'medium' | 'high' | 'critical';
  owaspCategory: 'LLM01';
  explanation: string;
  mitigations: string[];
}
```

---

### 3. getInjectionExamples

Retrieve predefined injection scenarios for educational use.

**Location**: `src/app/actions/examples.ts`

```typescript
'use server'

/**
 * Get predefined injection examples for the playground.
 * 
 * @param category - Optional filter by injection category
 * @param difficulty - Optional filter by difficulty level
 * @returns Array of InjectionScenario objects
 */
export async function getInjectionExamples(
  category?: InjectionCategory,
  difficulty?: 'beginner' | 'intermediate' | 'advanced'
): Promise<InjectionScenario[]>
```

**Output Schema**:
```typescript
interface InjectionScenario {
  id: string;
  name: string;
  category: InjectionCategory;
  description: string;
  attackPrompt: string;
  orderId: string;
  amount: number;
  expectedVulnerableOutcome: string;
  expectedSecureOutcome: string;
  difficulty: 'beginner' | 'intermediate' | 'advanced';
}
```

---

## Client-Side Contracts

### Custom Hooks

#### useEducationalSettings

```typescript
// src/hooks/useEducationalSettings.ts

interface UseEducationalSettingsReturn {
  settings: EducationalSettings;
  setSimulateMode: (enabled: boolean) => void;
  setGuardEnabled: (enabled: boolean) => void;
  setAdminBypassDemo: (enabled: boolean) => void;
  resetToDefaults: () => void;
  isLoaded: boolean;
}

function useEducationalSettings(): UseEducationalSettingsReturn
```

#### useAgentChat

```typescript
// src/hooks/useAgentChat.ts

interface UseAgentChatReturn {
  messages: ChatMessage[];
  isLoading: boolean;
  error: Error | null;
  sendMessage: (request: RefundRequest) => Promise<void>;
  clearMessages: () => void;
  retryLast: () => Promise<void>;
}

function useAgentChat(settings: EducationalSettings): UseAgentChatReturn
```

#### useSecurityAnalysis

```typescript
// src/hooks/useSecurityAnalysis.ts

interface UseSecurityAnalysisReturn {
  analysis: SecurityAnalysis | null;
  analyze: (input: string) => Promise<void>;
  isAnalyzing: boolean;
}

function useSecurityAnalysis(): UseSecurityAnalysisReturn
```

---

## Component Props Contracts

### ChatInterface

```typescript
interface ChatInterfaceProps {
  settings: EducationalSettings;
  onSettingsChange?: () => void;
}
```

### ChatMessage

```typescript
interface ChatMessageProps {
  message: ChatMessage;
  showSecurityAnalysis?: boolean;
}
```

### SettingsPanel

```typescript
interface SettingsPanelProps {
  settings: EducationalSettings;
  onSettingChange: (key: keyof EducationalSettings, value: boolean) => void;
  isOpen: boolean;
  onClose: () => void;
}
```

### SecurityAnalysis

```typescript
interface SecurityAnalysisProps {
  analysis: SecurityAnalysis;
  showMitigations?: boolean;
  compact?: boolean;
}
```

### VulnerabilityBadge

```typescript
interface VulnerabilityBadgeProps {
  severity: SecuritySeverity;
  category: InjectionCategory;
  onClick?: () => void;
}
```

---

## Test Contracts

### Mock Factories

```typescript
// tests/factories/agent.ts

export function createMockRefundRequest(
  overrides?: Partial<RefundRequest>
): RefundRequest

export function createMockAgentResponse(
  overrides?: Partial<AgentResponse>
): AgentResponse

export function createMockSecurityAnalysis(
  overrides?: Partial<SecurityAnalysis>
): SecurityAnalysis

export function createMockSettings(
  overrides?: Partial<EducationalSettings>
): EducationalSettings
```

### MSW Handlers

```typescript
// tests/mocks/handlers.ts

// Mock Azure OpenAI responses for integration tests
export const handlers = [
  // Successful response
  http.post('https://*.openai.azure.com/*', () => {
    return HttpResponse.json({ /* mock response */ })
  }),
  
  // Content filter error
  http.post('https://*.openai.azure.com/*', () => {
    return new HttpResponse(null, { status: 400, ... })
  }),
]
```

---

## Environment Contract

```typescript
// Required environment variables
interface EnvironmentVariables {
  AZURE_OPENAI_API_KEY: string;      // Required for live mode
  AZURE_OPENAI_ENDPOINT: string;     // Required for live mode
  AZURE_OPENAI_DEPLOYMENT_NAME: string; // Required for live mode
}

// Validation at startup
function validateEnvironment(): void {
  const required = [
    'AZURE_OPENAI_API_KEY',
    'AZURE_OPENAI_ENDPOINT', 
    'AZURE_OPENAI_DEPLOYMENT_NAME'
  ];
  
  const missing = required.filter(key => !process.env[key]);
  
  if (missing.length > 0) {
    console.warn(`Missing env vars: ${missing.join(', ')}. Live mode disabled.`);
  }
}
```
