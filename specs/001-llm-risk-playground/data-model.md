# Data Model: LLM Risk Playground

**Feature**: 001-llm-risk-playground  
**Date**: 2026-01-09  
**Status**: Complete

## Entity Definitions

### 1. EducationalSettings

Represents the current configuration of educational mode toggles.

```typescript
// src/lib/types/settings.ts

export interface EducationalSettings {
  /** 
   * When true, uses local simulation instead of real Azure OpenAI.
   * Faster, deterministic, no API costs.
   * @default true
   */
  simulateMode: boolean;

  /**
   * When true, blocks known prompt injection patterns before they reach the LLM.
   * Disable to demonstrate vulnerable behavior.
   * @default true
   */
  guardEnabled: boolean;

  /**
   * When true, allows "sys admin" claims to bypass policy.
   * INTENTIONALLY UNSAFE - for demonstration only.
   * @default false
   */
  adminBypassDemo: boolean;
}

export const DEFAULT_SETTINGS: EducationalSettings = {
  simulateMode: true,
  guardEnabled: true,
  adminBypassDemo: false,
};
```

**Validation rules**:
- All fields are required booleans
- No interdependencies between fields

**State transitions**:
- Settings can be toggled at any time
- Changes take effect on next message submission
- Settings persist to localStorage on every change

---

### 2. RefundRequest

Represents a user's refund request input.

```typescript
// src/lib/types/agent.ts

export interface RefundRequest {
  /** Order identifier for the refund */
  orderId: string;

  /** Refund amount in dollars */
  amount: number;

  /** User's message/reason for refund (may contain injection attempts) */
  userMessage: string;
}
```

**Validation rules**:
- `orderId`: Non-empty string, alphanumeric with optional hyphens
- `amount`: Positive number, max 2 decimal places
- `userMessage`: Non-empty string, max 2000 characters

---

### 3. AgentResponse

Represents the agent's response to a refund request.

```typescript
// src/lib/types/agent.ts

export interface AgentResponse {
  /** Raw text response from the agent/simulation */
  text: string;

  /** Whether a refund was issued */
  refundIssued: boolean;

  /** Refund details if issued */
  refundResult?: RefundResult;

  /** Security analysis of the interaction */
  securityAnalysis: SecurityAnalysis;

  /** Which mode was used to generate response */
  mode: 'simulate' | 'live';

  /** Response generation timestamp */
  timestamp: Date;
}

export interface RefundResult {
  ok: boolean;
  orderId: string;
  amount: number;
  refundId: string;
  status: 'issued' | 'denied';
}
```

---

### 4. SecurityAnalysis

Represents OWASP-based security analysis of user input.

```typescript
// src/lib/types/security.ts

export interface SecurityAnalysis {
  /** Whether any injection patterns were detected */
  detected: boolean;

  /** List of detected patterns with details */
  patterns: DetectedPattern[];

  /** Overall severity assessment */
  severity: SecuritySeverity;

  /** OWASP LLM category (LLM01 for MVP) */
  owaspCategory: 'LLM01';

  /** Human-readable explanation of the risk */
  explanation: string;

  /** Recommended mitigations */
  mitigations: string[];
}

export interface DetectedPattern {
  /** Pattern category identifier */
  category: InjectionCategory;

  /** The matched text from user input */
  matchedText: string;

  /** Start index in original input */
  startIndex: number;

  /** End index in original input */
  endIndex: number;

  /** Severity of this specific pattern */
  severity: SecuritySeverity;
}

export type SecuritySeverity = 'none' | 'low' | 'medium' | 'high' | 'critical';

export type InjectionCategory = 
  | 'direct_injection'
  | 'role_impersonation'
  | 'context_manipulation'
  | 'jailbreak_attempt'
  | 'policy_bypass';
```

---

### 5. ChatMessage

Represents a single message in the chat interface.

```typescript
// src/lib/types/agent.ts

export interface ChatMessage {
  /** Unique message identifier */
  id: string;

  /** Message sender role */
  role: 'user' | 'assistant' | 'system';

  /** Message content */
  content: string;

  /** Creation timestamp */
  timestamp: Date;

  /** For user messages: the refund request data */
  request?: RefundRequest;

  /** For assistant messages: full agent response */
  response?: AgentResponse;
}
```

---

### 6. InjectionScenario

Represents a predefined injection example for educational purposes.

```typescript
// src/lib/types/agent.ts

export interface InjectionScenario {
  /** Unique scenario identifier */
  id: string;

  /** Display name */
  name: string;

  /** Category of injection technique */
  category: InjectionCategory;

  /** Brief description of the attack */
  description: string;

  /** The attack prompt to use */
  attackPrompt: string;

  /** Expected order ID for demo */
  orderId: string;

  /** Expected amount for demo */
  amount: number;

  /** Expected outcome when guards are disabled */
  expectedVulnerableOutcome: string;

  /** Expected outcome when guards are enabled */
  expectedSecureOutcome: string;

  /** Difficulty level for learning path */
  difficulty: 'beginner' | 'intermediate' | 'advanced';
}
```

---

### 7. PersistedState

Represents the data structure saved to localStorage.

```typescript
// src/lib/types/storage.ts

export interface PersistedState {
  /** Schema version for migration support */
  version: number;

  /** Current educational settings */
  settings: EducationalSettings;

  /** Recent chat messages (max 50) */
  messages: ChatMessage[];

  /** Last update timestamp */
  lastUpdated: Date;
}

export const CURRENT_SCHEMA_VERSION = 1;
export const MAX_PERSISTED_MESSAGES = 50;
export const STORAGE_KEY = 'llm-risk-playground-state';
```

---

## Entity Relationships

```text
┌─────────────────────┐
│ EducationalSettings │
└─────────┬───────────┘
          │ configures
          ▼
┌─────────────────────┐      creates      ┌─────────────────────┐
│   RefundRequest     │ ─────────────────▶│   AgentResponse     │
└─────────────────────┘                   └─────────┬───────────┘
                                                    │ contains
                                                    ▼
                                          ┌─────────────────────┐
                                          │  SecurityAnalysis   │
                                          └─────────────────────┘

┌─────────────────────┐      contains     ┌─────────────────────┐
│   PersistedState    │ ─────────────────▶│    ChatMessage[]    │
└─────────────────────┘                   └─────────────────────┘
          │ contains
          ▼
┌─────────────────────┐
│ EducationalSettings │
└─────────────────────┘

┌─────────────────────┐      provides     ┌─────────────────────┐
│ InjectionScenario[] │ ─────────────────▶│   RefundRequest     │
└─────────────────────┘                   └─────────────────────┘
```

## State Management

### Client State (React)

| State | Location | Persistence |
|-------|----------|-------------|
| `settings` | `useEducationalSettings` hook | localStorage |
| `messages` | `useAgentChat` hook | localStorage |
| `isLoading` | `useAgentChat` hook | None (transient) |
| `error` | `useAgentChat` hook | None (transient) |

### Server State (Server Actions)

| State | Location | Persistence |
|-------|----------|-------------|
| Azure OpenAI client | Server Action scope | None (per-request) |
| Policy text | Loaded from file | None (per-request) |

## Data Flow

```text
User Input
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│ Client: useAgentChat hook                                    │
│ 1. Validate RefundRequest                                    │
│ 2. Add user ChatMessage to state                            │
│ 3. Call Server Action                                        │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│ Server Action: submitRefundRequest                           │
│ 1. Check guards (if enabled)                                 │
│ 2. Run security analysis                                     │
│ 3. Generate response (simulate or live)                      │
│ 4. Return AgentResponse                                      │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│ Client: Update State                                         │
│ 1. Add assistant ChatMessage to state                        │
│ 2. Persist to localStorage                                   │
│ 3. Render SecurityAnalysis component                         │
└─────────────────────────────────────────────────────────────┘
```

## Validation Functions

```typescript
// src/lib/validation/request.ts

export function validateRefundRequest(input: unknown): RefundRequest {
  // Zod or manual validation
}

export function validateSettings(input: unknown): EducationalSettings {
  // Ensure all booleans present
}

export function sanitizeUserMessage(message: string): string {
  // Trim, limit length, but preserve injection attempts for education
  return message.trim().slice(0, 2000);
}
```
