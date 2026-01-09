# Research: LLM Risk Playground

**Feature**: 001-llm-risk-playground  
**Date**: 2026-01-09  
**Status**: Complete

## Research Tasks Completed

### 1. AI SDK Azure OpenAI Integration

**Decision**: Use `@ai-sdk/azure` package for Azure OpenAI integration

**Rationale**:
- Official Vercel AI SDK support for Azure OpenAI
- Streaming support out of the box
- Type-safe API with excellent TypeScript support
- Server Actions compatible

**Alternatives considered**:
- Direct Azure OpenAI SDK: More verbose, less Next.js integration
- OpenAI SDK with Azure endpoint: Configuration complexity
- LangChain.js: Overkill for this use case, adds unnecessary abstraction

**Integration pattern**:
```typescript
import { createAzure } from '@ai-sdk/azure';
import { generateText } from 'ai';

const azure = createAzure({
  apiKey: process.env.AZURE_OPENAI_API_KEY,
  resourceName: process.env.AZURE_OPENAI_RESOURCE_NAME,
});

const result = await generateText({
  model: azure('deployment-name'),
  prompt: userMessage,
  system: systemPrompt,
});
```

### 2. TypeScript Port of refund_agent.py Vulnerabilities

**Decision**: Direct port maintaining identical vulnerability behavior

**Rationale**:
- Educational consistency between Python CLI and web demo
- Identical injection patterns ensure accurate demonstrations
- Same policy logic ensures comparable outcomes

**Key behaviors to preserve**:

| Python Behavior | TypeScript Implementation |
|-----------------|---------------------------|
| `_INJECTION_PATTERNS` regex | `INJECTION_PATTERNS` array of RegExp |
| `_looks_like_prompt_injection()` | `looksLikePromptInjection()` |
| `_simulate_llm_reply()` | `simulateLlmReply()` |
| `--simulate` flag | `settings.simulateMode` |
| `--no-guard` flag | `settings.guardEnabled` (inverted) |
| `--demo-admin-bypass` flag | `settings.adminBypassDemo` |

**Regex patterns to port**:
```typescript
export const INJECTION_PATTERNS: RegExp[] = [
  /\bignore\s+all\s+instructions\b/i,
  /\b(i\s*am\s*the\s*system|you\s*are\s*the\s*system)\b/i,
  /\b(system\s+prompt|developer\s+message)\b/i,
  /\bjailbreak\b/i,
];
```

### 3. OWASP LLM01 Security Analysis Patterns

**Decision**: Pre-built pattern library based on OWASP LLM Top 10 guidelines

**Rationale**:
- Educational focus requires deterministic, explainable analysis
- No external API dependencies for security scoring
- Immediate feedback without latency

**Pattern categories for LLM01**:

| Category | Patterns | Severity |
|----------|----------|----------|
| **Direct Injection** | "ignore", "disregard", "forget" + "instructions/rules" | Critical |
| **Role Impersonation** | "I am the system", "developer mode", "admin override" | Critical |
| **Context Manipulation** | "system prompt", "new instructions", "actually you are" | High |
| **Jailbreak Attempts** | "DAN", "jailbreak", "do anything now" | Critical |
| **Policy Bypass** | "ignore policy", "override limit", "supervisor approval" | High |

**Analysis output structure**:
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

### 4. Next.js Server Actions Best Practices

**Decision**: Use Server Actions for all AI SDK calls

**Rationale**:
- Secure: API keys never exposed to client
- Simple: No separate API route management
- Type-safe: End-to-end TypeScript
- Streaming: Native support for AI responses

**Implementation pattern**:
```typescript
// src/app/actions/agent.ts
'use server'

import { createAzure } from '@ai-sdk/azure';
import { generateText } from 'ai';

export async function submitRefundRequest(
  message: string,
  settings: EducationalSettings
): Promise<AgentResponse> {
  // Server-side only execution
  const azure = createAzure({...});
  // ...
}
```

### 5. Session Persistence Strategy

**Decision**: Browser localStorage for MVP

**Rationale**:
- No server-side state required
- Instant persistence without latency
- Works offline after initial load
- Sufficient for single-user educational tool

**Data to persist**:
- Educational settings (simulate mode, guards, admin bypass)
- Conversation history (last 50 messages)
- User preferences (theme, panel state)

**Implementation**:
```typescript
const STORAGE_KEY = 'llm-risk-playground';

interface PersistedState {
  settings: EducationalSettings;
  messages: ChatMessage[];
  version: number;
}
```

### 6. Testing Framework Selection

**Decision**: Vitest + React Testing Library + Playwright

**Rationale**:
- Vitest: Fast, Vite-native, excellent DX
- RTL: Component testing best practices
- Playwright: Cross-browser E2E, reliable

**Configuration approach**:
- Vitest for unit + component tests
- MSW (Mock Service Worker) for AI SDK mocking
- Playwright for full user journey tests

**Coverage tooling**:
```json
{
  "scripts": {
    "test": "vitest",
    "test:coverage": "vitest --coverage",
    "test:e2e": "playwright test"
  }
}
```

### 7. UI/UX Patterns for Educational Settings

**Decision**: Side panel with toggle switches and inline explanations

**Rationale**:
- Non-intrusive but always accessible
- Clear visual feedback for active modes
- Educational explanations visible without extra clicks

**Component structure**:
```
┌─────────────────────────────────────────┐
│ ⚙️ Educational Settings                 │
├─────────────────────────────────────────┤
│ ┌─────────────────────────────────────┐ │
│ │ 🧪 Simulate Mode            [ON]    │ │
│ │ Uses local simulation instead of    │ │
│ │ real LLM. Faster, deterministic.   │ │
│ └─────────────────────────────────────┘ │
│ ┌─────────────────────────────────────┐ │
│ │ 🛡️ Guard Enabled            [ON]    │ │
│ │ Blocks known injection patterns.   │ │
│ │ Disable to see vulnerable behavior.│ │
│ └─────────────────────────────────────┘ │
│ ┌─────────────────────────────────────┐ │
│ │ 🔓 Admin Bypass Demo       [OFF]    │ │
│ │ Allows "sys admin" claims to       │ │
│ │ bypass policy. INTENTIONALLY UNSAFE│ │
│ └─────────────────────────────────────┘ │
└─────────────────────────────────────────┘
```

## Resolved Unknowns

| Unknown | Resolution |
|---------|------------|
| AI SDK Azure compatibility | ✅ Full support via @ai-sdk/azure |
| TypeScript regex parity | ✅ Direct port possible with /i flag |
| Server Actions streaming | ✅ Supported with useChat hook |
| Testing 90% coverage | ✅ Achievable with component isolation |
| Azure content filter handling | ✅ Catch BadRequestError, fallback to simulate |

## Technology Stack Summary

| Layer | Technology | Version |
|-------|------------|---------|
| Framework | Next.js | 14.x |
| Runtime | Node.js | 20.x LTS |
| Language | TypeScript | 5.x |
| AI Integration | AI SDK + @ai-sdk/azure | 3.x |
| Styling | Tailwind CSS | 3.4.x |
| Unit Testing | Vitest | 1.x |
| Component Testing | React Testing Library | 14.x |
| E2E Testing | Playwright | 1.40.x |
| Mocking | MSW | 2.x |

## Next Steps

Phase 1 outputs ready to generate:
- [x] research.md (this file)
- [ ] data-model.md
- [ ] contracts/
- [ ] quickstart.md
