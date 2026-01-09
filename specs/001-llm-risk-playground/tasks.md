# Tasks: LLM Risk Playground

**Input**: Design documents from `/specs/001-llm-risk-playground/`
**Prerequisites**: plan.md ✅, spec.md ✅, research.md ✅, data-model.md ✅, contracts/ ✅

**Tests**: TDD is MANDATORY per constitution - 90% coverage required. All tests written FIRST, verified to FAIL, then implementation.

**Organization**: Tasks grouped by user story to enable independent implementation and testing.

## Format: `[ID] [P?] [Story?] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[US1/US2/US3]**: Which user story this task belongs to
- Include exact file paths in descriptions

## Path Conventions

- **Web app**: `frontend/src/`, `frontend/tests/`
- All paths relative to repository root

---

## Phase 1: Setup

**Purpose**: Project initialization and testing infrastructure

- [ ] T001 Create Next.js project with TypeScript and Tailwind in frontend/
- [ ] T002 Install core dependencies (ai, @ai-sdk/azure) in frontend/package.json
- [ ] T003 [P] Install dev dependencies (vitest, playwright, msw, testing-library) in frontend/package.json
- [ ] T004 [P] Create vitest.config.ts with 90% coverage threshold in frontend/vitest.config.ts
- [ ] T005 [P] Create playwright.config.ts for E2E testing in frontend/playwright.config.ts
- [ ] T006 [P] Create tests/setup.ts with testing-library setup in frontend/tests/setup.ts
- [ ] T007 [P] Configure environment variables in frontend/.env.local
- [ ] T008 Add test scripts to frontend/package.json (test, test:coverage, test:e2e)
- [ ] T009 Verify setup: run `npm test` and confirm vitest runs in frontend/

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core types and utilities that ALL user stories depend on

**⚠️ CRITICAL**: No user story work can begin until this phase is complete

### Foundational Tests (Write FIRST - must FAIL)

- [ ] T010 [P] Unit test for EducationalSettings type validation in frontend/src/lib/types/__tests__/settings.test.ts
- [ ] T011 [P] Unit test for RefundRequest type validation in frontend/src/lib/types/__tests__/agent.test.ts
- [ ] T012 [P] Unit test for SecurityAnalysis type in frontend/src/lib/types/__tests__/security.test.ts
- [ ] T013 [P] Unit test for injection patterns (20+ cases) in frontend/src/lib/agent/__tests__/injection-patterns.test.ts
- [ ] T014 [P] Unit test for policy retrieval in frontend/src/lib/agent/__tests__/policy.test.ts
- [ ] T015 [P] Unit test for OWASP analyzer in frontend/src/lib/security/__tests__/owasp-analyzer.test.ts
- [ ] T016 [P] Unit test for pattern detector (25+ cases) in frontend/src/lib/security/__tests__/pattern-detector.test.ts

### Foundational Implementation (Make tests GREEN)

- [ ] T017 [P] Create EducationalSettings interface and DEFAULT_SETTINGS in frontend/src/lib/types/settings.ts
- [ ] T018 [P] Create RefundRequest, AgentResponse, RefundResult interfaces in frontend/src/lib/types/agent.ts
- [ ] T019 [P] Create SecurityAnalysis, DetectedPattern, InjectionCategory types in frontend/src/lib/types/security.ts
- [ ] T020 [P] Create ChatMessage, InjectionScenario interfaces in frontend/src/lib/types/agent.ts
- [ ] T021 Implement INJECTION_PATTERNS array (port from Python) in frontend/src/lib/agent/injection-patterns.ts
- [ ] T022 Implement looksLikePromptInjection() function in frontend/src/lib/agent/injection-patterns.ts
- [ ] T023 Implement retrievePolicy() function in frontend/src/lib/agent/policy.ts
- [ ] T024 Implement REFUND_POLICY constant in frontend/src/lib/agent/policy.ts
- [ ] T025 Implement analyzeForOwasp() function in frontend/src/lib/security/owasp-analyzer.ts
- [ ] T026 Implement PatternDetector class in frontend/src/lib/security/pattern-detector.ts
- [ ] T027 Implement createSecurityAnalysis() helper in frontend/src/lib/security/owasp-analyzer.ts

### Foundational Verification

- [ ] T028 Verify all foundational tests pass: run `npm test` in frontend/
- [ ] T029 Verify coverage ≥90% for lib/ modules: run `npm run test:coverage` in frontend/

**Checkpoint**: Foundation ready - user story implementation can now begin

---

## Phase 3: User Story 1 - Basic Prompt Injection Demonstration (Priority: P1) 🎯 MVP

**Goal**: Users can access web interface, toggle educational settings, enter injection prompts, and see agent response with security analysis

**Independent Test**: Load interface → Configure settings → Enter injection → See response + analysis

### Tests for User Story 1 (Write FIRST - must FAIL)

#### Unit Tests

- [ ] T030 [P] [US1] Unit test for simulateLlmReply() (15+ cases) in frontend/src/lib/agent/__tests__/refund-agent.test.ts
- [ ] T031 [P] [US1] Unit test for issueRefund() in frontend/src/lib/agent/__tests__/refund-agent.test.ts
- [ ] T032 [P] [US1] Unit test for useEducationalSettings hook in frontend/src/hooks/__tests__/useEducationalSettings.test.ts
- [ ] T033 [P] [US1] Unit test for useAgentChat hook in frontend/src/hooks/__tests__/useAgentChat.test.ts
- [ ] T034 [P] [US1] Unit test for useSecurityAnalysis hook in frontend/src/hooks/__tests__/useSecurityAnalysis.test.ts

#### Component Tests

- [ ] T035 [P] [US1] Component test for SettingsPanel in frontend/src/components/settings/__tests__/SettingsPanel.test.tsx
- [ ] T036 [P] [US1] Component test for SettingsToggle in frontend/src/components/settings/__tests__/SettingsToggle.test.tsx
- [ ] T037 [P] [US1] Component test for ChatInterface in frontend/src/components/chat/__tests__/ChatInterface.test.tsx
- [ ] T038 [P] [US1] Component test for ChatMessage in frontend/src/components/chat/__tests__/ChatMessage.test.tsx
- [ ] T039 [P] [US1] Component test for ChatInput in frontend/src/components/chat/__tests__/ChatInput.test.tsx
- [ ] T040 [P] [US1] Component test for SecurityAnalysis display in frontend/src/components/security/__tests__/SecurityAnalysis.test.tsx
- [ ] T041 [P] [US1] Component test for VulnerabilityBadge in frontend/src/components/security/__tests__/VulnerabilityBadge.test.tsx

#### Integration Tests

- [ ] T042 [P] [US1] Integration test for submitRefundRequest Server Action in frontend/tests/integration/agent-action.test.ts
- [ ] T043 [P] [US1] Integration test for analyzeSecurityPatterns Server Action in frontend/tests/integration/security-action.test.ts
- [ ] T044 [P] [US1] Integration test for localStorage session persistence in frontend/tests/integration/session-persistence.test.ts

#### E2E Tests

- [ ] T045 [US1] E2E test: settings panel displays all toggles (US1-AC1) in frontend/tests/e2e/basic-demo.spec.ts
- [ ] T046 [US1] E2E test: injection with guards disabled shows vulnerability (US1-AC2) in frontend/tests/e2e/basic-demo.spec.ts
- [ ] T047 [US1] E2E test: educational explanation displayed after demo (US1-AC3) in frontend/tests/e2e/basic-demo.spec.ts

### Implementation for User Story 1 (Make tests GREEN)

#### Agent Logic

- [ ] T048 [US1] Implement simulateLlmReply() function (port from Python) in frontend/src/lib/agent/refund-agent.ts
- [ ] T049 [US1] Implement issueRefund() function in frontend/src/lib/agent/refund-agent.ts
- [ ] T050 [US1] Implement RefundTools class with state tracking in frontend/src/lib/agent/refund-agent.ts

#### Custom Hooks

- [ ] T051 [US1] Implement useEducationalSettings hook with localStorage in frontend/src/hooks/useEducationalSettings.ts
- [ ] T052 [US1] Implement useAgentChat hook in frontend/src/hooks/useAgentChat.ts
- [ ] T053 [US1] Implement useSecurityAnalysis hook in frontend/src/hooks/useSecurityAnalysis.ts

#### Server Actions

- [ ] T054 [US1] Implement submitRefundRequest Server Action in frontend/src/app/actions/agent.ts
- [ ] T055 [US1] Implement analyzeSecurityPatterns Server Action in frontend/src/app/actions/security.ts
- [ ] T056 [US1] Add AI SDK Azure OpenAI integration to agent.ts in frontend/src/app/actions/agent.ts
- [ ] T057 [US1] Add error handling for content filter and network errors in frontend/src/app/actions/agent.ts

#### UI Components - Settings

- [ ] T058 [P] [US1] Create SettingsToggle component in frontend/src/components/settings/SettingsToggle.tsx
- [ ] T059 [US1] Create SettingsPanel component with all toggles in frontend/src/components/settings/SettingsPanel.tsx
- [ ] T060 [US1] Add toggle explanations to SettingsPanel in frontend/src/components/settings/SettingsPanel.tsx

#### UI Components - Chat

- [ ] T061 [P] [US1] Create ChatInput component in frontend/src/components/chat/ChatInput.tsx
- [ ] T062 [P] [US1] Create ChatMessage component in frontend/src/components/chat/ChatMessage.tsx
- [ ] T063 [US1] Create ChatInterface container component in frontend/src/components/chat/ChatInterface.tsx
- [ ] T064 [US1] Add loading and error states to ChatInterface in frontend/src/components/chat/ChatInterface.tsx

#### UI Components - Security

- [ ] T065 [P] [US1] Create VulnerabilityBadge component in frontend/src/components/security/VulnerabilityBadge.tsx
- [ ] T066 [US1] Create SecurityAnalysis display component in frontend/src/components/security/SecurityAnalysis.tsx
- [ ] T067 [US1] Add inline vulnerability explanation to SecurityAnalysis in frontend/src/components/security/SecurityAnalysis.tsx

#### Page Integration

- [ ] T068 [US1] Update main page layout in frontend/src/app/layout.tsx
- [ ] T069 [US1] Create main playground page with chat and settings in frontend/src/app/page.tsx
- [ ] T070 [US1] Add Tailwind styling for responsive layout in frontend/src/app/globals.css

### User Story 1 Verification

- [ ] T071 [US1] Verify all US1 tests pass: run `npm test` in frontend/
- [ ] T072 [US1] Verify E2E tests pass: run `npm run test:e2e` in frontend/
- [ ] T073 [US1] Verify coverage ≥90%: run `npm run test:coverage` in frontend/
- [ ] T074 [US1] Manual verification: complete all US1 acceptance scenarios

**Checkpoint**: User Story 1 (MVP) should be fully functional and independently testable

---

## Phase 4: User Story 2 - Interactive Injection Playground (Priority: P2)

**Goal**: Users can create custom injection scenarios with real-time pattern detection and compare configurations

**Independent Test**: Enter custom prompts → See real-time detection → Compare different settings

### Tests for User Story 2 (Write FIRST - must FAIL)

- [ ] T075 [P] [US2] Unit test for injection examples data in frontend/src/data/__tests__/injection-examples.test.ts
- [ ] T076 [P] [US2] Unit test for getInjectionExamples filtering in frontend/src/app/actions/__tests__/examples.test.ts
- [ ] T077 [P] [US2] Component test for ExampleSelector in frontend/src/components/chat/__tests__/ExampleSelector.test.tsx
- [ ] T078 [P] [US2] Component test for real-time pattern highlighting in frontend/src/components/chat/__tests__/ChatInput.test.tsx
- [ ] T079 [US2] E2E test: real-time pattern detection (US2-AC1) in frontend/tests/e2e/playground.spec.ts
- [ ] T080 [US2] E2E test: compare different configurations (US2-AC2) in frontend/tests/e2e/playground.spec.ts

### Implementation for User Story 2 (Make tests GREEN)

- [ ] T081 [P] [US2] Create injection examples data with categories in frontend/src/data/injection-examples.ts
- [ ] T082 [US2] Implement getInjectionExamples Server Action in frontend/src/app/actions/examples.ts
- [ ] T083 [P] [US2] Create ExampleSelector component in frontend/src/components/chat/ExampleSelector.tsx
- [ ] T084 [US2] Add real-time pattern highlighting to ChatInput in frontend/src/components/chat/ChatInput.tsx
- [ ] T085 [US2] Add debounced security analysis as user types in frontend/src/components/chat/ChatInput.tsx
- [ ] T086 [US2] Add preset scenario buttons to ChatInterface in frontend/src/components/chat/ChatInterface.tsx
- [ ] T087 [US2] Create comparison view for different settings in frontend/src/components/chat/ComparisonView.tsx

### User Story 2 Verification

- [ ] T088 [US2] Verify all US2 tests pass: run `npm test` in frontend/
- [ ] T089 [US2] Verify E2E tests pass: run `npm run test:e2e` in frontend/
- [ ] T090 [US2] Manual verification: complete all US2 acceptance scenarios

**Checkpoint**: User Stories 1 AND 2 should both work independently

---

## Phase 5: User Story 3 - Prompt Injection Documentation (Priority: P3)

**Goal**: Users can access comprehensive LLM01 documentation with examples and mitigations

**Independent Test**: Navigate to docs → View LLM01 content → Follow learning path

### Tests for User Story 3 (Write FIRST - must FAIL)

- [ ] T091 [P] [US3] Unit test for OWASP LLM01 content data in frontend/src/data/__tests__/owasp-llm01-content.test.ts
- [ ] T092 [P] [US3] Component test for LLM01Documentation in frontend/src/components/documentation/__tests__/LLM01Documentation.test.tsx
- [ ] T093 [P] [US3] Component test for MitigationTips in frontend/src/components/documentation/__tests__/MitigationTips.test.tsx
- [ ] T094 [US3] E2E test: documentation displays (US3-AC1) in frontend/tests/e2e/documentation.spec.ts
- [ ] T095 [US3] E2E test: learning path progression (US3-AC2) in frontend/tests/e2e/documentation.spec.ts

### Implementation for User Story 3 (Make tests GREEN)

- [ ] T096 [P] [US3] Create OWASP LLM01 content data in frontend/src/data/owasp-llm01-content.ts
- [ ] T097 [P] [US3] Create MitigationTips component in frontend/src/components/documentation/MitigationTips.tsx
- [ ] T098 [US3] Create LLM01Documentation component in frontend/src/components/documentation/LLM01Documentation.tsx
- [ ] T099 [US3] Add documentation panel/tab to main page in frontend/src/app/page.tsx
- [ ] T100 [US3] Add progressive learning path with difficulty levels in frontend/src/components/documentation/LearningPath.tsx

### User Story 3 Verification

- [ ] T101 [US3] Verify all US3 tests pass: run `npm test` in frontend/
- [ ] T102 [US3] Verify E2E tests pass: run `npm run test:e2e` in frontend/
- [ ] T103 [US3] Manual verification: complete all US3 acceptance scenarios

**Checkpoint**: All user stories should now be independently functional

---

## Phase 6: Polish & Cross-Cutting Concerns

**Purpose**: Final quality improvements and validation

- [ ] T104 [P] Add comprehensive JSDoc comments to all public functions in frontend/src/lib/
- [ ] T105 [P] Add accessibility attributes (aria-*) to all interactive components in frontend/src/components/
- [ ] T106 Code cleanup and consistent formatting with Prettier in frontend/
- [ ] T107 [P] Add loading skeletons for async operations in frontend/src/components/
- [ ] T108 Performance optimization: memoize expensive computations in frontend/src/hooks/
- [ ] T109 [P] Add responsive design breakpoints in frontend/src/app/globals.css
- [ ] T110 Final coverage verification: ensure ≥90% overall in frontend/
- [ ] T111 Run quickstart.md validation: verify all setup steps work
- [ ] T112 Update README.md with usage instructions in frontend/README.md

---

## Dependencies & Execution Order

### Phase Dependencies

```text
Phase 1: Setup ──────────────────────────────────┐
                                                 │
Phase 2: Foundational ◄──────────────────────────┘
    │
    │ BLOCKS ALL USER STORIES
    ▼
┌─────────────────────────────────────────────────────┐
│ Phase 3: US1 (P1) ◄─── MVP COMPLETE AFTER THIS     │
│ Phase 4: US2 (P2) ◄─── Can start after Phase 2     │
│ Phase 5: US3 (P3) ◄─── Can start after Phase 2     │
│                                                     │
│ All user stories can run in PARALLEL after Phase 2 │
└─────────────────────────────────────────────────────┘
    │
    ▼
Phase 6: Polish ◄─── After desired stories complete
```

### User Story Independence

| Story | Dependencies | Can Start After |
|-------|--------------|-----------------|
| US1 (P1) | Phase 2 only | Phase 2 complete |
| US2 (P2) | Phase 2 only | Phase 2 complete |
| US3 (P3) | Phase 2 only | Phase 2 complete |

### Within Each User Story (TDD Order)

1. **Tests FIRST**: All tests written and FAIL
2. **Implementation**: Make tests GREEN one by one
3. **Refactor**: Clean up while tests stay green
4. **Verification**: Coverage check before moving on

---

## Parallel Opportunities

### Phase 1: Setup (All can run in parallel)
```bash
T003, T004, T005, T006, T007 # All [P] tasks
```

### Phase 2: Foundational Tests (All can run in parallel)
```bash
T010, T011, T012, T013, T014, T015, T016 # All [P] tests
```

### Phase 2: Foundational Implementation (Parallel groups)
```bash
# Group 1: Types (all parallel)
T017, T018, T019, T020

# Group 2: Agent logic (sequential after types)
T021 → T022 → T023 → T024

# Group 3: Security logic (parallel with Group 2)
T025, T026, T027
```

### Phase 3: User Story 1 Tests (All can run in parallel)
```bash
# Unit tests
T030, T031, T032, T033, T034

# Component tests
T035, T036, T037, T038, T039, T040, T041

# Integration tests
T042, T043, T044
```

### User Stories in Parallel (After Phase 2)
```bash
# With multiple developers:
Developer A: Phase 3 (US1) T030-T074
Developer B: Phase 4 (US2) T075-T090
Developer C: Phase 5 (US3) T091-T103
```

---

## Implementation Strategy

### MVP First (Recommended)

1. Complete Phase 1: Setup (T001-T009)
2. Complete Phase 2: Foundational (T010-T029)
3. Complete Phase 3: User Story 1 (T030-T074)
4. **STOP and VALIDATE**: Test US1 independently
5. Deploy/demo if MVP is sufficient

### Incremental Delivery

| Increment | Tasks | Value Delivered |
|-----------|-------|-----------------|
| MVP | T001-T074 | Basic prompt injection demo |
| +Playground | T075-T090 | Custom scenarios, real-time detection |
| +Documentation | T091-T103 | Educational content, learning paths |
| +Polish | T104-T112 | Production-ready quality |

---

## Task Summary

| Phase | Task Range | Task Count | Parallel Opportunities |
|-------|------------|------------|------------------------|
| Setup | T001-T009 | 9 | 5 parallel |
| Foundational | T010-T029 | 20 | 15 parallel (tests + types) |
| US1 (P1) MVP | T030-T074 | 45 | 20+ parallel (tests) |
| US2 (P2) | T075-T090 | 16 | 5 parallel |
| US3 (P3) | T091-T103 | 13 | 4 parallel |
| Polish | T104-T112 | 9 | 5 parallel |
| **Total** | T001-T112 | **112** | **~50 parallel** |

---

## Notes

- All [P] tasks can run in parallel (different files, no dependencies)
- TDD is MANDATORY: Write tests → Verify FAIL → Implement → Verify PASS
- Coverage gate: 90% required before any task is considered complete
- Each user story is independently testable after completion
- Stop at any checkpoint to validate and potentially deploy
