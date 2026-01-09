# MAF-PY Development Guidelines

Auto-generated from all feature plans. Last updated: 2026-01-09

## Active Technologies

### Python (Existing - CLI)
- Python 3.12+
- agent-framework >= 1.0.0b251120
- python-dotenv
- pytest (testing)

### TypeScript/Next.js (New - Web Frontend)
- TypeScript 5.x
- Node.js 20.x LTS
- Next.js 14+ with App Router
- AI SDK (@ai-sdk/azure) for Azure OpenAI
- React 18+
- Tailwind CSS 3.4+
- Vitest (unit testing)
- React Testing Library (component testing)
- Playwright (E2E testing)
- MSW (API mocking)

## Project Structure

```text
maf-py/
├── # Python CLI agents (existing, untouched)
├── refund_agent.py
├── simplest_agent.py
├── vision_agent.py
├── gemini-agent.py
├── muti-turn_convs.py
├── pyproject.toml
│
├── # Next.js frontend (new)
├── frontend/
│   ├── src/
│   │   ├── app/
│   │   │   ├── actions/        # Server Actions
│   │   │   └── page.tsx
│   │   ├── components/
│   │   │   ├── chat/
│   │   │   ├── settings/
│   │   │   ├── security/
│   │   │   └── documentation/
│   │   ├── lib/
│   │   │   ├── agent/          # TypeScript port of vulnerabilities
│   │   │   ├── security/       # OWASP analyzers
│   │   │   └── types/
│   │   ├── hooks/
│   │   └── data/
│   └── tests/
│       ├── unit/
│       ├── integration/
│       └── e2e/
│
├── # Specifications
├── specs/
│   └── 001-llm-risk-playground/
│       ├── spec.md
│       ├── plan.md
│       ├── research.md
│       ├── data-model.md
│       ├── quickstart.md
│       └── contracts/
│
└── .specify/
    ├── memory/
    │   └── constitution.md
    └── templates/
```

## Commands

### Python (existing)
```bash
# Run refund agent
python refund_agent.py [--simulate] [--no-guard] [--demo-admin-bypass] <order_id> <amount> "<message>"

# Run tests
pytest
```

### TypeScript/Next.js (new)
```bash
cd frontend

# Development
npm run dev

# Testing (TDD workflow)
npm test                 # Watch mode
npm run test:run         # Single run
npm run test:coverage    # With coverage report
npm run test:e2e         # Playwright E2E

# Build
npm run build
```

## Code Style

### Python
- PEP 8 compliant
- Type hints required
- Docstrings for all public functions
- black/ruff for formatting

### TypeScript
- Strict mode enabled
- ESLint + Prettier
- React functional components only
- Server Actions for API calls
- Vitest for testing

## Testing Requirements

**Coverage Gate**: ≥90% line coverage required (constitutional requirement)

### TDD Workflow
1. **RED**: Write failing test first
2. **GREEN**: Minimal implementation to pass
3. **REFACTOR**: Clean up while tests stay green
4. **COVERAGE**: Verify ≥90% maintained
5. **COMMIT**: Only commit when tests pass

### Test Categories
- **Unit**: Vitest - individual functions/classes
- **Component**: React Testing Library - UI components
- **Integration**: Vitest + MSW - Server Actions
- **E2E**: Playwright - complete user journeys

## Recent Changes

### 001-llm-risk-playground (2026-01-09)
- Added Next.js frontend project structure
- Added AI SDK for Azure OpenAI integration
- Added Vitest, Playwright, MSW for testing
- Added TypeScript port of refund_agent.py vulnerabilities

<!-- MANUAL ADDITIONS START -->
<!-- Add project-specific notes here that should persist across updates -->
<!-- MANUAL ADDITIONS END -->
