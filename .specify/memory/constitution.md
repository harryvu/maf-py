<!--
SYNC IMPACT REPORT - Constitution Update

Version Change: Template → 1.0.0 (Initial establishment)

Modified Principles:
- All placeholders filled with concrete educational/TDD principles
- Added explicit 90% coverage requirement as mandated
- Established Microsoft Agent Framework specific principles

Added Sections:
- Testing Requirements (comprehensive coverage and testing strategy)
- Code Quality Standards (Python-specific standards)

Removed Sections:
- Template placeholder sections replaced with concrete requirements

Templates Requiring Updates:
- ✅ plan-template.md: Constitution Check section aligns with new principles
- ✅ spec-template.md: User story testing requirements support TDD approach
- ✅ tasks-template.md: Test-first task organization supports TDD workflow

Follow-up TODOs:
- None - all placeholders have been filled with concrete values
-->

# MAF-PY Constitution
<!-- Microsoft Agent Framework Python Tutorials Constitution -->

## Core Principles

### I. Test-Driven Development (NON-NEGOTIABLE)
**TDD is mandatory for all code**: Tests MUST be written first, approved by stakeholders, deliberately fail, then implementation follows. Red-Green-Refactor cycle is strictly enforced. No code commits without corresponding tests.

**Coverage requirement**: Minimum 90% code coverage required across all modules. Coverage reports MUST be generated and validated before any pull request merge.

**Rationale**: Quality assurance and maintainability of tutorial code that developers will learn from requires exemplary testing practices.

### II. Educational Clarity
**Tutorial-first approach**: All code MUST be written with learning objectives in mind. Complex implementations require step-by-step documentation. Code clarity takes precedence over performance optimizations unless specifically demonstrating performance concepts.

**Documentation standards**: Every function, class, and module MUST have clear docstrings explaining purpose, parameters, return values, and educational context.

**Rationale**: This project serves as educational material for Microsoft Agent Framework; code readability and learning value are paramount.

### III. Agent Framework Integration
**Microsoft Agent Framework compliance**: All agents MUST follow the official Agent Framework patterns and conventions. Integration with agent-framework>=1.0.0b251120 is mandatory.

**Best practices demonstration**: Code MUST exemplify proper agent design patterns, error handling, and state management as defined by the Microsoft Agent Framework.

**Rationale**: Students learning from these tutorials must see correct implementation patterns that they can confidently use in production.

### IV. Incremental Complexity
**Progressive learning curve**: Tutorial agents progress from simple (simplest_agent.py) to complex (multi-turn_convs.py, vision_agent.py). Each example builds upon previous concepts without breaking backward compatibility.

**Standalone examples**: Each agent file MUST be independently executable and self-contained with clear prerequisites documented.

**Rationale**: Learners need to build confidence with simple concepts before advancing to complex scenarios.

### V. Production Readiness Standards
**Error handling**: All agents MUST implement comprehensive error handling with user-friendly messages. No unhandled exceptions in tutorial code.

**Configuration management**: Environment variables and configuration MUST be properly managed using python-dotenv patterns. Sensitive data never hardcoded.

**Rationale**: Tutorial code should demonstrate production-ready practices, not just functional proof-of-concepts.

## Testing Requirements
**Coverage gates**: All pull requests MUST include coverage reports showing ≥90% line coverage. Coverage checks are automated and block merges below threshold.

**Test categories required**:
- Unit tests: Test individual functions and classes in isolation
- Integration tests: Test agent interactions with external services (mocked in CI)
- Contract tests: Verify agent framework integration points
- End-to-end tests: Complete user scenarios for complex agents

**Test data management**: Test cases MUST use deterministic, reproducible test data. No dependencies on external APIs during automated testing.

## Code Quality Standards
**Python standards**: Code MUST follow PEP 8 guidelines with automated formatting via black and linting via ruff. Type hints required for all public interfaces.

**Dependency management**: Use pyproject.toml for dependency specification. Pin major versions, allow minor/patch updates. Security scanning required for all dependencies.

**Version control practices**: Conventional commits required. Feature branches for all changes. Squash merging preferred to maintain clean history.

## Governance
**Constitution authority**: This constitution supersedes all other practices and guidelines. All code reviews MUST verify constitutional compliance.

**Amendment procedure**: Constitutional changes require:
1. Proposal with rationale and impact analysis
2. Review by all project contributors 
3. Documentation of migration plan for existing code
4. Version bump following semantic versioning

**Quality gates enforcement**: 
- Pre-commit hooks enforce formatting and basic linting
- CI/CD pipeline validates test coverage ≥90%
- Automated security scanning on all dependencies
- Documentation completeness checks for new features

**Compliance verification**: All pull requests require constitutional compliance checklist completion. Reviewers MUST verify TDD practices were followed.

**Version**: 1.0.0 | **Ratified**: 2026-01-08 | **Last Amended**: 2026-01-08
