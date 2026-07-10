# Review Rules — Codebase-specific guidance for AI reviewer

<!-- Place this file at .github/review-rules.md in your repo. -->
<!-- The content below is injected into every reviewer pass. -->

## General
- Focus on bugs that could cause runtime failures, data loss, or security issues.
- This project uses Python 3.12+ with Flask. Check for async correctness, SQLAlchemy session leaks, and CSRF issues.
- Database is SQLite — flag any operations that could cause "database is locked" under concurrency.

## Architecture
- Deployer uses a job queue with SQLite. Race conditions in job state transitions are critical.
- All API keys and tokens must be encrypted at rest. Flag any plaintext storage.
- Docker volumes are shared between instances. Flag any path collisions or missing isolation.

## Skip
- Do not flag the absence of type hints in existing code.
- Do not flag TODO/FIXME comments.
- Do not flag test coverage gaps unless tied to a specific runtime risk.
