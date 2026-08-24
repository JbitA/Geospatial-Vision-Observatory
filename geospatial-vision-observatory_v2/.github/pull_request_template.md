## Problem

<!-- What problem or research question does this solve? -->

## Approach

<!-- Summarize the design and why it is preferable to simpler alternatives. -->

## Evidence

- [ ] Tests added/updated
- [ ] `ruff check .` passes
- [ ] `ruff format --check .` passes
- [ ] strict `mypy src` passes
- [ ] coverage remains >= 80%
- [ ] Scientific split/provenance implications documented, if applicable
- [ ] Security implications considered, if applicable

## Integrity checklist

- [ ] No fabricated or hand-edited model metrics
- [ ] No external-test leakage into selection/tuning
- [ ] No secrets, raw large datasets, databases, or caches committed
- [ ] No model/evidence hash verification bypassed
