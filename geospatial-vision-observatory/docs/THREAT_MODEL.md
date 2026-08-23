# Threat model

## Assets and trust boundaries

Protected assets include observation authenticity/provenance, curated training data identity, audit-chain integrity, model artifacts, reported scientific results, service availability and host/container integrity.

Untrusted boundaries include upstream HTTP/STAC payloads, DNS/proxy behavior, image decoders/GDAL, package/container supply chains, model archive files, browser requests and operator-supplied configuration.

## Principal threats and controls

| Threat | Primary controls | Residual risk / next control |
| --- | --- | --- |
| Source abuse / retry storm | low polling cadence, persistent hourly/daily budgets, spacing, capped retries, `Retry-After`, circuit breaker | multiple deployments need a shared fleet-level budget |
| SSRF / DNS rebinding | no user fetch URL, exact HTTPS hosts, public-IP resolution, port 443, redirects disabled | enforce the same egress rules at firewall/orchestrator level |
| Malicious/oversized preview | content-type/signature checks, streamed size ceiling, image decoder verification, geometry bounds | decoder zero-days remain possible; maintain isolation and patch cadence |
| Remote COG abuse | fixed upstream URL derivation/allowlists, operator-only workflow, bounded AOI/window | GDAL has a broad parsing/network surface; run curation in an isolated job environment |
| Tampered/replayed observation | source-ID deduplication, timestamps, SHA-256, atomic writes, provenance, audit chain | upstream compromise is not independently signed |
| Curated-data mutation | every generated raster/image SHA-256 in manifest; reuse requires re-verification; dataset signature covers all scene identities | source URLs may later serve altered bytes; preserve curated outputs or upstream ETags/checksums when available |
| Split leakage / metric manipulation | AOI-level frozen split, validation-only seed selection, explicit seed-selection record, external test evaluated after selection | humans can still repeatedly rerun after inspecting test results; treat the external set as frozen governance data |
| Model artifact replacement | local-only bundle, SHA-256 manifest, evaluation/bundle experiment-signature match, API re-hashes mounted model before exposing metrics | sign bundle manifest externally for stronger provenance |
| Unsafe model loading | PT2 export archive, no deployment pickle checkpoint, digest verification before `torch.export.load`, read-only model mount | model runtimes remain complex; never load arbitrary third-party archives and isolate inference |
| Audit-log rewrite | chained digest plus optional HMAC | attacker with DB + HMAC key can rewrite history; publish signed roots externally |
| Secret leakage | secret-aware settings, no secrets in query/dashboard, local `.env` ignored | privileged host users may inspect process environment; use orchestrator secret stores in production |
| API/browser attack | read-only API, request bounds, CSP, clickjacking/MIME/referrer protections, loopback Compose bind | add identity-aware TLS proxy for remote access |
| Supply-chain compromise | exact direct versions, lock-preferred environment, resolved-environment freeze, CycloneDX SBOM, Dependabot, Ruff/mypy/tests, `pip-audit`, CodeQL, Trivy, non-root/read-only runtime | sign release/SBOM and pin base-image digest for production |

## Fail-closed scientific evidence

The project deliberately treats scientific evidence as an integrity-sensitive asset. The GitHub result block is generated only from a trained model/evaluation pair whose dataset signature, experiment signature and model SHA-256 agree. The API refuses to surface model metrics when the mounted model digest differs from the report.

The research gate cannot convert a failed threshold into a pass. A threshold miss is recorded as non-eligible; local runtime may continue, but strict publication mode still fails. Changing a gate is a policy/code change that must be independently reviewable in version control.

## Incident response

1. Stop ingestion/training/inference jobs without deleting evidence.
2. Preserve the data volume and generated manifests read-only.
3. Export logs, audit-chain head, dataset signature, experiment signature and model bundle hashes.
4. Compare artifacts against the repository/release evidence.
5. Rotate any affected secrets and rebuild only from verified inputs.
6. Re-run external evaluation if model/data integrity was affected.
7. Document the affected acquisition/model interval and invalidate published results when provenance cannot be restored.
