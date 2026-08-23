# Security policy

## Reporting a vulnerability

Please report security vulnerabilities privately through **GitHub Security Advisories** for this repository. Do not open a public issue containing exploit payloads, credentials, private infrastructure details, or data that could make exploitation easier.

Include, when possible, the affected component, reproduction conditions, impact, and a minimal proof of concept. Reports should avoid interacting with third-party systems beyond what is necessary to demonstrate the issue.

## Supported release

Only the current stable release on the default branch is supported. This research showcase carries no uptime or security-response SLA.

## Security boundaries

The project is designed around a narrow trust boundary:

- operational outbound traffic is HTTPS-only and restricted to explicit source hosts;
- redirects and private-address resolution are rejected for operational ingestion;
- upstream payloads are size-bounded and decoded before persistence;
- request budgets, retry bounds, and circuit breaking reduce upstream abuse and retry storms;
- frames and model-support files are content-addressed or SHA-256 verified;
- optional Docker Desktop containers run non-root, read-only, with Linux capabilities dropped;
- deployment model loading is local-only and verifies a `torch.export` PT2 bundle before use;
- dashboard/API responses use restrictive browser security headers;
- dependency, source, and container scanning run in CI.

The detailed attack analysis and residual risks are documented in [`docs/THREAT_MODEL.md`](docs/THREAT_MODEL.md).

## Responsible disclosure expectations

Please allow maintainers reasonable time to investigate before public disclosure. Security fixes should include a regression test whenever practical. Never include live credentials in tests, issues, pull requests, or advisories.
