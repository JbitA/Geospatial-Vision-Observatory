# Contributing

Contributions are welcome when they preserve measurable geospatial ML, reproducible science, constrained external access, strong provenance and a clear Windows-first user experience.

## Windows development setup

From PowerShell:

```powershell
.\scripts\setup.cmd
```

Run the local source-quality gate:

```powershell
.\scripts\quality.cmd -Strict -AllowUntrained
```

The supported host interpreter is Python 3.12. Do not introduce a Bash/WSL requirement into the primary setup, training, validation or packaging path.

## ML and data changes

Changes to training, data preparation, metrics or evaluation must preserve the frozen train/validation/external split unless the scientific protocol is explicitly revised; prevent external-test information from influencing model selection; retain source/processing provenance and hashes; document changes in labels, resampling, masking, calibration or normalization; and add deterministic tests.

Raw Sentinel-2, WorldCover and other large rasters must not be committed. Generated publication evidence must remain compact and reviewable.

## Security-sensitive changes

Changes to fetching, URL validation, model loading, filesystem paths, process launchers, container privileges or evidence verification require focused security tests. Report vulnerabilities privately as described in [`SECURITY.md`](SECURITY.md).

## Pull requests

Keep pull requests focused. Explain the problem, design choice, tests/evidence, scientific implications, security implications and Windows compatibility impact. CI must pass before merge.
