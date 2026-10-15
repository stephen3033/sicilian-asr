# Validation

The 2026-10-15 cleanup preserves measured behavior. **141 tests pass**, up from 131 before the refactor. Ruff checks and formatting pass; installed CLI entry points and package contents are verified.

| Check | Result |
| --- | --- |
| Layout | `src` contains 66 Python files only; maintained Markdown files reduced from 37 to 11, including the root index |
| Italian study | All 367 acoustic exports and 1,724 decoding evaluations match; aggregate report, model selection and adoption gate unchanged |
| Italian MP4 acceptance | Baseline and repaired runs match audio, probabilities, transcripts, metrics, segment scores and diagnostics exactly |
| Sicilian offline workflow | Extraction, audit, corpus, LM and IPA measurements unchanged; only implementation/catalog provenance hashes differ |
| Sicilian held-out test | Both videos retain identical transcripts, metrics and scores |
| Sicilian CLI | All 21 segment probability exports, transcripts, metrics and scores match |
| Integrity | Both languages pass strict resource/graph validation; all 979 prior output files remain byte-identical |
| Migration | Verified resource moves completed; a second dry run proposes no changes |
| Packaging | Wheel contains code and license metadata; source distribution includes settings, resource inputs and docs, excluding acquired data |

The fresh study outputs are `outputs/ita_cleanup_study` and `outputs/scn_cleanup_study`; acceptance runs use `outputs/ita_cleanup_acceptance_{baseline,repaired}` and `outputs/scn_cleanup_acceptance`. Comparisons use their earlier `ita_layout_*`, `scn_poc` and `scn_poc_cli` runs. Timings, memory, relocated paths and cache fingerprints may change; measured recognition results do not. Study source hashes match all 66 current modules.

CPU eager/scripted inference and their frame comparison pass. Metal remains unsupported: eager inference fails in k2 and scripted output is nonfinite. Native compilation/probes required host access after the sandbox blocked OpenMP shared memory. These are existing platform constraints; dependencies, models, scoring rules and data splits were not changed.

[Current evidence](validation.json) records the checks and local artifact paths. The [earlier layout validation](audits/layout-2026-10-12.json) and [Sicilian results](scn/results.json) remain historical snapshots. See the [Italian](ita/README.md) and [Sicilian](scn/README.md) guides for measured quality and dataset limits. Source judgments retain their original evidence dates.

## Reproduce

```sh
.venv/bin/python -m pytest -q
.venv/bin/python scripts/validate_environment.py --acoustic
.venv/bin/python scripts/validate_resources.py --strict --config ita
.venv/bin/python scripts/validate_resources.py --strict --config scn
.venv/bin/python scripts/benchmark_italian.py --output outputs/ita_study_new
uv build
```

Use fresh output directories. The language guides document the Sicilian stages and acceptance commands; `.cache/cleanup` holds this run's detailed comparison logs.
