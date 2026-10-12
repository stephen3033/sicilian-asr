# Italian ASR

Italian preparation converts the original PhonItalia dictionary into ZIPA token sequences, derives audited pronunciation/contraction alternatives, selects the 20,000-word MLS overlap, and compiles a Compact CTC–lexicon–trigram graph.

| Module | Responsibility |
| --- | --- |
| `prepare.py` | Resource verification, graph identity, compilation workflow |
| `lexicon.py` | PhonItalia conversion and pronunciation alternatives |
| `lm.py` | Italian vocabulary selection |
| `spoken.py`, `evaluation.py` | Italian spoken-number metrics |
| `benchmark.py` | Fixed study selection, inference, aggregation, and adoption gate |
| `acceptance.py` | Deterministic real-audio acceptance selection |

[config.yaml](config.yaml) owns language settings. [resources/manifest.yaml](resources/manifest.yaml) owns PhonItalia, MLS, and Emozionalmente provenance. Acquired dictionary and LM files remain ignored inside their resource folders; graphs stay under `.cache/graphs/ita`.

## Run and reproduce

```sh
.venv/bin/sicilian-asr prepare --config ita
.venv/bin/python scripts/benchmark_italian.py --output outputs/ita_study_new
.venv/bin/python scripts/acceptance_italian.py --run --diagnostics \
  --data-output data/acceptance/new --output outputs/ita_acceptance_new
```

The benchmark prepares an unadapted baseline automatically, or accepts `--baseline` for an existing verified graph. It reuses fresh WAV probabilities across ablations, evaluates five scales on development data, freezes selection, and then evaluates held-out data. Restart with identical code/configuration/output arguments to resume. The study excludes the previously inspected acceptance clip and keeps publisher references unchanged.

This corpus contains 18 repeated acted sentences. It is a regression benchmark, not a broad Italian accuracy estimate. See [current validation](docs/validation.md), [conversion rules](docs/linguistic-conversion.md), and [resource evidence](docs/resources.md). The dated [original acceptance](docs/history/acceptance-2026-10-08.md) and [repair study](docs/history/repair-2026-10-08.md) preserve prior findings.
