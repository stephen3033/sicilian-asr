# Italian pipeline

Preparation converts PhonItalia to ZIPA tokens, adds the configured pronunciation/contraction alternatives, selects the 20,000-word MLS overlap and compiles a Compact CTC graph. [Settings](../../configs/ita.yaml) and [resource/conversion rules](resources.md) define the inputs.

## Run and reproduce

```sh
.venv/bin/sicilian-asr prepare --config ita
.venv/bin/sicilian-asr transcribe --config ita \
  --input recording.mp4 --reference subtitles.srt --output outputs/ita_example
.venv/bin/python scripts/benchmark_italian.py --output outputs/ita_study_new
.venv/bin/python scripts/acceptance_italian.py --run --diagnostics \
  --data-output data/acceptance/new --output outputs/ita_acceptance_new
```

See [setup and outputs](../usage.md). The benchmark builds an unadapted baseline, or accepts `--baseline` for a verified graph. It shares fresh posteriors across ablations, selects among five acoustic scales on development data, freezes selection, then evaluates held-out data. Identical inputs/code/output arguments allow resumption. The inspected acceptance clip is excluded from the test split.

## Measured results

The 2026-10-12 reproduction matched all 1,724 decoding evaluations and 367 acoustic arrays. The adoption gate selected pronunciation alternatives and contractions at acoustic/LM scales 1/1.

| Split / configuration | Recordings | Strict WER | Strict CER | Spoken WER | Spoken CER |
| --- | ---: | ---: | ---: | ---: | ---: |
| Development / baseline | 165 | 34.14% | 13.70% | 33.13% | 12.89% |
| Development / combined | 165 | 22.02% | 8.87% | 20.91% | 8.19% |
| Test / baseline | 202 | 30.69% | 11.45% | 29.57% | 10.58% |
| Test / combined | 202 | 17.47% | 6.73% | 16.35% | 6.03% |

These speaker-disjoint splits contain 18 repeated acted sentences: a regression benchmark, not a broad accuracy estimate. References retain publisher spellings, including `E’` versus `è`. See [validation evidence](../validation.md) for the current refactor.

The MP4 acceptance reference is `L’ho incontrato oggi dopo 2 anni`. The baseline produced `lo ha incontrato dogi dopo due anni`; the combined graph produced `l'ho incontrato oggi dopo due anni`. Combined strict WER/CER were 16.67%/11.11%; spoken-form scores were zero because they separately verbalize the integer. Contractions fix a lexical coverage gap; number formatting is an evaluation distinction. Wider beams alone did not resolve those causes in the original repair study.

The original repair analysis found that ZIPA favored doubled closures and prevelar `n` where dictionary paths used `ː` and `ŋ`: `oggi`, `anni` and `incontrato` exposed the mismatch. For `oggi` versus `dogi`, the baseline margin was 0.12 natural-log units; pronunciation alternatives reversed it by 7.14. Contraction derivation separately restored missing internal-apostrophe words.

| Development ablation, acoustic scale 1 | Spoken WER | Spoken CER |
| --- | ---: | ---: |
| Baseline | 33.13% | 12.89% |
| Pronunciations only | 22.83% | 8.51% |
| Contractions only | 31.52% | 12.60% |
| Combined | 20.91% | 8.19% |

Historical baseline/combined graph files were 4.51/6.22 GB, with compilation times 138/191 seconds. Acceptance runs took 27/33 seconds and peaked at 9.05/12.40 GB. These are dated single-run observations including resource/graph checks, not throughput guarantees. Probability arrays, per-recording comparisons and frozen selections remain in the local study outputs.
