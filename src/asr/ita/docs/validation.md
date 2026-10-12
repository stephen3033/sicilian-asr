# Italian reproduction — 2026-10-12

The refactored pipeline reproduces the original Italian repair study: **all 1,724 decoded results and metric objects match**, and all 367 fresh acoustic probability arrays are identical. The adoption gate passes with pronunciation alternatives and contractions enabled at acoustic/LM scales **1/1** and a **20,000-word** vocabulary.

## Full study

| Split / configuration | Recordings | Strict WER | Strict CER | Spoken WER | Spoken CER |
| --- | ---: | ---: | ---: | ---: | ---: |
| Development / original | 165 | 34.14% | 13.70% | 33.13% | 12.89% |
| Development / repaired | 165 | 22.02% | 8.87% | 20.91% | 8.19% |
| Held-out / original | 202 | 30.69% | 11.45% | 29.57% | 10.58% |
| Held-out / repaired | 202 | 17.47% | 6.73% | 16.35% | 6.03% |

The 165 development recordings and 202 held-out recordings use speaker-disjoint publisher splits. The previously inspected acceptance clip is excluded. Fresh WAV posteriors were computed once and shared across baseline, pronunciation-only, contractions-only, and combined graphs. The combined graph was tested at scales 0.5, 0.75, 1, 1.5, and 2; development selection was frozen before held-out evaluation.

References, split membership, original pronunciation alternatives, and retained LM weights are unchanged. No retuning was needed. This is a regression corpus of **18 repeated acted sentences**, not a broad Italian or Sicilian accuracy estimate. See the [dated repair report](history/repair-2026-10-08.md) for the original causal analysis, ablations, and remaining errors.

## Real-media acceptance

The same independently selected `1612969363951.wav` was converted to an MP4 and passed through media extraction, segmentation, ZIPA, WFST decoding, and evaluation. Both runs exported original probabilities and candidate diagnostics.

```text
Reference: L’ho incontrato oggi dopo 2 anni
Baseline:  lo ha incontrato dogi dopo due anni
Repaired:  l'ho incontrato oggi dopo due anni
```

| Configuration | Strict WER | Strict CER | Spoken WER | Spoken CER | Entire run | Peak RSS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline | 66.67% | 25.93% | 50.00% | 13.79% | 27.49 s | 9.05 GB |
| Repaired | 16.67% | 11.11% | 0.00% | 0.00% | 33.26 s | 12.40 GB |

The repaired strict error is the written `2` versus spoken `due`; spoken-form scores preserve that distinction separately. Timings include resource/graph verification and loading and are single-run measurements, not throughput estimates. Segment times are not word alignment; sampled competing paths are not exhaustive N-best results or confidence scores.

| Newly built graph | Compilation only | TLG file |
| --- | ---: | ---: |
| Baseline | 138.05 s | 4.51 GB |
| Repaired | 190.84 s | 6.22 GB |

Compilation excludes resource verification, vocabulary/ARPA preparation, and final hashing. Temporary ablation graphs were removed by the benchmark; baseline and repaired graphs remain cached. Source-layout changes deliberately created new cache identities.

## Artifacts and commands

- `outputs/ita_layout_study/`: frozen configuration, selection, new probabilities, per-recording results, graph receipts, and aggregate report.
- `outputs/ita_layout_acceptance_baseline/` and `outputs/ita_layout_acceptance_repaired/`: complete MP4 runs.
- [Tracked validation evidence](../../../../docs/validation.json): aggregate results, artifact comparisons, graph sizes, and environment outcomes.

```sh
.venv/bin/python scripts/benchmark_italian.py --output outputs/ita_study_new
.venv/bin/python scripts/acceptance_italian.py --run --diagnostics \
  --data-output data/acceptance/new --output outputs/ita_acceptance_new
```

For baseline MP4 acceptance, pass `--config baseline.yaml` with `languages: [ita]` and both `decoder.pronunciation_variants` and `decoder.contractions` set to false. Use a fresh output directory. Historical graphs, publisher data, and original run artifacts were preserved.
