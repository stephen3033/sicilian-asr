# Validation after the layer refactor

Reproduced on **2026-10-12**, using Python 3.12.12 on macOS-27.0.1-arm64-arm-64bit. Dependency pins, model bytes, language resources, and decoding settings are unchanged.

| Check | Observed result |
| --- | --- |
| Test suite | **107 passed**; source/test layers, migration, CLI, and configuration covered |
| Dependencies and CPU probes | Pass; ordinary and upstream TorchScript outputs are finite and agree within tolerance |
| Metal probes | Still unsupported: ordinary ZIPA fails at the k2 activation; TorchScript returns nonfinite values |
| Italian resources and current graph | Both ready; checksums and prepared artifacts verified |
| Full Italian study | **367 new inferences; 1,724 decoding evaluations**; all paths and metrics match historical results |
| Probability comparison | All 367 original-frame arrays match exactly; maximum difference **0** |
| MP4 acceptance | Baseline and repaired transcripts/strict metrics reproduced; diagnostics and probability exports present |
| Sicilian offline reproduction | Corpus, audit measurements, LM diagnostics, and IPA compatibility unchanged |
| Packaging | All layer packages/configuration/manifests included; acquired corpora, weights, and graphs excluded |

The [Italian report](../src/asr/ita/docs/validation.md) contains the measured accuracy and runtime results. [validation.json](validation.json) records aggregate and per-run evidence. The original **416 output files** remain byte-for-byte unchanged; new results are under `outputs/ita_layout_*`.

## Reproduce

```sh
.venv/bin/python -m pytest -q
.venv/bin/python scripts/validate_environment.py --acoustic
.venv/bin/python scripts/benchmark_italian.py --output outputs/ita_study_new
.venv/bin/python scripts/validate_resources.py --strict --config ita
```

Run the [Sicilian offline workflow](../src/asr/scn/resources/sources/README.md) to reproduce its evidence. The refactor changes audit implementation/catalog fingerprints, not measurements or review decisions. Historical research-validation records are explicitly labeled and retained.

The native checks were run with host access after the sandbox blocked OpenMP shared memory. Hardware-probe results above come from the successful host-access run. Reports and large artifacts remain local; the tracked JSON preserves their measured summary. Historical results are available in the [initialization record](history/initialization-2026-10-08.md) and the Italian report's dated comparisons.

## Sicilian POC

The 2026-10-12 [Sicilian study](../src/asr/scn/docs/poc.md) adds actual LM training, graph construction and final-video evaluation. Its [tracked results](../src/asr/scn/docs/results.json) record accuracy, cache identities and validation. The rebuilt Italian acceptance graph reproduces the saved transcript, metrics, posteriors and path scores exactly. Earlier full Italian study results above remain the layout-refactor baseline.
