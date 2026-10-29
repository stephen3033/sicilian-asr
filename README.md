# Sicilian ASR

Local speech recognition: **media → audio segments → ZIPA IPA probabilities → WFST words → evaluation**.

Italian uses PhonItalia and the pretrained MLS trigram. Sicilian uses reviewed text, provisional pronunciations and a locally trained KenLM model. Both run on CPU with separate graphs; bilingual decoding is reserved.

![Shared acoustic model and separate language decoders](docs/images/pipeline-architecture.png)

## Current best recorded scores

Overall strict word error rate (WER) and character error rate (CER) for each language's best selected configuration, verified against the saved evaluation reports on 2026-10-28. Rates aggregate edit counts across the stated evaluation set; lower is better.

| Language | WER | CER | Evaluation set | Selected configuration / evidence |
| --- | ---: | ---: | --- | --- |
| Sicilian (`scn`) | **44.53%** | **16.72%** | 3 development recordings, 3,209 reference words | [`baseline-lm0.8-b0`](docs/scn/residual-diagnosis.md), retained research selection |
| Italian (`ita`) | **17.47%** | **6.73%** | 202 held-out test recordings, 1,248 reference words | [Combined pronunciation alternatives and contractions](docs/ita/README.md#measured-results), acoustic/LM scales 1/1 |

Italian's separate spoken-form scoring gives **16.35% WER / 6.03% CER** on the same test set. Sicilian's selected result uses development data and has not changed production defaults; independent acceptance requires fresh recordings. The Italian benchmark contains 18 repeated acted sentences. These scores describe different evaluation sets, not a directly comparable estimate of general accuracy.

## Run

Requires macOS ARM64, Python 3.12, FFmpeg/FFprobe and Apple Command Line Tools.

```sh
uv sync --frozen --extra research --extra corpus
.venv/bin/sicilian-asr prepare --config ita
.venv/bin/sicilian-asr transcribe --config ita \
  --input recording.mp4 --reference subtitles.srt --output outputs/example
```

The reference is optional. Use a fresh output directory. Existing checkouts should run `scripts/migrate_layout.py --dry-run`, then `--apply`, before preparation.

| Guide | Contents |
| --- | --- |
| [Usage](docs/usage.md) | Setup, configuration, outputs and development |
| [Architecture](docs/architecture.md) | Code layers, contracts and WFST mechanics |
| [Italian](docs/ita/README.md) | Workflow, measured results and limitations |
| [Sicilian](docs/scn/README.md) | Corpus review, training and measured results |
| [Resources](docs/resources.md) | Integrity, migration, provenance and licensing |
| [Validation](docs/validation.md) | Reproduction evidence |

Code is MIT; model and dataset terms are separate. Source code lives in `src`, settings in `configs`, resource inputs in `resources`, and project documentation in `docs`.
