# Sicilian ASR

Local speech recognition: **media → audio segments → ZIPA IPA probabilities → WFST words → evaluation**.

Italian uses PhonItalia and the pretrained MLS trigram. Sicilian uses reviewed text, provisional pronunciations and a locally trained KenLM model. Both run on CPU with separate graphs; bilingual decoding is reserved.

![Shared acoustic model and separate language decoders](docs/images/pipeline-architecture.png)

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
