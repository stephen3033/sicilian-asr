# Sicilian ASR

A working CPU POC: **ZIPA → Sicilian pronunciations → KenLM → Compact CTC WFST**. Output targets Cademia Siciliana DOS24. Italian and Sicilian keep separate language resources and graphs; bilingual decoding is still disabled.

## Use the selected model

The model is built locally, under `.cache/scn`. Once the POC study below has selected a model:

```sh
.venv/bin/python -m pipeline prepare --config scn
.venv/bin/python -m pipeline transcribe --config scn \
  --input recording.mp4 --reference subtitles.srt --output outputs/scn_example
```

The reference is optional. Strict scoring preserves Sicilian accents, retroflex spelling, and lexical apostrophes. It does not automatically repair spelling in a supplied reference. For the reviewed YouTube references and measured accuracy, see the [POC study](docs/poc.md).

## Reproduce the POC

Install the [project environment](../../../docs/dependencies.md), then run:

```sh
.venv/bin/python scripts/setup_scn_tools.py
.venv/bin/python -m pipeline scn-poc --stage acquire
.venv/bin/python -m pipeline scn-poc --stage corpus
.venv/bin/python -m pipeline scn-poc --stage train
.venv/bin/python -m pipeline scn-poc --stage development
.venv/bin/python -m pipeline scn-poc --stage test
.venv/bin/python -m pipeline scn-poc --stage report
```

The corpus stage also needs the acquired and screened background sources; [resource workflow commands](resources/sources/README.md) reproduce those inputs. Each stage resumes from checksum-verified local artifacts. The default output is `outputs/scn_poc`; use `--output` for another study. Once final tests exist, that study's corpus, training, and development stages are frozen.

[The recipe](resources/poc.json) fixes six channel videos for training, one for development, and your two supplied videos for final testing. Only uploaded `scn` captions qualify. The three newer videos without those captions are skipped. Original captions and audio stay in `.cache/scn/youtube`; transcript corrections are explicit, hashed [review decisions](resources/poc_reviews.json).

## Find the implementation

| Module | Responsibility |
| --- | --- |
| `dataset.py`, `orthography.py` | Reviewed text, DOS24 corrections, sentence deduplication and held-out overlap screening |
| `lexicon.py` | WikiPron observations and provisional Sicilian pronunciation rules, projected to ZIPA tokens |
| `training.py` | Train and verify bigram/trigram KenLM bundles |
| `prepare.py`, `evaluation.py` | Compile the separate Sicilian graph and apply Sicilian scoring |
| `acquisition.py`, `corpus.py`, `audit.py` | Background-source downloads, extraction and contextual review |
| `discovery.py`, `report.py`, `lm.py` | Source discovery, audit reports and historical feasibility probes |

Cross-layer study orchestration lives in `pipeline/scn_poc.py` and `pipeline/scn_study.py`; YouTube acquisition lives in `media_input/youtube.py`. Tests follow those same directories.

The [source inventory](docs/resources.md), [DOS24 audit](docs/dos24-audit.md), and [pronunciation evidence](docs/pronunciation.md) explain the linguistic inputs. Pronunciations and spelling reviews are provisional project judgments, not independently certified gold. The [remaining work](docs/roadmap.md) starts with better independent references and pronunciation validation.
