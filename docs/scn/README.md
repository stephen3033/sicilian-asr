# Sicilian pipeline

A CPU proof of concept: **ZIPA → Sicilian pronunciations → KenLM → Compact CTC WFST**. Output targets DOS24. Pronunciations and contextual reviews remain provisional project judgments. [Resources](resources.md) documents their evidence and limitations.

## Run

After a study has selected a local model:

```sh
.venv/bin/python -m pipeline prepare --config scn
.venv/bin/python -m pipeline transcribe --config scn \
  --input recording.mp4 --reference subtitles.srt --output outputs/scn_example
```

For a new study, install the [environment](../usage.md) and acquire/screen the [background sources](resources.md#reproduce-the-source-audit), then run:

```sh
.venv/bin/python scripts/setup_scn_tools.py
.venv/bin/python -m pipeline scn-poc --stage acquire
.venv/bin/python -m pipeline scn-poc --stage corpus
.venv/bin/python -m pipeline scn-poc --stage train
.venv/bin/python -m pipeline scn-poc --stage development
.venv/bin/python -m pipeline scn-poc --stage test
.venv/bin/python -m pipeline scn-poc --stage report
```

Stages resume from verified artifacts. `--output` selects a study directory; the default is `outputs/scn_poc`. Final scores freeze that directory's corpus/training/development stages. The selected model receipt and acoustic scale live in `.cache/scn/selected.json`; models, graphs, source media and acoustic outputs remain in their respective `.cache` folders.

## Training and evaluation

The [recipe](../../configs/scn-poc.json) fixes six uploaded Sicilian caption tracks from Sicilian with Nick for training, one for development, and two for testing. Uploaded captions do not establish authorship or linguistic correctness. Only reviewed background passages enter training. [Training reviews](../../resources/scn/poc_reviews.json) bind corrections to exact source text; [final reviews](../../resources/scn/poc_test_reviews.json) are separate.

Training sentences are deduplicated and checked against fixed 12-word development/final-caption overlap fingerprints. Final words or scores do not select vocabulary, order or scale. Background review is development-informed. KenLM trains unpruned bigrams/trigrams with a logged sparse-data discount fallback; ZIPA is unchanged. Vocabulary comes from each training slice and supported pronunciations, with at most four pronunciations per word.

At each corpus size, prefer a bigram within 1% of trigram development perplexity on the same vocabulary. Decode the chosen order across recipe scales, then select the smallest eligible model within the recipe's WER tolerance of the best. If no model meets the development OOV target, retain that failure explicitly. Perplexities across different vocabularies are not comparable.

Strict scores preserve accents, retroflex spelling and lexical apostrophes. Complete audio is scored against reviewed caption text, retaining foreign examples and timing gaps. Caption omissions/paraphrases affect scores; no phoneme error rate is available without gold IPA. References never propose decoding candidates.

## Measured results — 2026-10-12

Selected: **11,441 training tokens, trigram, 3,318 words**, 14.3 MiB graph, acoustic scale 1. Development WER/CER: **56.23%/23.37%**; OOV: **6.09%**, missing the 5% target. The admitted corpus was exhausted below the 25k checkpoint.

| Final video | WER | CER | Vocabulary OOV |
| --- | ---: | ---: | ---: |
| `sctItzC2xgw` | 67.55% | 28.75% | 15.99% |
| `C2xRDIuT1Dw` | 74.22% | 34.43% | 19.33% |
| Combined, 2,120 words | **71.60%** | **32.21%** | **18.02%** |

Accuracy is insufficient for reliable general transcription. One learning channel is not representative Sicilian speech. Priorities are independent references, verified regional pronunciations and reviewed vocabulary expansion; bilingual decoding remains deferred.

The 17.2 minutes of audio took 44.3 seconds for first acoustic inference and 21.4 seconds for cached-posterior decoding, with 2.51 GiB peak process memory during concurrent validation. These exclude acquisition, tool builds and graph construction. [Tracked results](results.json) retain detailed measurements; [validation](../validation.md) distinguishes fresh reproduction from this historical baseline. No post-test tuning was performed.
