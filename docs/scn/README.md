# Sicilian pipeline

The [accuracy improvement workflow](improvement.md) tests programmatic DOS24
conversion, matched KenLM controls and channel-grounded pronunciation mappings.
Its comparison baseline is 44.53% WER / 16.72% CER. Targets remain WER below 10%
and CER below 5%; independent acceptance requires fresh recordings.

The [residual-error investigation](residual-diagnosis.md) classifies all 1,184
known-word errors in that baseline, records unresolved attribution explicitly,
and tests LM construction, pronunciation costs, and segmentation with ZIPA
frozen. Its research candidate does not change production selection.

A CPU proof of concept: **ZIPA → DOS24 word lexicon → KenLM → Compact CTC WFST**.
The lexicon maps acoustic labels to written words. Wikipedia-derived phonology
provides initial hypotheses and fallbacks; repeated ZIPA observations paired
with DOS24 training text guide label choices and costs. Success is measured by
DOS24 WER/CER, not linguistic IPA agreement. The [training-label audit and
empirical mapping experiments](improvement.md#execution) retain per-recording
and word-position evidence. [Resources](resources.md) documents source limits.

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

### Reproduce the expansion experiments

```sh
.venv/bin/python -m pipeline scn-improve --stage all
```

The [workflow](improvement.md) describes individual stages, receipts, corpus review
queues and the final experimental transcription configuration. Earlier diagnosis
results below are historical evidence; their one-off driver has been retired.

Run the residual investigation through the same workflow:

```sh
.venv/bin/python -m pipeline scn-improve --config configs/scn-residual.json --stage all
```

The stages are `residual-prepare`, `residual-score`, `residual-models`,
`residual-calibrate`, `residual-audio`, `residual-repairs`, `residual-tests`,
`residual-cleanup`, `residual-validate`, and `residual-report`. The report links
the per-error evidence ledger, missing-facts review queue, controlled outcomes,
and experimental transcription configuration. Development-constrained paths
never become training evidence.

#### Diagnostic results — 2026-10-19

The [tracked diagnosis](diagnosis-results.json) records 28 configurations on the existing development and two final recordings. Separate interventions produced these combined final-video scores:

| Configuration | WER |
| --- | ---: |
| Frozen production baseline | 71.60% |
| Exploratory final-vowel pronunciation alternatives | 63.44% |
| Soft-C/G gemination repair | 69.76% |
| Rule pronunciations with seed fallback | 69.76% |
| Shorter audio segments | 69.86% |
| Elision-preserving training normalization, original scoring | 71.46% |
| Doubled search beams and active-state limit | 71.65% |

The final-vowel probe had the largest measured effect with vocabulary and LM fixed. It indicates a ZIPA/pronunciation interface mismatch; it does not distinguish inaccurate acoustic labels from unmodeled speech variation. The normalization defect is real but small here. The existing trigram at LM weight 1 remained best on development data; unigram and bigram controls scored 76.13% and 71.84% on final videos. Only 4.29% of final-caption trigrams occur in training.

Coverage is independently limiting: 377 of 382 OOV reference tokens are absent from training text, while five were present but rejected by pronunciation generation. Another 1,035 substitutions/deletions involve words already in vocabulary. Adding 322 supported reference word types to a flat-grammar control reduced its WER from 86.18% to 81.13%; that pair uses identical word costs and is not a learned-LM result. Ignoring vowel accents and apostrophes alone scores 69.53%. These effects overlap and are not additive. The production model remains unchanged.

### Corpus and model selection

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
