# Sicilian YouTube POC

This study trains a small word language model and compiles a Sicilian WFST against the existing, unchanged ZIPA acoustic model. It establishes an executable pipeline; one language-learning channel is not a representative Sicilian speech benchmark.

## Measured result — 2026-10-12

Selected **11,441 training tokens, a trigram, and 3,318 words**. The TLG is 14.3 MiB; acoustic scale is 1.0. Development WER/CER is **56.23% / 23.37%**, with **6.09% OOV**. The 5% coverage gate was missed, and the reviewed corpus was exhausted below the 25k checkpoint.

| Final video | WER | CER | Vocabulary OOV |
| --- | ---: | ---: | ---: |
| `sctItzC2xgw` | 67.55% | 28.75% | 15.99% |
| `C2xRDIuT1Dw` | 74.22% | 34.43% | 19.33% |
| Combined, 2,120 reference words | **71.60%** | **32.21%** | **18.02%** |

The pipeline runs, but these errors are too high for reliable general transcription. The final videos expose considerably more missing vocabulary than development. Provisional pronunciations and caption/audio mismatches also limit interpretation; no post-test tuning was performed.

The two videos total 17.2 minutes. First acoustic inference took 44.3 seconds and decoding cached posteriors took 21.4 seconds, with a 2.51 GiB process peak. These are observed wall times during concurrent validation, not an isolated performance benchmark. A normal full-video `transcribe --config scn` run reproduced the study transcript and metrics exactly. The existing Italian acceptance clip reproduced its transcript, metrics, probabilities and path scores exactly. All 131 tests pass, including bigram/trigram WFST score equivalence, cache integrity, review binding and frozen-study checks.

## Data and review

The frozen [recipe](../resources/poc.json) uses six uploaded Sicilian caption tracks from **Sicilian with Nick** for training, one for development (`mDrR25umUpE`), and reserves `sctItzC2xgw` and `C2xRDIuT1Dw` for final evaluation. All nine captioned videos are from 2023. An uploaded track does not establish who wrote it or certify DOS24 conformity.

Training also includes individually reviewed passages from the existing UNESCO, localization, and Sicilian Wikipedia snapshots. The larger source collections are not automatically admitted. The review ledger binds each decision to the exact original text, records corrections, and excludes identified foreign demonstrations or extraction fragments. These are contextual project reviews, not native-speaker certification. Regional grammar is retained; lexical retroflex corrections are explicit, with no blanket `dd → ḍḍ` substitution.

Sentences are deduplicated. A fixed 12-word overlap screen removes training sentences shared with development or final captions; held-out captions provide only duplicate fingerprints at this stage. Final vocabulary, model order, and scale never use final-reference words or scores. Targeted background review is based on development gaps and is therefore development-informed.

## Training and selection

KenLM 0.3.0 estimates modified Kneser–Ney bigrams and trigrams. `scripts/setup_scn_tools.py` builds checksum-pinned command-line tools and Boost locally; the Python KenLM package alone does not supply `lmplz`. No acoustic model is retrained.

The lexicon combines acquired WikiPron observations with bounded Sicilian spelling-to-IPA rules. Unsupported tokenizer projections are rejected. Each retained word has at most four pronunciations. Generated records remain explicitly provisional; penultimate-vowel stress, ambiguous `z`, hiatus, loans and regional realizations need further validation. No Italian G2P fallback is used.

The learning curve starts with channel captions, then adds reviewed background sentences toward 10k/25k/50k/100k/200k tokens, bounded by the admitted corpus. Vocabulary comes only from each training slice and supported pronunciations. No arbitrary top-N vocabulary is imposed. All models use the estimator's default unpruned n-grams; a logged discount fallback is available for sparse data.

At each size, compare LM orders on the same vocabulary and prefer the bigram if its development perplexity is within 1% of the trigram. Perplexities across different vocabularies are not comparable. Decode the chosen order at acoustic scales 0.5, 1.0 and 1.5, with LM scale 1. Select the smallest tested model within one absolute WER point of the best model meeting 5% development OOV. If none qualifies, use the same WER rule across tested models and record the unmet coverage gate.

ZIPA posteriors are cached independently of the graph, so every LM sees identical acoustic evidence. Graph, model, audio, reference, and decode artifacts have checksums. Final reference review uses a separate ledger after selection. Repeating final scoring checks the frozen selection; it is not another tuning pass.

## Scoring and artifacts

Evaluation compares complete video audio with the reviewed caption text. No audio intervals are removed. Each result reports the union of caption times and all timing gaps; gaps may contain speech, so caption omissions and paraphrases can inflate WER. Foreign examples remain in evaluation references. Strict WER/CER preserve accents and lexical apostrophes, and CER excludes spaces. There is no phoneme error rate without a gold IPA reference.

| Location | Contents |
| --- | --- |
| `.cache/scn/youtube/<video>/` | Original audio/captions, timed cues, metadata and acquisition receipt |
| `.cache/scn/models/<hash>/` | Training text, vocabulary, lexicon, ARPA, estimator log and model receipt |
| `.cache/graphs/scn/<hash>/` | T/L/G/LG/TLG, word table and compilation receipt |
| `.cache/scn/posteriors/<hash>/` | Full-audio segments and reusable ZIPA frame probabilities |
| `.cache/scn/selected.json` | Active model receipt and selected acoustic scale |
| `outputs/scn_poc/` | Corpus, candidate learning curve, frozen selection, final references, comparisons and report |

Read `outputs/scn_poc/report.md` for the local run and [results.json](results.json) for the compact tracked measurements. Acquisition time, first acoustic inference, model fitting and cached-posterior decoding are separate costs. Reported RSS is the process high-water mark, not the incremental memory of one graph.

Raw channel media and transcripts remain ignored local artifacts. Source attribution and terms stay with acquisition metadata and the existing [source catalog](../resources/sources/catalog.json); this POC does not establish permission to redistribute channel media, transcripts or derived models.
