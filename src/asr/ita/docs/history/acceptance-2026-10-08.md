# Italian ASR implementation and acceptance report

> Historical evidence from 2026-10-08. Commands and paths below describe that checkout; see [current validation](../validation.md) for the layer refactor.

Completed on 2026-10-08 on Apple M5, 32 GB unified memory, macOS 27.0.1 ARM64, Python 3.12.12. Italian preparation and real-media transcription now run successfully with the existing ZIPA checkpoint and source revisions. Only `kaldilm==1.15.4` was added; the lock preserves every previously pinned package version.

## Implemented scope

The CPU pipeline performs FFprobe/FFmpeg extraction, contiguous silence-aware segmentation, strict float32 ZIPA inference with Lhotse 80-bin Fbank, full-distribution Compact CTC decoding through a statically composed lexicon/MLS trigram graph, and WER/CER evaluation. Separate preparation and local-only transcription commands are available through `python -m sicilian_asr`.

Preparation preserves original resources and creates auditable derivatives, including normalized pronunciation alternatives, exclusions/collisions, selected vocabulary, restricted ARPA with unchanged retained weights, `T`, disambiguated `L`, `G`, `LG`, `TLG`, symbol tables, and diagnostics. Cache identity includes resources, rules, settings, versions, and compiler hashes. Runtime failures retain diagnostic metadata; failed WFST decoding never falls back to greedy IPA.

The supplied CLiC-it paper's Sections 3.3/4.1 provide the static Compact CTC–lexicon–trigram architecture. ZIPA replaces the paper's trained wav2vec2 model, and PhonItalia/MLS replace its eSpeak/CommonVoice language resources as requested. This is an implementation of that architecture, not a reproduction of the paper's experiments.

## Fixed real-audio acceptance

Dataset: [Emozionalmente v1.1](https://zenodo.org/records/12616095), linked by [OpenSLR 161](https://www.openslr.org/161/), CC BY 4.0. Selection was fixed before decoding: the lexically first filename with `emotion_expressed=neutrality` in the publisher's test split. No model, lexicon, vocabulary, scales, or beams were tuned against this reference.

- Recording: `1612969363951.wav`.
- Source WAV SHA-256: `89d8b6b721f5d100ee1f9d7aeea873ab8ef18b1b51ca1c8dfad59f643da86942`.
- Test CSV SHA-256: `006d7ed32a7e351ff004788dc881b7ea35ed43fdbffea0a5c30907e64d0ebb7c`.
- Independent publisher sentence: **L’ho incontrato oggi dopo 2 anni**.
- Reference file SHA-256: `7b9404ea10ce6133cb448a0218c6fe899072b14e231d3a52c697995f5ddf2220`.
- MP4 SHA-256: `29daa7b3e0a0d829ce856c4f1dabc855dd9cdc8b4978533e81f4bbef6d26ce8a`.

`scripts/acceptance_italian.py` created black video plus AAC audio in MP4. The transcription command then exercised normal FFmpeg extraction, not direct WAV inference. FFmpeg 9.0.2 selected audio stream 1, with timestamp origin 0. Extracted audio contains 92,942 samples at 16 kHz: **5.808875 seconds**, one segment. ZIPA returned **287 valid frames**, expanded privately to **574 Compact CTC frames**. The resulting lattice had 1,422,039 arcs.

Actual outputs:

```text
Reference normalized: l'ho incontrato oggi dopo 2 anni
WFST transcription:   lo ha incontrato dogi dopo due anni
Greedy IPA:           loinkontratoɔddʒidopodueanni
```

| Metric | Substitutions | Deletions | Insertions | Reference denominator | Rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| WER | 3 | 0 | 1 | 6 words | **66.67%** |
| CER | 3 | 1 | 3 | 27 nonspace characters | **25.93%** |

The alignment substitutes `l'ho→lo`, inserts `ha`, substitutes `oggi→dogi`, and substitutes `2→due`. The digit remains unchanged in the publisher reference under the requested evaluation policy; this orthographic mismatch counts as an error. `l'ho` and `2` are the two OOV tokens/types for both the full supported lexicon and selected vocabulary. The remaining three words match. These scores establish successful integration, not good recognition accuracy or general Italian accuracy. No PER was computed because no IPA reference was supplied.

The six required outputs and additional diagnostics are saved in `outputs/ita_acceptance/`: `transcription.txt`, `ipa_greedy.txt`, `reference_normalized.txt`, `comparison.txt`, `metrics.json`, `run_metadata.json`, plus `segments.json`, `reference_cues.json`, and `audio.wav`. Probability export was disabled. Selection provenance is in `data/acceptance/emozionalmente/selection.json`. These files remain Git-ignored.

## Measured runtime and graph size

| Acceptance stage | Seconds |
| --- | ---: |
| Source hash verification | 6.992 |
| Prepared artifact hash verification | 11.975 |
| Graph loading | 2.022 |
| FFmpeg extraction | 0.118 |
| Segmentation | 0.001 |
| ZIPA loading | 3.285 |
| Acoustic inference | 0.237 |
| WFST decoding | 2.368 |
| Reference parsing and evaluation | 0.001 |
| Entire run, including bookkeeping/output hashing | **27.079** |

Peak process RSS was **9,358,131,200 bytes (9.36 GB / 8.72 GiB)**. Acoustic inference plus decoding took 2.605 seconds for 5.809 seconds of audio (approximately 0.45 real-time factor); the full run was approximately 4.66 times audio duration due primarily to verification/loading. These are one warm-filesystem run's measurements, not a throughput benchmark.

The default graph retains 20,000 words plus the original LM unknown symbol from 72,286 shared supported words. Original ARPA counts are 700,630/10,502,874/27,413,663 for orders 1/2/3. Restricted counts are 20,003/3,729,469/13,365,926. No LM retraining or weight renormalization occurred.

| Cached graph | States | Arcs |
| --- | ---: | ---: |
| T | 129 | 382 |
| L, disambiguated | 142,071 | 162,117 |
| G | 3,730,123 | 20,845,518 |
| LG | 26,018,997 | 53,910,520 |
| TLG | 78,058,064 | 184,007,650 |

Graph compilation took **152.831 seconds**, excluding acquisition, lexicon conversion, ARPA restriction, and final cache hashing. `TLG.pt` occupies **4,511,663,696 bytes**. Preparation succeeded without shrinking the requested vocabulary. Current cache key: `13dfce46bb2839d35f441e05af7ee364b5050372ce433a70f04880f7dc95e4d1`.

## Validation

- **50 tests passed**, retaining the original tests with their obsolete disabled-decoder/FBK expectations updated.
- Synthetic Compact CTC tests cover repeats, blank separation, optional initial prefix, no interword prefix, auxiliary-frame values/lengths, termination, and unchanged original matrices; ordinary greedy CTC is tested separately.
- A tiny weighted lexicon/trigram exercises full composition, homophones, alternatives, unknowns, complete distributions, LM-scale effects, impossible lattices, and intermediate graph serialization.
- Exhaustive short controlled word sequences, including unknowns and the empty sentence, compare natural-log grammar path scores against KenLM's log10 scores with BOS/EOS and backoff.
- Tests cover audio conversion/stream selection, contiguous segmentation/timestamps, subtitle metadata and numeric lines, Unicode/apostrophe normalization, edit counts/undefined rates, OOV reporting, lexical alternatives/exclusions, retained ARPA weights, stale/corrupt caches, academic checksums, derived-artifact integrity, unsupported settings, and failed-run metadata.
- `scripts/validate_environment.py` passed, including kaldilm and native k2/PyTorch compatibility. `scripts/validate_resources.py --strict --config ita` passed with both resources and prepared graph ready.

## Remaining limitations

The 20,000-word static graph is large even after vocabulary restriction. Full shared vocabulary is configurable but was not compiled in this acceptance run. Short-segment acceptance does not establish long-recording accuracy or performance. Sequential cuts prevent duplicate/omitted audio but can break linguistic context, and each segment restarts the LM. Full-file audio extraction currently retains the waveform in memory.

Compact CTC can emit repeated labels without an intervening acoustic blank; this differs from ordinary CTC collapse and is intentional. Tropical epsilon-backoff decoding follows Icefall; controlled fixtures match KenLM, but arbitrary ARPA models can have competing backoff paths. See [architecture.md](../../../../../docs/architecture.md).

PhonItalia's ambiguous `tz`/`bv` entries remain excluded. Its inherited licensing discrepancy remains unresolved. Unknown handling does not recover arbitrary OOV words, and the observed contraction/number mismatch exposes this vocabulary limitation. Subtitle omissions, paraphrasing, overlap, and synchronization require independent reference curation; no word-level forced alignment is provided.

MPS remains unsupported on the pinned stack. G2P, acoustic/LM training, Sicilian resources, and code-switching remain outside scope.
