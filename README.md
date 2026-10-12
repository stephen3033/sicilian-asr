# sicilian-asr

Italian CPU ASR: **audio/video → ZIPA frame probabilities → Compact CTC–lexicon–MLS trigram WFST → words**.

The Italian pipeline is implemented and has completed a real MP4 transcription and evaluation. Sicilian (`scn`) and Italian–Sicilian configurations remain unsupported scaffolds. Acoustic training, G2P, MPS, code-switching, and word-level alignment are outside this milestone.

## Install and prepare

The frozen environment targets native macOS ARM64, Python 3.12.12. Install [uv](https://docs.astral.sh/uv/getting-started/installation/), Apple's Command Line Tools, and FFmpeg/FFprobe first. CPU inference uses four threads. Allow at least 20 GB of free disk space; the repaired default graph alone is 6.22 GB. Graph compilation and decoding are intended for a machine with at least the validated 32 GB memory budget.

```bash
uv sync --frozen --extra research
uv run --frozen --extra research python -m sicilian_asr prepare \
  --config configs/ita.yaml
```

Preparation verifies or acquires the pinned ZIPA checkpoint, tokenizer, source trees, PhonItalia IPA dictionary, and pretrained MLS Italian LM. It extracts the original trigram and documentation, audits the lexicon, selects vocabulary, and caches `T`, disambiguated `L`, `G`, `LG`, `TLG`, symbol tables, and diagnostics. It neither trains an LM nor uses PAISÀ. The old unavailable FBK entry remains historical provenance only.

The default vocabulary comprises the 20,000 highest-unigram-probability words shared by the supported lexicon and MLS. Set `decoder.vocabulary_size: null` for the full shared vocabulary (78,883 words with contraction derivation enabled), which requires a new, potentially much larger compilation. Retained LM probabilities and backoff weights are unchanged. Compilation errors produce diagnostics without reducing vocabulary or changing topology.

## Transcribe and evaluate

From the repository root:

```bash
uv run --frozen --extra research python -m sicilian_asr transcribe \
  --config configs/ita.yaml \
  --input data/video.mp4 \
  --reference data/reference.srt \
  --output outputs/ita_test/
```

Transcription uses local prepared artifacts and requires an empty output directory. WAV and FFmpeg-supported audio/video containers are accepted. The first audio stream is selected by default; `--audio-stream N` selects an absolute FFprobe stream index. References may be SRT, VTT, or corrected UTF-8 `.txt`; omit `--reference` to transcribe without scoring.

Outputs include:

- `transcription.txt`, `ipa_greedy.txt`, `reference_normalized.txt`, `comparison.txt`, `metrics.json`, and `run_metadata.json`.
- `segments.json`, parsed `reference_cues.json` when supplied, and extracted `audio.wav`.
- Original valid probability frames with token IDs only when `--export-probabilities` is requested.
- `--diagnostics` also exports probabilities and `diagnostics.json`: the chosen pronunciation, acoustic/grammar score contributions, sampled competing paths, and OOV causes. Candidates come from the decoding lattice, never the reference. Sampled candidates are not an exhaustive N-best list.

Audio is resampled to 16 kHz mono and segmented contiguously: 20-second target, 24-second maximum, with silence of at least 300 ms preferred. Forced low-energy cuts are flagged. Sample offsets and source timestamps describe segment boundaries, not word alignment. Thresholds, beams, active-state bounds, and scales are configurable in [configs/ita.yaml](configs/ita.yaml).

Hypothesis and reference share a versioned normalizer: NFC, lowercase, standardized apostrophes, punctuation removal, and collapsed whitespace. Existing top-level WER/CER remain strict: accents, digits, and internal apostrophes remain. The separate `spoken_form` metrics verbalize standalone Italian integers from 0 through 999999 and record each transformation. Decimals, dates, signed numbers, leading zeros, and identifiers are not guessed. Both views include edit counts and denominators; CER excludes spaces. Empty references have undefined rates. `<unk>` remains visible and scored, but cannot guarantee recognition of arbitrary OOV words. Subtitle omissions, paraphrases, and synchronization errors can inflate measured errors. No PER is reported without an IPA reference.

The optional `decoder.pronunciation_variants` and `decoder.contractions` settings add audited Italian pronunciation alternatives and LM-attested `l'`/`c'` elisions. Original dictionary pronunciations and language-model weights remain intact. Changes to either setting invalidate the graph cache. See [linguistic conversion](docs/linguistic-conversion.md) for the exact rules.

The repair benchmark reuses original WAV posteriors across all graph and scale comparisons, tunes on the publisher's neutral development split, and freezes its choice before evaluating the neutral test split. The previously inspected acceptance clip is excluded from test. It keeps the original graph, removes only its own temporary ablation graphs, and retains the combined graph:

```bash
.venv/bin/python scripts/benchmark_italian.py \
  --baseline .cache/graphs/ita/13dfce46bb2839d35f441e05af7ee364b5050372ce433a70f04880f7dc95e4d1 \
  --output outputs/ita_repair_study_new
```

Completed rows can be resumed with identical code/configuration. Results include per-recording JSONL, aggregate and per-speaker/per-sentence metrics, graph receipts, frozen selection, and an explicit adoption gate. The dataset repeats 18 acted sentences, so this measures regression performance, not general Italian ASR quality.

## Validation and observed result

```bash
.venv/bin/python -m pytest -q
.venv/bin/python scripts/validate_environment.py
.venv/bin/python scripts/validate_resources.py --strict --config ita

# Fixed neutral test-split selection, publisher reference, then MP4 acceptance.
.venv/bin/python scripts/acceptance_italian.py --run --output outputs/ita_acceptance_new/
```

The original fixed Emozionalmente sample `1612969363951.wav` produced **WER 66.67% (4/6)** and **CER 25.93% (7/27)**. Its artifacts remain unchanged. See the [historical implementation report](docs/implementation-report.md) for those results.

The repaired default was selected on 165 neutral development recordings and passed the adoption gate on 202 held-out neutral recordings: **strict WER 30.69% → 17.47%**, **spoken-form WER 29.57% → 16.35%**, and **spoken-form CER 10.58% → 6.03%**. ZIPA, the WFST topology, the 20,000-word limit, and acoustic/LM scales 1/1 remain unchanged. This corpus repeats 18 acted sentences; the result is a regression benchmark, not a broad Italian accuracy estimate. See the [repair report](docs/italian-repair-report.md).

All tests require the research extra; the tokenizer-specific test skips when its pinned resource is absent. Synthetic graph tests require no checkpoint or network. `scripts/setup.sh` retains the original full acquisition and device-probe workflow; `prepare` is the narrower Italian setup command. MPS remains unsupported on the pinned stack.

## Architecture, provenance, and licensing

- [Technical architecture](docs/architecture.md) and [linguistic conversion](docs/linguistic-conversion.md)
- [Acoustic interface](docs/acoustic-model.md), [dependency pins](docs/dependencies.md), and [resource provenance](docs/resources.md)
- [Historical initialization report](docs/initialization-report.md) and [Sicilian scaffolds](docs/sicilian-resources.md)

The [manifest](resources/manifest.yaml) pins Git revisions or academic release checksums. Acquired resources, graphs, recordings, and run outputs are ignored by Git. Original ARPA and dictionary bytes are preserved. The supplied [CLiC-it paper](docs/93_main_long.pdf), Sections 3.3/4.1, motivates static Compact CTC–lexicon–trigram composition; this implementation uses ZIPA/PhonItalia/MLS and does not claim to reproduce that paper's trained models or experiments.

Project code uses MIT; [adaptation notices](docs/third-party-notices.md) and resource licenses remain separate. ZIPA-CR-NS weights are CC BY-NC-SA 4.0. MLS and Emozionalmente are CC BY 4.0. PhonItaliaR declares CC BY 4.0, while the original corpus announcement specifies noncommercial research use; that discrepancy remains recorded and unresolved.

## Sicilian resource preparation

The resource discovery and corpus preparation phase is documented in
[the Sicilian resource inventory](docs/scn-resource-inventory.md). It includes
source-specific permissions, ignored original downloads, corpus measurements,
DOS24 compatibility, pronunciation requirements and auxiliary n-gram probes.
The [expanded DOS24 audit](docs/scn-dos24-audit.md) separates collection and training
rights from spelling conformity, with current Wikipedia/LiITA measurements,
versioned rules, contextual decisions and explicit unreviewed coverage.
[Reproduction commands](resources/scn/sources/README.md) use the optional `corpus`
extra. Sicilian and bilingual decoding remain disabled; no final Sicilian LM,
G2P or acoustic model was trained.
