# Sicilian DOS24/WFST experiments

The comparison baseline is **44.53% WER / 16.72% CER**. Targets remain WER below
10% and CER below 5%. All three previously evaluated recordings are development
data. Independent acceptance requires fresh recordings.

The single experiment workflow uses a [versioned recipe](../../configs/scn-improve.json):

```sh
.venv/bin/python -m pipeline scn-improve --stage all
```

Individual stages are `export`, `replay`, `cleanup-inventory`, `cleanup`, `convert`,
`label-analysis`, `lm`, `evaluate`, `vocabulary`, `pronunciation`, `combine`, and
`validate`, and `report`. Outputs are in `outputs/scn_dos24`. Preparation and transcription retain
the ordinary `prepare` and `transcribe` commands; `experimental-scn.yaml` selects
the final experimental candidate without changing production defaults.

The baseline bundle preserves graph inputs, learned costs, vocabulary, audio,
posteriors, references and six-recording training observations. Replay rebuilds
its graph and verifies exact complete-recording text and edit counts before
cleanup. `cleanup/deletion-inventory.json` enumerates every retired file;
`cleanup/history.json` retains prior hypotheses, scores and failures. Retired
receipts are historical evidence, not replayable artifact claims.

## Conversion and admission

The converter in `asr.scn.dos24` follows cited DOS24 v2.2 rules, independently of
strict scoring normalization. LiITA and UD provide lemma/POS/morphological
analyses and variant relationships; their canonical labels are not replacement
instructions. Training captions supply additional evidence, never a vocabulary
whitelist. Published consonant exceptions, stress/hiatus, elisions, contractions
and grammatical distinctions have separate conditions. Prior contextual
replacement dictionaries are audited and not imported as universal rules.

Every sentence retains original text, paragraph offsets, article/revision
provenance, source checksums, rule IDs, span edits and unresolved facts. Unresolved
sentences enter `corpus/review-queue.jsonl`. No excluded span is bridged to create
an n-gram. Exact/near duplication and development-passage exclusion are applied
to source and converted text. Coverage and fixture agreement do not certify
real-world conversion accuracy.

`review/priority.jsonl` groups missing facts and ranks candidate spellings by
channel-training frequency for inspection. Ranking never changes admission or
claims that a frequent candidate is correct.

## Controlled comparisons

Fourteen LMs comprise channel-only bigrams/trigrams and converted/raw Wikipedia
mixtures at effective token shares near 5%, 20% and 50%. Matched controls use the
same source passages, vocabulary, OOV handling and shared bounded pruning policy.
Unique evidence counts and effective oversampling counts are reported separately.
The strongest two converted LMs by development perplexity, their raw controls and
the strongest channel LM decode at scales 0.6, 0.8 and 1.0 with fixed word bonus.

Vocabulary expansion is a separate comparison: supported forms from five or more
independent admitted articles plus channel-training forms. Unsupported acoustic
paths are reported. New pronunciation costs use only the six training recordings,
with three-recording recurrence, count and fraction thresholds. Each validation
fold rebuilds mappings without its held-out recording. Existing global/positional
costs and whole-word paths remain controls. Neighboring-word observations are
reported; uncertain syntactic doubling is not invented from text.

Only improvements in controlled comparisons can be combined. Promotion requires
lower WER without higher CER, or lower CER at equal WER. A newly improved graph
permits bounded cost retuning; unchanged historical sweeps are not repeated.
The final selection receives a fresh full-recording replay and portable candidate
bundle. A measured plateau applies to this declared experiment family only.

`validate` checks immutable source spans, conversion edits, passage exclusion,
article support, the fourteen matched LM inputs, fixed vocabularies and pruning,
complete-recording scores, fold membership and portable receipts. `report`
produces `report.md`, `report.json` and a compact `experiment-ledger.json`.
Production regression evidence is recorded in `regression.json`.

Prior experiment families and their disposition are summarized in
[history.md](history.md). Detailed retired receipts remain historical evidence;
only retained bundles are described as replayable.
