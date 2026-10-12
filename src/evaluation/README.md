# Evaluation

`metrics.py` compares normalized hypothesis/reference text and returns metrics plus a readable alignment. `edits` records substitutions, deletions, insertions, correct units, and reference denominators. WER uses words; CER excludes spaces. Empty references have undefined rates.

Strict normalization is versioned in [common/text.py](../common/text.py): NFC, lowercase, standardized apostrophes, punctuation removal, and collapsed whitespace. Accents, digits, internal apostrophes, and `<unk>` remain visible and scored.

`evaluate` optionally accepts a spoken-form normalizer and its rule version. [Italian evaluation](../asr/ita/evaluation.py) supplies conservative integer verbalization; the shared evaluator does not import Italian rules. Strict and spoken-form scores remain separate.

Reports distinguish words absent from the supported lexicon from words outside the selected decoding vocabulary. `<unk>` cannot identify arbitrary unseen speech. Subtitle omissions, paraphrases, and timing mismatch can inflate errors; no PER is reported without an IPA reference.
