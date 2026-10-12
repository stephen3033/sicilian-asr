# Next steps for Sicilian ASR

The [monolingual POC](poc.md) now trains an LM, builds a Compact CTC graph and scores the two reserved videos. ZIPA remains fixed. Its provisional pronunciations and single-channel evaluation limit what the measured accuracy establishes.

1. Have a DOS24-competent speaker review the corrected captions and a small gold pronunciation set. Include stress, hiatus, retroflexes, `z`, loans and regional forms.
2. Add independent, verbatim speech references from multiple speakers and regions. Keep speakers/documents separate across training, development and testing.
3. Review additional source passages where development coverage warrants it. Preserve source-specific permissions, spelling decisions and the bounded learning curve.
4. Test pronunciation variants against development acoustic evidence before increasing LM size or adding neural rescoring.
5. Build the combined Italian–Sicilian graph only after both monolingual baselines are stable.

For a later switching graph, retain language-tagged internal word IDs and separate monolingual histories. Switching should consume a word and reset unsupported cross-language history; avoid free epsilon cycles. Tune switch costs on authentic bilingual development speech, with separate monolingual regressions. Italian borrowings, shared spellings and quoted titles need annotation rather than automatic language labels.

The [original first-WFST plan](history/first-wfst-plan.md) preserves the earlier resource assessment and detailed switching proposal. Its statements about unimplemented components describe the repository before this POC.
