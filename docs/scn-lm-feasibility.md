# Sicilian n-gram LM feasibility

**A DOS24 final language model cannot yet be selected empirically:** the verified DOS24 cohort is empty. Once a reviewed vocabulary and small licensed corpus exist, use a unigram as a composition smoke baseline, then test a bigram. The acquired auxiliary cohorts support that staged WFST approach; they do not justify relabeling older text as DOS24 or choosing a neural LM by default.

## What was measured

[diagnostics.json](../resources/scn/lm/diagnostics.json) contains temporary interpolated Witten–Bell count probes for orders 1–4. No final ARPA, KenLM binary, decoding graph or neural model was produced. The existing Italian pipeline consumes a **published pretrained trigram**; its ARPA module restricts vocabulary and weights, rather than estimating an LM. These diagnostic counts are a separate feasibility tool, not a replacement for that Italian pipeline.

The vocabulary uses training words occurring at least twice, plus UNK and EOS. The base distribution is uniform over that vocabulary; each history interpolates its counts with the lower-order distribution using the number of observed continuation types. Sentence starts initialize history; EOS is scored. Unseen histories back off. Tests check that distributions normalize and records do not form cross-sentence n-grams.

Documents linked by sufficiently long exact or detected near duplicates share a split. A deterministic hash assigns groups to train/development/test with nominal 80/10/10 buckets. This is a document-group split, not a guaranteed token ratio; large connected groups can skew it. Vocabulary selection uses training only. Choose the **lowest order within 1% of the best development perplexity**, then evaluate that order on test. Short duplicate labels are removed without merging whole documents. Approximate near-duplicate detection cannot guarantee removal of every related passage.

The probes exclude lexical/evaluation roles, exact duplicates, unresolved PDF layout, Omnilingual transcripts pending review, language-screen flags, detected HTML/wiki markup or URLs, and records longer than 250 tokens. Filtering is not proof of language purity. UNESCO text remains measured in the descriptive inventory but contributes zero probe records. Unreviewed speech transcripts remain outside LM probes; official dev/test rows stay reserved.

## Auxiliary results

| Cohort | Train tokens | Development tokens | Test tokens | Train vocabulary, count ≥2 | Dev OOV | Selected order | Selected-order test perplexity |
|---|---:|---:|---:|---:|---:|---:|---:|
| Other Sicilian (mostly 2023 Wikipedia) | 1,804,140 | 257,503 | 194,631 | 73,621 | 10.66% | 3 | 443.28 |
| Uncertain (mostly localization + SmolDoc) | 57,435 | 2,696 | 5,832 | 3,260 | 20.33% | 2 | 134.35 |

| Order | Other-Sicilian dev perplexity | Dev unseen n-grams | Uncertain dev perplexity | Dev unseen n-grams |
|---|---:|---:|---:|---:|
| 1 | 1188.34 | 0.00% | 210.88 | 0.00% |
| 2 | 413.46 | 25.84% | 110.88 | 24.32% |
| 3 | 372.84 | 59.24% | 119.20 | 57.65% |
| 4 | 404.37 | 75.47% | 136.93 | 76.55% |

Unseen rates above are measured **after UNK mapping**, with sentence boundaries. Zero unseen unigrams therefore does not mean zero OOV. Perplexities across cohorts with different vocabularies are not directly comparable. A high UNK rate can make perplexity deceptively low. Test perplexity was not used to select order.

| Order | Other-Sicilian train n-gram types | Singleton fraction | Uncertain train n-gram types | Singleton fraction |
|---|---:|---:|---:|---:|
| 1 | 73,623 | 0.00% | 3,262 | 0.00% |
| 2 | 762,001 | 74.11% | 24,577 | 63.52% |
| 3 | 1,361,447 | 88.56% | 41,982 | 80.95% |
| 4 | 1,586,551 | 94.09% | 48,679 | 87.76% |

Unigram singleton counts are zero by construction after the minimum-count vocabulary mapping; raw word-type and singleton counts are retained separately in the JSON. Higher-order sparsity is severe. Four-gram development perplexity is worse than the selected lower order in both cohorts. The current graph compiler also fixes its maximum order at three; order generalization belongs to a future isolated change.

## Vocabulary and pronunciation coverage

| Top training words | Other-Sicilian dev token coverage | Uncertain dev token coverage |
|---|---:|---:|
| 500 (actual uncertain vocabulary capped at 500) | 51.87% | 62.69% |
| 1000 (actual uncertain vocabulary capped at 1,000) | 57.79% | 69.96% |
| 5000 (actual uncertain vocabulary capped at 5,000) | 71.49% | 80.82% |
| 10000 (actual uncertain vocabulary capped at 7,092) | 77.01% | 85.16% |
| 25000 (actual uncertain vocabulary capped at 7,092) | 83.67% | 85.16% |
| 50000 (actual uncertain vocabulary capped at 7,092) | 87.73% | 85.16% |

These are surface-form token estimates on auxiliary text, including retained numbers and names. They are not predicted DOS24 speech coverage. The 1,488 normalized spelling candidates in the union of the acquired WikiPron tables cover only **28.30%** of other-Sicilian development tokens and **32.60%** of uncertain-cohort development tokens. No correctness, DOS24 conformity or regional equivalence is implied by a spelling match. The gold DOS24 pronunciation lexicon has not been created, so usable lexicon–LM overlap still requires validation.

Two predeclared source holdouts remove groups linked to the held source, train on the remaining uncertain-cohort training groups, and evaluate the held source. Holding out SmolDoc gives **37.16% OOV**; holding out Firefox localization gives **17.76% OOV**. Both again favor bigrams by development-style perplexity comparison. These checks reveal domain mismatch and are not an independent final test of a production model. The smaller cohort has few development documents; changing the licensed, reviewed source mix may change the order decision.

## Feasibility by order

| Order | Assessment for the next DOS24 baseline |
|---|---|
| Unigram | Mechanically feasible once a reviewed word list with frequencies exists. Useful for composition and lexicon checks; current data do not measure DOS24 quality. |
| Bigram | Best initial contextual candidate after certification: the smaller auxiliary corpus selects it, but dev OOV remains about 20%. Re-measure on DOS24 before final estimation. |
| Trigram | Supported by the larger older-orthography auxiliary cohort, with high sparsity and about 10.7% dev OOV. This is evidence of scale dependence, not permission to use Wikipedia as DOS24. |
| Four-gram | Not recommended initially: worse dev perplexity in both probes, many singleton/unseen contexts, additional graph cost and a compiler order constraint. |

Final training should use a licensed reviewed cohort, train-only vocabulary, independently validated pronunciations, documented smoothing and pruning, and separate development/test data. The first decoder should report vocabulary coverage and reference OOV alongside WER/CER; errors on unsupported words cannot be repaired merely by raising LM order. Avoid train/test leakage through source translations, repeated stories, UI templates, speakers and prompts. When speech transcripts become eligible, preserve speaker and prompt grouping as well as the release’s original partitions.

## Alternatives

A character LM can smooth inflectional or spelling gaps and may later support an explicitly designed OOV path. It does not by itself map ZIPA IPA to DOS24 words: a pronunciation/transduction component is still required. Evaluate it against word n-grams after a stable standard and reference set exist.

A multilingual pretrained neural model could eventually rescore lattices, but its actual Sicilian/DOS24 competence, tokenizer coverage, licensing, tendency to rewrite regional wording and inference cost need measurement. A translation model is not evidence of a well-calibrated Sicilian speech LM. Training a small neural LM on this heterogeneous corpus may learn the same orthographic and domain errors with less auditability. Neither option removes the need for rights clearance, gold pronunciations or held-out speech. Keep the n-gram WFST baseline primary, and add neural rescoring only if controlled tests demonstrate a material gain.
