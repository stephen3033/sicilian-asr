# Residual word-error diagnosis

All **1,184 target errors** are accounted for: 805 known-word substitutions, 248 deletions, and 131 insertions. The other 245 OOV-reference errors remain in every complete-recording score.

Baseline: **44.53% WER / 16.72% CER**. Research selection: **44.53% WER / 16.72% CER** (`baseline-lm0.8-b0`). Production remains unchanged.

Exact factor reconstruction passed on all 85 segments (maximum residual 1.1977153e-05). 1471 individual/joint reference counterfactuals were scored on the original frozen frames. Compiled-grammar backoff scores differed slightly from KenLM on 6 segments; the diagnostic decomposition uses the actual grammar.

Each counterfactual holds the surrounding baseline hypothesis fixed and uses existing legal pronunciation paths. Joint adjacent-error corrections are also tested. Attribution is conditional on that context; these scores are not a fully correct transcript, an acoustic accuracy ceiling, or an independent annotation of what was spoken.

## Classification

These are evidence-supported mechanisms or explicit unresolved cases, not independently certified speech-error causes. Alignment ambiguity takes precedence over word-specific attribution; secondary mechanisms remain in the ledger.

| Classification | Errors | Repaired in any trial | Repaired by selection | Tested or required solution |
|---|---:|---:|---:|---|
| acoustic or pronunciation support unresolved | 516 | 314 | 0 | Check independently observed training pronunciations and reference timing; distinguish missing lexical support from ZIPA evidence conflict. |
| ambiguous edit alignment | 420 | 220 | 0 | Inspect the complete adjacent-error group and its audio/caption interval before changing a word-specific rule. |
| lm preference | 200 | 118 | 0 | Repair corpus boundary construction and conditional probabilities; calibrate LM scale and word bonus; retain written distinctions. |
| alignment or segmentation unresolved | 24 | 12 | 0 | Rerun frozen ZIPA across the boundary; compare general segmentation; verify reference wording and timing against audio. |
| mixed or unresolved | 14 | 8 | 0 | Use joint counterfactuals and controlled intervention outcomes; inspect the reference/audio when attribution remains ambiguous. |
| pronunciation cost | 8 | 8 | 0 | Recalibrate pronunciation penalties or add independently recurring channel label paths, preserving fallback paths. |
| mixed lm pronunciation | 2 | 2 | 0 | Compare independent LM and lexical-cost repairs, then combine only demonstrated improvements. |

## Representative score comparisons

Deltas compare the corrected sequence with the decoded sequence on the same frames. Positive values favor the correction. Acoustic ties can still be resolved incorrectly by lexical penalties or LM context. These examples are diagnostic constraints, never proposed training sentences.

| Reference | Hypothesis | Classification | Acoustic delta | Pronunciation delta | LM delta |
|---|---|---|---:|---:|---:|
| dî | di | pronunciation cost | 0.0000 | -0.0499 | 0.0054 |
| cci | mmeci | acoustic or pronunciation support unresolved | -8.7296 | 0.0499 | 5.1280 |
| pi | ppi | lm preference | 0.0000 | 0.0499 | -1.6603 |
| su' | sud | ambiguous edit alignment | -4.7596 | 0.0000 | 1.5822 |
| iḍḍi | ∅ | alignment or segmentation unresolved | 8.0541 | -0.0499 | -8.1262 |
| pussibili | pussìbbili | mixed or unresolved | 0.0000 | 0.0000 | 0.0000 |
| putiti | potti | mixed lm pronunciation | 4.6583 | -0.8171 | -4.6017 |

## Controlled outcomes

Each row below is the lowest-WER candidate in its family, with CER breaking ties. A family winner may still be a tradeoff; research selection applies the joint WER/CER acceptance rule.

| Candidate (best WER per family) | WER | CER | Target errors repaired | New reference-position errors/insertions |
|---|---:|---:|---:|---:|
| baseline-lm0.8-b0 | 44.53% | 16.72% | 0 | 0 |
| converted-original-lm0.6-b0.5 | 52.73% | 19.19% | 218 | 481 |
| channel-o3-support-lm0.8-b0 | 49.36% | 18.62% | 211 | 366 |
| converted-w50-o3-support-lm0.8-b0 | 50.76% | 18.75% | 222 | 422 |
| raw-w50-o3-support-lm0.8-b0 | 51.01% | 18.88% | 217 | 425 |
| channel-boundaries-cues-lm0.8-b0 | 49.33% | 18.61% | 211 | 365 |
| channel-boundaries-joined-lm0.8-b0 | 49.42% | 18.44% | 211 | 368 |
| baseline-normalized-lm0.8-b0 | 44.72% | 16.78% | 5 | 11 |
| channel-context | 44.53% | 16.72% | 0 | 0 |
| pronunciation-cost-x1.5 | 44.94% | 16.62% | 25 | 38 |
| segmentation-30-36 | 45.87% | 16.83% | 95 | 138 |

Converted text outperforms matched raw text under the joint WER/CER rule in 11 of 12 repaired-model settings. These comparisons use the same source passages, real-token weighting, vocabulary, and decoder setting; neither family surpasses the historical baseline.

The construction repair also has matched before/after controls. These quantify an improvement to the newly trained models even when the historical winning baseline remains stronger.

| Original model/scale | Original word/character errors | Repaired word/character errors |
|---|---:|---:|
| channel-o3-lm0.6 | 1740 / 2958 | 1635 / 2683 |
| channel-o3-lm0.8 | 1812 / 3100 | 1584 / 2760 |
| converted-w50-o3-lm0.6 | 1715 / 2880 | 1649 / 2691 |
| converted-w50-o3-lm0.8 | 1775 / 3029 | 1629 / 2780 |
| raw-w50-o3-lm0.6 | 1734 / 2916 | 1654 / 2694 |
| raw-w50-o3-lm0.8 | 1785 / 3058 | 1637 / 2799 |

Repair counts use stable baseline reference positions and insertion gaps. They are outcome associations, not causal attribution; flagged edit-alignment ambiguity remains relevant.

0 compiled-path search failures were demonstrated. 0 recurring channel pronunciation paths satisfying the retained thresholds were missing from the baseline. More permissive paths cannot be justified by forced development alignments.

Joined-waveform diagnostics repaired 124 distinct target positions in at least one probe. These overlapping local probes are not one deployable candidate. The 30/36-second segmentation row is the full-recording control, using newly computed posteriors from the unchanged ZIPA checkpoint.

## Remaining limitations

The channel-only control had 32,515 artificial one-word vocabulary sentences alongside 695 real training cues. The support-floor repair removes those artificial contexts and recomputes normalized backoff weights. The floor mass (0.01) was selected by six training-recording folds. Raw and converted repairs retain exactly matched source passages and repetition counts. The historical baseline unigram mass is 1.049117; normalization is tested separately.

The `w50` repair names identify their original parents. After removing the artificial vocabulary sentences, Wikipedia supplies 65.02% of real tokens in both matched repairs; their real channel/Wikipedia counts remain unchanged. The selected support mass was the highest of the three tested values, not an established optimum.

The fixed vocabulary imposes a 7.63% WER floor from 245 reference tokens it cannot emit. Reaching WER below 10% with that vocabulary would leave room for at most 75 known-word errors, compared with the baseline 1,184. This is a vocabulary-specific bound, not a bound on an expanded system.

The six training recordings provide limited independent pronunciation recurrence and domain context. Where the unpenalized reference paths fit the posteriors worse, these measurements cannot separate missing lexical variants, acoustic-model mismatch, subtitle differences, and timing error without independent phone/word annotation. Acoustically tied written forms require grammatical context. Artificial vocabulary sentences and unnormalized ARPA weights are measured construction properties; their removal is a tested hypothesis, not an assumed accuracy gain.

The selected result remains 34.53 WER percentage points and 11.72 CER percentage points above the 10%/5% targets. Independent acceptance on fresh recordings remains separate from development selection.

## Reproduction and artifacts

Validation passed 226 Python tests, exact Italian and Sicilian production regressions, portable-candidate preparation/transcription, and wheel packaging. Retirement removed 38.49 MiB of superseded diagnostic attempts after verifying their replacement. Current controlled candidate graphs remain available for replay; the earlier repository cleanup is recorded separately.

Run `.venv/bin/python -m pipeline scn-improve --config configs/scn-residual.json --stage all`. Each stage is resumable and checksummed; native workers enforce the existing resource limit. Diagnostic word-trie queries build a small local C++ helper, independently checked against k2 products.

- [error-ledger.jsonl](../../outputs/scn_residual/error-ledger.jsonl)
- [review-queue.jsonl](../../outputs/scn_residual/review-queue.jsonl)
- [counterfactuals.json](../../outputs/scn_residual/counterfactuals.json)
- [calibration.json](../../outputs/scn_residual/calibration.json)
- [interventions.json](../../outputs/scn_residual/interventions.json)
- [lm/models.json](../../outputs/scn_residual/lm/models.json)
- [pronunciation/coverage.json](../../outputs/scn_residual/pronunciation/coverage.json)
- [boundary-outcomes.json](../../outputs/scn_residual/boundary-outcomes.json)
- [research-candidate/candidate.json](../../outputs/scn_residual/research-candidate/candidate.json)
- [experimental-scn.yaml](../../outputs/scn_residual/experimental-scn.yaml)
- [validation.json](../../outputs/scn_residual/validation.json)
- [cleanup.json](../../outputs/scn_residual/cleanup.json)
- [completion.json](../../outputs/scn_residual/completion.json)

Diagnostic corrections are never training evidence. Current production configuration and the historical winning bundle are retained unchanged.
