# Retained Sicilian experiment history

All figures below use the same strict development-caption scoring. These are
development selections, not independent acceptance results. The detailed ledger
is `outputs/scn_dos24/cleanup/history.json`; original notes are preserved as
`cleanup/prior-experiment-notes.md` and are historical documentation, including
commands for implementations that have since been retired.

| Experiment family | Measured result or finding | Disposition |
| --- | --- | --- |
| Original ZIPA/WFST | 67.15% WER / 29.65% CER | Superseded comparison evidence |
| Training-derived label alternatives and positional costs | Progressed to 47.46% WER / 18.71% CER with optional length labels | Reusable mapping functions retained |
| Recurring channel whole-word paths and cost tuning | 45.59% WER / 16.99% CER; tighter recurrence gave WER/CER tradeoffs | Winning paths preserved in baseline |
| Public exact paths appended to channel paths | 44.94% WER / 16.83% CER | Existing paths retained as a historical control; no new public observations in the DOS24 phase |
| Training-attested vocabulary with tuned added-word costs | 44.53% WER / 16.72% CER | Portable baseline retained |
| Neighboring costs and wider decoder search | 59 required neighboring configurations did not improve the selected baseline | Do not repeat unchanged sweeps |
| Earlier mixed-orthography context LMs | 53.13–56.44% WER | Superseded by matched raw/DOS24 controls |
| Direct-text model/adaptation experiments | Separate abandoned architecture; failures and available metrics retained in the ledger | Code, isolated environments and verified unused model artifacts retired |

Historical failures included OpenMP shared-memory initialization, invalid lattice
cases, and interrupted native workers. Recovery attempts have separate receipts;
failed or partial results are not complete-recording scores. Source identities,
settings, available metrics and failures are retained in the machine ledger.

The DOS24 phase retains acquisition/audit utilities, source datasets and licenses,
ZIPA, production Italian/Sicilian paths, and the exported baseline. The explicit
deletion inventory binds each removed file to its checksum. Retired graph/model
receipts are historical evidence and do not imply their artifacts still exist.

Current-phase comparisons, conditional combination decisions, target gaps and
reproduction evidence are in `outputs/scn_dos24/report.md` and
`outputs/scn_dos24/experiment-ledger.json`. Repeat an experiment only when its
inputs, supported mapping evidence or controlled hypothesis have changed.
