# Sicilian language models

`training.py` now estimates the POC's KenLM bigrams and trigrams from explicitly reviewed text. Generated ARPA files and receipts live in `.cache/scn/models`; `.cache/scn/selected.json` identifies the active model. See the [POC study](../../docs/poc.md) for selection and results.

`diagnostics.json` and the [feasibility assessment](../../docs/lm-feasibility.md) preserve the earlier auxiliary count probes. Those measurements predate the trained POC and do not describe its accuracy.
