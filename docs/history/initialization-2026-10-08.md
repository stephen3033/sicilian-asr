# Initialization report — 2026-10-08

> Historical evidence from 2026-10-08. Commands and paths below describe that checkout; see [current validation](../validation.md) for the layer refactor.

The repository scaffold and native Mac research environment are initialized. No transcription pipeline has been implemented.

| Component | Result |
| --- | --- |
| Python package and configurations | Importable package; `ita`, `scn`, and `ita_scn` YAML scaffolds |
| Dependency stack | Frozen uv lock; native PyTorch 2.4.0, torchaudio 2.4.0, matching ARM64 k2 CPU wheel, and KenLM 0.3.0 installed |
| ZIPA / Icefall sources | Exact source revisions acquired into ignored caches; imports pass without upstream patches |
| Italian lexicon | Original TSV, R data, author-supplied IPA dictionary, and documentation acquired; SHA-256 and available Git blob checks pass |
| ZIPA-CR-NS large 800k | 1,200,537,906-byte averaged checkpoint acquired; publisher SHA-256 matches |
| Acoustic inventory | 127 contiguous labels, blank ID 0; SentencePiece/token table match at every ID |
| Italian LM | **Unavailable:** official FBK Drive folder returns 404; artifact license and exact file format remain unverified |
| Sicilian resources | Documentation placeholders only; no acquisition, generation, or training |
| OpenFst / OpenGrm | Investigated; deliberately not installed alongside the selected k2 stack |

## Validation evidence

- **18 offline tests passed**, covering imports, configuration/manifest consistency, provenance, downloader integrity/atomicity, missing resources, and lexicon validation.
- The frozen environment check passed for all 66 installed packages. The Python wheel built successfully and contains only project code and package metadata; downloaded resources, source caches, and validation reports remain Git-ignored. Local documentation links and setup shell syntax passed checks.
- All 11 acquired manifest artifacts passed offline checksum/accessibility validation. The original table has 120,000 rows and 94,505 distinct written forms; the IPA dictionary has 94,356 rows and 94,290 distinct written forms.
- ZIPA ordinary CPU and upstream TorchScript CPU tests returned finite normalized `[2, 71, 127]` log-probability tensors with valid lengths `[46, 71]`. Maximum CPU-path difference was `2.6702880859375e-05`, within the documented tolerances.
- **MPS did not validate.** Metal is visible outside the process sandbox, but ordinary ZIPA failed at k2's `swoosh_r_forward`; the scripted MPS path returned nonfinite values. CPU fallback was disabled. No custom Metal port was attempted.
- Strict Italian resource-readiness validation exits **1**, correctly naming only `fbk_ita_lm` as missing. Default validation reports that warning without claiming readiness.

Full machine-specific evidence is in `.cache/validation/environment.json`, `.cache/validation/resources.json`, and `.cache/resource-status.json`; those reports and all downloaded artifacts are ignored by Git. They can be regenerated with the README commands.

## Remaining requirements

Obtain the existing FBK LM and establish its license/format; reconcile the pronunciation inventory with ZIPA tokens and CTC topology; develop Sicilian DOS24 resources in a separate task. PhonItalia's original noncommercial-use statement and the wrapper's CC BY 4.0 declaration are both preserved as an unresolved licensing discrepancy.

CPU is the validated baseline. Metal support for this pinned model/environment is unresolved, and neither config names nor dependency availability imply a working decoder. Graph composition, ASR accuracy evaluation, training, code-switching, G2P, and corpus processing remain outside this initialization.

## CPU and Metal evidence

Initialization probes used float32 weights on Apple M5/macOS 27.0.1, PyTorch 2.4.0, and `PYTORCH_ENABLE_MPS_FALLBACK=0`. Device probes ran in separate subprocesses.

| Path | Result |
| --- | --- |
| Ordinary ZIPA CPU | Finite normalized synthetic outputs; now also real speech acceptance passed |
| Ordinary ZIPA MPS | Failed at `k2.swoosh_r_forward`; native activation lacks this device |
| Upstream TorchScript CPU | Passed; max difference from ordinary CPU `2.6702880859375e-05` |
| Upstream TorchScript MPS | Executed but returned nonfinite CTC values |

The scripted probe uses upstream `convert_scaled_to_non_scaled` and `export.EncoderModel`; no replacement kernels or upstream source patches were introduced. MPS remains unsupported and is rejected explicitly. Reproduce historical device probes with `.venv/bin/python scripts/validate_environment.py --acoustic`. The current normal dependency check includes kaldilm and writes `.cache/validation/environment.json`; a dependency-only invocation replaces the older combined report, while per-device JSON/NPZ probe files remain available separately.
