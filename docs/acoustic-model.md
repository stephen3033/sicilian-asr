# ZIPA checkpoint and frame-level interface

The selected model is `anyspeech/zipa-large-crctc-ns-800k`: ZIPA-CR-NS large, 299,970,501 loaded parameters, with the final ten checkpoints already averaged. The [official model repository](https://huggingface.co/anyspeech/zipa-large-crctc-ns-800k/tree/861eed23d6dbffd72c80e3d3fb12caefc0bf2f1a) is revision-pinned in the manifest. Only the averaged `.pth`, token table, model card, and matching SentencePiece tokenizer are acquired. Raw training checkpoints, other model variants, training data, and ONNX weights are not downloaded.

The checkpoint is a PyTorch state dictionary, not a Hugging Face Transformers `AutoModelForCTC` package. The validation script builds the matching upstream architecture and loads the verified weights with `weights_only=True`, `map_location="cpu"`, and strict state-dictionary matching. The NS weights use CC BY-NC-SA 4.0; ZIPA's root source-code license does not replace that model license.

## Frame access

Upstream [`zipa_ctc_inference.py`](https://github.com/lingjzhu/zipa/blob/21e9e8aa4cc6c130cb05d836a1979886f9245302/zipa_ctc_inference.py) performs greedy decoding after the acoustic head. For future WFST work, obtain the head output **before** that greedy collapse:

```python
# model is the upstream AsrModel, in eval mode.
with torch.inference_mode():
    encoder_out, frame_lengths = model.forward_encoder(features, feature_lengths)
    log_probs = model.ctc_output(encoder_out)  # [batch, frames, 127]
```

The [CTC head](https://github.com/lingjzhu/zipa/blob/21e9e8aa4cc6c130cb05d836a1979886f9245302/zipformer_crctc/model.py) ends with `LogSoftmax`, so this returns natural-log probabilities, not raw logits. If an experiment requires raw logits, access the head before its final LogSoftmax (`model.ctc_output[:-1](encoder_out)`) in evaluation mode. Avoid applying another normalization unnecessarily.

Preprocessing is 16 kHz mono audio with Lhotse's upstream 80-bin Fbank configuration. Carry the returned frame lengths; never infer them from padded batch dimensions or assume a fixed temporal reduction from a configuration label. Synthetic 1.0 s and 1.5 s inputs produced output shape `[2, 71, 127]` with valid lengths `[46, 71]` in this environment.

The future swappable acoustic adapter contract is: ordered token inventory, blank ID, sample-rate/feature metadata, `[batch, frames, vocabulary]` log probabilities, and valid frame lengths. Other CTC IPA models will supply their own identifiers and inventories. No adapter registry or general-purpose inference interface is implemented yet.

## Token inventory and unresolved decoder compatibility

The checkpoint token table and SentencePiece model match at all 127 IDs. IDs 0–3 are `<blk>`, `<sos/eos>`, `<unk>`, and the SentencePiece boundary symbol `▁`. Use the provided order; do not derive acoustic IDs by sorting IPA characters or assume one Unicode character equals one phone.

Before decoding can be implemented, resolve and validate:

- PhonItalia's tokenized affricates, long consonants, voicing/assimilation conventions, and orthographic accent normalization against ZIPA tokenization.
- Stress, diacritics, Unicode normalization, and word boundaries without silently deleting phonetic information.
- The difference between CTC blank, FST epsilon, boundary symbols, and lexicon disambiguation symbols; repeated-label/blank transitions and the appropriate k2 CTC topology.
- Italian LM vocabulary coverage, case/apostrophe handling, OOV policy, and the conversion of the acquired LM format into a decoding graph.
- Separate Sicilian inventory and DOS24 decisions after resources exist.

These decisions remain explicitly unresolved in YAML. No lexicon rewriting, CTC graph construction, WFST composition, or language switching is performed.

## Metal findings

Tests ran with float32 weights on Apple M5/macOS 27.0.1, PyTorch 2.4.0, and `PYTORCH_ENABLE_MPS_FALLBACK=0`. Feature extraction is on CPU. Device probes are separate subprocesses, protecting the report from native failures.

| Path | Result |
| --- | --- |
| Ordinary ZIPA on CPU | Finite, normalized frame outputs; passed |
| Ordinary ZIPA on MPS | Failed at `k2.swoosh_r_forward`; k2's activation kernel does not support this device |
| Upstream TorchScript encoder on CPU | Passed; maximum absolute difference from ordinary CPU was `2.6702880859375e-05` |
| Upstream TorchScript encoder on MPS | Executed but returned nonfinite CTC values; failed |

The scripted probe uses upstream `convert_scaled_to_non_scaled` and `export.EncoderModel`, which route activations through ordinary PyTorch operations. No custom replacement kernels or source patches were introduced. CPU comparisons use valid frames only and `rtol=1e-3`, `atol=1e-3`. These tests establish device/numerical compatibility on a synthetic input, not recognition accuracy, speed, or all-input equivalence.

MPS is **not validated** on the pinned stack. CPU remains the default. k2 decoding is separately configured for CPU regardless of any future acoustic MPS support. Reproduce the result with `.venv/bin/python scripts/validate_environment.py --require-mps`; the expected current exit code is 1. Full local traces are in `.cache/validation/environment.json`.
