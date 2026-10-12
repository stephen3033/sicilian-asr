# ZIPA checkpoint and frame-level interface

The selected model is `anyspeech/zipa-large-crctc-ns-800k`: ZIPA-CR-NS large, 299,970,501 parameters, with the final ten checkpoints already averaged. The [official model repository](https://huggingface.co/anyspeech/zipa-large-crctc-ns-800k/tree/861eed23d6dbffd72c80e3d3fb12caefc0bf2f1a) and all upstream sources remain pinned at the initialization revisions. The weights use CC BY-NC-SA 4.0; ZIPA's source-code license does not replace the model license.

The checkpoint is a PyTorch state dictionary, not a Transformers `AutoModelForCTC` package. `ZipaAcoustic` builds the matching upstream `large_params` architecture, verifies the checkpoint hash, loads it with `weights_only=True`, `map_location="cpu"`, and `strict=True`, then uses float32 evaluation mode without gradients. Four CPU threads are the default.

## Frame access and preprocessing

The implemented `AcousticOutput` contract contains ordered tokens, blank ID, preprocessing metadata, `[batch, frames, vocabulary]` natural-log probabilities, and valid frame lengths. It validates finite normalized probabilities, float32 CPU placement, inventory dimensions, and positive integer lengths bounded by the frame tensor.

```python
with torch.inference_mode():
    encoder_out, frame_lengths = model.forward_encoder(features, feature_lengths)
    log_probs = model.ctc_output(encoder_out)
```

The [upstream CTC head](https://github.com/lingjzhu/zipa/blob/21e9e8aa4cc6c130cb05d836a1979886f9245302/zipformer_crctc/model.py) ends in `LogSoftmax`. No second normalization is applied. Feature extraction uses upstream Lhotse `Fbank()` defaults (80 bins) on 16 kHz mono waveforms. The serialized Fbank configuration is included in run metadata. No wav2vec2 waveform normalization is introduced.

Valid lengths come from ZIPA, never from assumed downsampling. Initial synthetic 1.0/1.5 s inputs yielded shape `[2, 71, 127]` and lengths `[46, 71]`. The real acceptance recording yielded 287 valid acoustic frames. Compact CTC expansion to 574 frames occurs only in the decoder adapter; optional probability exports retain the original valid frames.

`greedy_ctc` separately merges consecutive repeated IDs and removes blanks. It is a diagnostic IPA output, never a replacement for WFST words. Every returned token keeps its pinned ID: blank 0, `<sos/eos>` 1, `<unk>` 2, `▁` 3, and remaining IPA symbols through 126. The token table and SentencePiece model are checked at every ID.

Phone normalization, stress/length/affricate policies, and the utterance-prefix convention are specified in [linguistic-conversion.md](linguistic-conversion.md). Static topology/composition and failure handling are specified in [architecture.md](architecture.md).

## CPU and Metal evidence

Initialization probes used float32 weights on Apple M5/macOS 27.0.1, PyTorch 2.4.0, and `PYTORCH_ENABLE_MPS_FALLBACK=0`. Device probes ran in separate subprocesses.

| Path | Result |
| --- | --- |
| Ordinary ZIPA CPU | Finite normalized synthetic outputs; now also real speech acceptance passed |
| Ordinary ZIPA MPS | Failed at `k2.swoosh_r_forward`; native activation lacks this device |
| Upstream TorchScript CPU | Passed; max difference from ordinary CPU `2.6702880859375e-05` |
| Upstream TorchScript MPS | Executed but returned nonfinite CTC values |

The scripted probe uses upstream `convert_scaled_to_non_scaled` and `export.EncoderModel`; no replacement kernels or upstream source patches were introduced. MPS remains unsupported and is rejected explicitly. Reproduce historical device probes with `.venv/bin/python scripts/validate_environment.py --acoustic`. The current normal dependency check includes kaldilm and writes `.cache/validation/environment.json`; a dependency-only invocation replaces the older combined report, while per-device JSON/NPZ probe files remain available separately.
