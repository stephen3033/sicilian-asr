# Dependency stack

Validated on 2026-10-08: Apple M5, 32 GB unified memory, macOS 27.0.1 ARM64, Python 3.12.12, uv 0.12.23. The committed lock targets macOS ARM64/Python 3.12 only.

| Component | Pin | Role and observed result |
| --- | --- | --- |
| PyTorch / torchaudio | 2.4.0 / 2.4.0 | Native ARM64; ZIPA CPU output passes |
| NumPy | 1.26.4 | Matches the older PyTorch baseline; avoids NumPy 2 ABI assumptions |
| k2 | 1.24.4.dev20260625+cpu.torch2.4.0 | Primary WFST stack; native CPU kernel and PyTorch version checks pass |
| KenLM | 0.3.0 | Native Python extension; controlled ARPA path scores checked against k2 |
| kaldilm | 1.15.4 | Established Kaldi ARPA-to-FST conversion; native Python 3.12 universal2 wheel |
| Lhotse | 1.33.0 | Upstream 80-bin Fbank extraction |
| SentencePiece | 0.2.2 | Checkpoint inventory and tokenizer agree at all 127 IDs |
| SoundFile | 0.14.0 | Audio dependency installed and import-checked |
| ZIPA source | `21e9e8aa4cc6c130cb05d836a1979886f9245302` | Imported from ignored source cache |
| Icefall source | `c1eda5427c46502b2cfdcbeb910bf35f2134a0d4` | Imported from ignored source cache |

The PyTorch baseline follows ZIPA's [reference environment](https://github.com/lingjzhu/zipa/blob/21e9e8aa4cc6c130cb05d836a1979886f9245302/icefall_container.def), with its CUDA-specific packages replaced explicitly by matching official native Mac CPU builds. No CUDA packages are installed. MPS belongs to PyTorch; k2 has no Metal implementation.

The [official k2 wheel](https://huggingface.co/csukuangfj2/k2/blob/566a071486bde284bedcf41a8e3dc92340ec81bc/macos/1.24.4.dev20260625/k2-1.24.4.dev20260625%2Bcpu.torch2.4.0-cp312-cp312-macosx_11_0_arm64.whl) is pinned by immutable repository revision and wheel hash in `uv.lock`. Its SHA-256 is `921a4ec55d44784bb3db9f1f1de79d3cbba72b9ca090e6a2be7cb9a1c28ab5ae`. Do not replace it with the generic PyPI k2 package or an Intel wheel. [k2 installation documentation](https://k2-fsa.github.io/k2/installation/from_wheels.html) requires matching the PyTorch/CUDA build combination.

## Installation mechanics

`uv sync --frozen --extra research` installs the Python dependencies. Direct dependencies are pinned in `pyproject.toml`; transitive dependencies and artifact hashes are frozen in `uv.lock`. KenLM builds in isolation with CMake 3.31.6, setuptools 75.8.2, and wheel 0.45.1 constrained in the project configuration. These build dependencies are separate from runtime dependencies. This installs KenLM's native Python binding, not a separate collection of LM-training command-line executables.

`scripts/download_resources.py` retrieves commit-pinned ZIPA/Icefall source archives into `.cache/downloads/` and extracts them under `.cache/external/`. Validation adds those directories to `sys.path` only within its process. There are no editable copies in `src/`, global `PYTHONPATH` edits, or upstream source patches. Archive hashes and extraction revision markers are checked before importing these sources.

The installed `kaldialign`, `pypinyin`, `tensorboard`, and `wandb` satisfy upstream import-time requirements, even though this project performs no training, Mandarin processing, or experiment uploads. Validation sets `WANDB_MODE=disabled`. Lhotse may initialize its normal `~/.lhotse/tools` directory on import. No full Icefall recipe dependency set is installed.

## Investigated and deferred

| Project | Investigated version | Decision |
| --- | --- | --- |
| [OpenFst](https://www.openfst.org/twiki/bin/view/FST/FstDownload) | 1.8.5 | Not installed; k2 is the selected decoding stack, and no OpenFst graph tools are needed by the smoke tests |
| [OpenGrm NGram](https://www.opengrm.org/twiki/bin/view/GRM/NGramDownload) | 1.3.17, C++17 | Not installed; needs OpenFst and would add an unused LM/WFST toolchain at this stage |

The implementation adds only `kaldilm==1.15.4`; all prior direct/transitive dependency versions and ZIPA/Icefall revisions remain unchanged. kaldilm converts ARPA to OpenFst text, imported by k2 with natural-log scores. KenLM independently validates controlled fixture scores; a KenLM binary is never treated as a k2 graph. FFmpeg and FFprobe are external runtime requirements for media extraction and metadata.

## Licenses and upgrades

k2 and Icefall declare Apache-2.0. ZIPA declares MIT at its root, with Apache-2.0 notices in inherited source files. KenLM is predominantly LGPL-2.1-or-later with file-specific exceptions; see its [license](https://github.com/kpu/kenlm/blob/master/LICENSE). Preserve these notices in the source cache. Model and dataset licenses are recorded separately in the resource manifest.

For an upgrade, change the PyTorch/torchaudio/k2 combination together, select a matching ARM64 wheel, regenerate the lock deliberately, and repeat dependency and CPU/MPS probes. Current Metal failures do not establish that every future version or model will fail. The initial setup does not silently upgrade dependencies to seek a different result.
