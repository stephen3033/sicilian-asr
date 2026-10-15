# Usage

## Setup

The lock targets macOS ARM64 and Python 3.12. Install FFmpeg/FFprobe and Apple's Command Line Tools, then run from the repository root:

```sh
uv sync --frozen --extra research --extra corpus
.venv/bin/python scripts/validate_environment.py --acoustic
```

[pyproject.toml](../pyproject.toml) pins direct dependencies; [uv.lock](../uv.lock) pins transitive dependencies and artifacts. The working stack uses PyTorch/torchaudio 2.4.0, NumPy 1.26.4 and the matching ARM64 k2 wheel. Keep that combination together when upgrading. CPU is supported; prior Metal probes failed. Generic PyPI k2 or Intel wheels are not interchangeable with the pinned wheel.

KenLM's Python binding builds with constrained CMake/setuptools/wheel versions. Sicilian training also needs `scripts/setup_scn_tools.py`, which builds pinned KenLM executables and Boost locally. ZIPA/Icefall source archives are verified under `.cache/external`; they are not copied into `src`. Upstream import dependencies remain installed even when unused by this pipeline. OpenFst/OpenGrm executables are unnecessary.

## Commands

```sh
.venv/bin/sicilian-asr prepare --config ita
.venv/bin/sicilian-asr transcribe --config ita \
  --input recording.mp4 --reference subtitles.srt --output outputs/example
```

`prepare` acquires and verifies resources, then compiles the selected graph. `transcribe` uses local resources only. `python -m pipeline` exposes the same commands. `--root` selects a repository checkout; configuration paths inside YAML remain relative to that root. The installed CLI still requires a checkout and its resources.

References may be SRT, VTT or UTF-8 text. Omit `--reference` to transcribe without scoring. `--audio-stream` selects an audio stream. `--export-probabilities` saves original frame probabilities; `--diagnostics` also records sampled competing paths and OOV causes. See the [Italian](ita/README.md) and [Sicilian](scn/README.md) guides for their study commands.

## Configuration

`--config` accepts `ita`, `scn`, `ita_scn`, or a YAML file. Assembly order is shared Python defaults, [acoustic settings](../configs/ipa.yaml), the selected language profile, then user overrides. Mappings merge recursively; lists/scalars replace previous values, including explicit nulls. The bilingual profile remains disabled.

```yaml
languages: [ita]
audio:
  target_seconds: 18
decoder:
  contractions: false
```

A custom file must declare `languages`. Sicilian takes the selected acoustic scale from `.cache/scn/selected.json` before user overrides. Invalid settings, unsupported devices and manifest/path mismatches fail explicitly. Known former built-in configuration and resource paths are translated for compatibility.

## Outputs and failures

Each run needs an empty output directory. It writes transcription, greedy IPA, segments, metadata and metrics; references add parsed cues, normalized text and a comparison. IPA is diagnostic. Timestamps describe segments, not word alignment; competing paths are sampled candidates, not confidence scores or exhaustive N-best results.

Metadata records configuration, versions, hashes, stage timings, frame counts and peak process memory. A Python failure records its active stage and traceback. An interrupted process may retain a `running` receipt; only `complete` means success. Audio extraction holds the whole waveform in memory, even though inference runs segment by segment.

## Development

```sh
.venv/bin/python -m pytest -q
.venv/bin/python scripts/validate_resources.py --strict --config ita
uv build
```

Tests mirror source layers and protect dependency direction. The wheel contains Python code; the source distribution also includes tracked settings, resources and documentation. Downloads, models, graphs and run outputs stay excluded. Changes to pinned native dependencies require fresh environment and acoustic probes; see [validation](validation.md).
