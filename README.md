# sicilian-asr

Python research scaffolding for Italian (`ita`) and Sicilian (`scn`) ASR:

**Audio → CTC IPA acoustic model → CTC/WFST decoder → orthographic transcription.**

The repository initializes the resources and environment for the requested CLiC-it 2026 architecture. It does **not** implement the transcription pipeline, training, graph composition, G2P, language switching, or corpus ingestion.

ZIPA-CR-NS large and the Italian lexicon are available locally after setup. **The FBK Italian LM is unavailable at its published download link. CPU inference works; ZIPA Metal validation failed on the pinned stack.** See the [initialization report](docs/initialization-report.md).

## Setup

Target: native **macOS ARM64**, Python **3.12.12**, with no CUDA. Install [uv](https://docs.astral.sh/uv/getting-started/installation/) and Apple's Command Line Tools (`xcode-select --install`) first. The compiler is needed for KenLM. Allow several GB of disk space; the checkpoint alone is 1.2 GB.

From the repository root:

```sh
bash scripts/setup.sh
```

This syncs the frozen dependencies, retrieves pinned resources, runs the offline tests, verifies artifact checksums, and probes ZIPA on CPU and MPS. Repeating it reuses verified downloads. It reports known unavailable resources without inventing replacements. MPS probes run in separate processes with CPU fallback disabled; failure does not invalidate the working CPU setup.

The installation is intentionally restricted to this platform. It does not install CUDA, emulate Intel wheels, or modify system Python. The [dependency notes](docs/dependencies.md) explain the selected versions, source installations, and deferred OpenFst/OpenGrm tools.

## Commands

```sh
# Preview acquisition without network access or file writes.
.venv/bin/python scripts/download_resources.py --all --dry-run

# Default acquisition excludes the large checkpoint; --all includes it.
.venv/bin/python scripts/download_resources.py
.venv/bin/python scripts/download_resources.py zipa_checkpoint

# Unit tests need no downloaded data, weights, or network.
.venv/bin/python -m pytest

# Offline accessibility, provenance, dictionary, and vocabulary checks.
.venv/bin/python scripts/validate_resources.py
.venv/bin/python scripts/validate_resources.py --strict --config ita

# Dependency imports; optionally repeat bounded acoustic compatibility tests.
.venv/bin/python scripts/validate_environment.py
.venv/bin/python scripts/validate_environment.py --acoustic
.venv/bin/python scripts/validate_environment.py --require-mps
```

`--strict --config ita` currently exits **1** because the FBK LM is absent. Default resource validation still fails on corrupt artifacts, but reports missing resources as warnings. `--require-mps` exits **1** unless the scripted MPS output passes both CPU comparisons. A process sandbox can hide the GPU; run the acoustic check from an ordinary Terminal process when assessing hardware support.

To install only the lightweight package and offline tests, use `uv sync --frozen`. Restore the complete research environment with `uv sync --frozen --extra research` before resource inventory or acoustic validation.

## Configuration and resources

- `configs/ita.yaml`: Italian lexicon and unavailable FBK LM reference.
- `configs/scn.yaml`: DOS24, vocabulary, lexicon, and trigram LM placeholders.
- `configs/ita_scn.yaml`: references both languages; bilingual output and switching are reserved.

Paths are relative to the repository root. `null` means an artifact or implementation does not exist yet. Acoustic and decoder devices are separate; both default to CPU. Decoder topology, graphs, and phone-inventory mappings are unset. The configuration's backend identifier describes the intended future adapter; there is no backend registry or transcription API yet.

The tracked [resource manifest](resources/manifest.yaml) contains official URLs, immutable revisions, license evidence, and SHA-256 checksums. The downloader stores receipts in `.cache/resource-status.json`. Validation reports and synthetic probe outputs go into `.cache/validation/`. These machine-specific files are ignored by Git.

Original Italian data live under `resources/ita/`; weights and external source archives live under `.cache/`. No third-party repositories are copied into `src/`. Sicilian directories contain documentation only.

Read the [resource provenance](docs/resources.md), [acoustic interface and compatibility notes](docs/acoustic-model.md), and [Sicilian preparation notes](docs/sicilian-resources.md) before adding pipeline code.

## Licensing and academic references

The project code uses the existing MIT license. That license does **not** cover downloaded third-party resources. ZIPA-CR-NS weights are CC BY-NC-SA 4.0. PhonItaliaR declares CC BY 4.0, while the original corpus announcement specifies noncommercial research use; both statements are recorded. FBK LM licensing could not be established. Downloaded data and weights are not included in the Python package or committed to Git.

- Zhu, Samir, Chodroff, and Mortensen (2025), [ZIPA: A family of efficient models for multilingual phone recognition](https://aclanthology.org/2025.acl-long.961/).
- Goslin, Galluzzi, and Romani (2014), [PhonItalia: a phonological lexicon for Italian](https://doi.org/10.3758/s13428-013-0400-8), with [Stefano Coretta's IPA additions](https://github.com/stefanocoretta/phonItaliaR).
- Matassoni and Gretter, [FBK Italian ASR and LM resources](https://github.com/SpeechTechLab/FBK_ASR_public), February 2023.

The exact CLiC-it 2026 proceedings citation remains to be added; this initialization follows the architecture specified for the project and does not claim experimental reproduction.
