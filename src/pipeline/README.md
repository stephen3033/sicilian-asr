# Pipeline orchestration

`cli.py` exposes preparation and transcription. `transcribe.py` verifies resources and the graph, extracts and segments audio, loads ZIPA once, decodes each segment, writes artifacts, and optionally evaluates an independent reference.

A run writes running metadata before inference and completes only after its outputs are written. Python failures retain the active stage and traceback. Abrupt termination may leave a running receipt; it is not a completed run. Nonempty output directories are rejected.

Metadata includes configuration, versions, resource/graph/output hashes, stage timings, frame lengths, lattice sizes, audio duration, and peak RSS. Full-file audio extraction holds the waveform in memory; very long inputs therefore add audio-buffer memory even though inference runs segment by segment.

## Configuration

`load_config(name_or_path, root, manifest)` assembles:

1. Shared defaults from `common/settings.py`.
2. Model settings from `ipa/config.yaml`.
3. The selected Italian, Sicilian, or disabled bilingual profile.
4. Explicit values from a user YAML file, when supplied.

Mappings merge recursively. Lists and scalars replace prior values; explicit nulls are retained. The resolved result is validated and recorded in artifacts. A custom file must declare `languages`, which selects its underlying profile:

```yaml
languages: [ita]
audio:
  target_seconds: 18
decoder:
  contractions: false
```

Paths inside configuration remain repository-relative. Both the profile name `ita` and the path `src/asr/ita/config.yaml` work. Sicilian loads the model receipt and acoustic scale from `.cache/scn/selected.json`, with explicit user YAML values taking precedence. The bilingual profile remains disabled.

## Supporting commands

`acquire.py`, `validation.py`, and the validation CLIs coordinate resource checks. `benchmark.py` runs the Italian study with frozen settings and isolated workers. `migration.py` verifies and moves resources from the former layout. The corresponding files in `scripts/` only invoke these commands.

`scn_poc.py` runs acquisition, reviewed corpus preparation, training, development selection and final evaluation. `scn_study.py` caches acoustic outputs independently of LM candidates. See the [Sicilian workflow](../asr/scn/README.md).
