# Common infrastructure

Shared code with no imports from higher pipeline layers.

| Module | Responsibility |
| --- | --- |
| `settings.py` | Global defaults, sample rate, and configuration/manifest locations |
| `utils.py` | Safe repository paths, checksums, atomic JSON writes, subprocess errors |
| `resources.py` | Manifest validation, verified downloads, safe archive extraction |
| `text.py` | Versioned strict text normalization |
| `errors.py` | Shared `ResourceError` |
| `environment.py` | Runtime package versions for provenance |

`DEFAULTS` is a template: configuration loading copies it before applying overrides. Keep language rules in their language directories. See [resource integrity](../../docs/resources.md) for acquisition and migration.
