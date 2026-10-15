# Resource integrity and licensing

## Manifests

[IPA](../resources/ipa/manifest.yaml), [Italian](../resources/ita/manifest.yaml) and [Sicilian](../resources/scn/manifest.yaml) each own a manifest. `load_resources` validates and combines them, rejecting duplicate IDs. Available entries retain pinned identities, URLs, checksums and license evidence; unavailable/planned entries retain their reasons.

The Sicilian [catalog](../resources/scn/sources/catalog.json) is authoritative for detailed source assessments. `sync-manifest` copies acquisition fields and a catalog reference, preserving unrelated entries. Detailed assessments are not duplicated in the manifest. [Italian provenance](ita/resources.md) and [Sicilian provenance](scn/resources.md) explain language inputs.

## Acquisition

```sh
.venv/bin/python scripts/download_resources.py --all --dry-run
.venv/bin/python scripts/download_resources.py --all
```

Pass resource IDs for individual downloads. Without `--all`, the default excludes the large checkpoint. Sicilian entries use their reviewed acquisition handler. Downloads use temporary files, HTTP length checks and fingerprints before installation. Existing bytes are verified; corrupt inputs are not silently replaced. Source archives use restrictive extraction and revision markers; acquisition does not execute downloaded setup scripts.

Named archive members remain bound to original bytes through checksummed receipts. Published Git LFS hashes and Git blob identities are checked where available; other hashes pin observed bytes, not publisher signatures. Acquisition receipts stay in `.cache`. Acquired dictionaries/LMs are ignored under `resources`; original upstream documentation is ignored under `docs/ita/references`. Raw source trees and downloaded documents inside `.cache` remain intact.

## Migration

```sh
.venv/bin/python scripts/migrate_layout.py --dry-run
.venv/bin/python scripts/migrate_layout.py --apply
```

After updating/reinstalling the checkout, migration moves verified artifacts from former source-resource directories to the new locations without downloading them again. It checks destinations and archive members before any move, rejects unknown non-code files, updates operational receipts, and preserves their originals in `.cache/code-only-migration`. Repeating it is safe. Resources already at the original root layout are also recognized.

Old graphs and run outputs remain evidence. Code fingerprints deliberately change during refactoring: run `prepare` for a current graph and use fresh study/output directories. Migration does not rewrite frozen study receipts or relax cache checks.

## Licenses and attribution

Code is MIT. ZIPA weights declare CC BY-NC-SA 4.0; source-code licenses do not automatically cover data or weights. PhonItalia's original noncommercial terms conflict with its wrapper's CC BY declaration; that discrepancy remains unresolved. Sicilian collection, analysis, training and redistribution assessments are separate. Preserve source-specific conditions and attribution before sharing derived artifacts.

Code adaptation credits are preserved in [NOTICE](NOTICE), with the full [Apache License 2.0](licenses/Apache-2.0.txt). Both are included in built distributions.
