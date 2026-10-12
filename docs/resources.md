# Resource integrity

## Manifests

Each resource has one owner:

- [IPA manifest](../src/ipa/resources/manifest.yaml): ZIPA weights, tokenizer, and upstream model code.
- [Italian manifest](../src/asr/ita/resources/manifest.yaml): PhonItalia, MLS, and acceptance speech.
- [Sicilian manifest](../src/asr/scn/resources/manifest.yaml): linguistic references and reviewed research sources, including multilingual sources used by Sicilian research.

`common.resources.load_resources(root)` validates and combines them, rejecting duplicate IDs. Available resources retain official URLs, revisions or release identities, checksums, formats, citations, and license evidence. Planned or unavailable entries retain their reasons. Sicilian catalog synchronization updates only its own manifest.

## Acquisition

```sh
.venv/bin/python scripts/download_resources.py --all --dry-run
.venv/bin/python scripts/download_resources.py --all
```

Pass resource IDs for individual downloads. Without `--all`, the default selection excludes the large checkpoint. Sicilian entries use their reviewed acquisition handler; see [its workflow](../src/asr/scn/resources/sources/README.md).

Downloads go to temporary files, with HTTP length and fingerprint checks before atomic installation. Existing bytes are verified and corrupt files are never silently replaced. Source archives use Python's restrictive tar `data` filter and revision markers; downloaded setup code is never executed. Named LM members remain bound to their verified archive through checksummed receipts.

The ZIPA checksum comes from published Git LFS metadata. Git blob identities are checked where available; other hashes record pinned official bytes and are integrity pins, not publisher signatures. Acquisition dates and receipts live in `.cache/resource-status.json` or the Sicilian acquisition cache.

Tracked configuration, manifests, and research metadata are packaged with the code. Acquired language data, weights, archives, and graphs are ignored and excluded from the wheel. Test fixtures never become linguistic resources.

## Migration

After reinstalling the project with `uv sync`, existing checkouts can move resources without downloading them again:

```sh
.venv/bin/python scripts/migrate_layout.py --dry-run
.venv/bin/python scripts/migrate_layout.py --apply
```

The migration verifies all affected files before moving them from the former root resource directories into their language folders. It checks archive members against original archive bytes, preserves old receipts under `.cache/layout-migration/original-receipts`, and rejects conflicting destinations. Repeating it is safe.

Old graph caches remain available as historical evidence. Source fingerprints change during the refactor, so run `prepare --config ita` to build a current graph. Transcription never silently adopts an old graph.

## Licensing and provenance

[Italian provenance](../src/asr/ita/docs/resources.md) records original bytes, corpus counts, MLS training evidence, the unavailable FBK lead, and the unresolved PhonItalia licensing discrepancy. [IPA](../src/ipa/README.md) describes the CC BY-NC-SA model. [Sicilian research](../src/asr/scn/docs/resources.md) retains source-specific permissions, copyright layers, and uncertainty. A source repository's code license does not automatically cover its data or weights.
