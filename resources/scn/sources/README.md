# Sicilian source catalog

`catalog.json` is the reviewed source registry, including metadata-only leads. `search-log.json` records the discovery scope, pinned repository trees and negative findings. The shared `resources/manifest.yaml` registers every catalog entry; SCN entries route acquisition through the separate permission/robots-aware handler.

Catalog schema v2 preserves all 83 original entries and adds independent collection,
research/training, copyright-layer, date and DOS24 assessments. Every assessment
cites evidence and a checked date. `access-audit.json` records bounded landing-page
and robots observations; `expanded-search-log.json` records subsequent searches,
institutional links, failures and limits. Legacy `planned` status means unacquired,
not prohibited or unavailable. Source-level conformity is only an inference;
verified units require hashed contextual decisions.

Original files live in `.cache/scn/acquired/`, with receipts in `.cache/scn/acquisition.json` and robots evidence in `.cache/scn/robots-evidence.json`. No original text, audio, PDF, archive or final LM is distributed in this directory. Each acquired artifact has a measured SHA-256, MD5, byte size, version, URL, final URL and UTC acquisition date. Hashes are local observations, not publisher signatures.

Install the pinned corpus extra without removing the working research environment:

```sh
uv sync --frozen --extra research --extra corpus
.venv/bin/python scripts/scn_resources.py acquire --dry-run
.venv/bin/python scripts/scn_resources.py acquire
.venv/bin/python scripts/scn_resources.py acquire --offline
.venv/bin/python scripts/scn_resources.py extract
.venv/bin/python scripts/scn_resources.py audit
.venv/bin/python scripts/scn_resources.py audit-validate
.venv/bin/python scripts/scn_resources.py audit-report
.venv/bin/python scripts/scn_resources.py analyze
.venv/bin/python scripts/scn_resources.py probe-lm
.venv/bin/python scripts/scn_resources.py inventory-ipa
.venv/bin/python scripts/scn_resources.py sync-manifest
```

`audit` checks every retained text unit, extracts current MediaWiki and LiITA RDF
with source identifiers, replaces the legacy PDF extraction with bounded text
blocks, and applies `resources/scn/audit/reviews.json`. `extract` handles the older
formats; the current XML/RDF sources are delegated to `audit`. Full text and indexes
remain in `.cache/scn/audit/`. Aggregate statistics, rules and review hashes are
tracked. Run `audit-access --refresh` only to repeat network observations. A changed
rule, source, text or audit implementation invalidates the corresponding report or
review rather than silently promoting text.

`acquire` fetches only catalog entries with explicit acquisition permission. It checks robots for the research agent and GPTBot, including redirects, waits between requests and respects retry delays. A denied or unclear source remains metadata-only. The standard downloader delegates these entries to this handler as well. Use `--record` only for a newly reviewed source's first acquisition: it records hashes; it is not permission to refresh mismatched pinned bytes. Existing downloads are verified before use, and a changed upstream response fails closed.

Git and HF file URLs are revision-pinned where available. PDFs, weekly Tatoeba exports and the HF transcript viewer are byte-pinned HTTP snapshots. Their URLs can change content; retaining the original cache enables exact offline replay. Viewer JSON contains expiring audio URL metadata, so fresh downloads may differ even if transcript text is unchanged. It does not contain audio bytes. A fresh acquisition after such changes needs a separately reviewed catalog version; the scripts do not silently bless a mismatch.

Local analysis rights and redistribution rights are separate. Preserve upstream license/author notices and source pointers before sharing any derived corpus, lexicon or LM. Keep MPL, CC BY, CC BY-SA, IGO and GPL/EUPL source partitions distinguishable; this repository's code license does not replace their terms. ND orthography references are not training text or distributable adapted documents.

For current counts, findings and remaining blockers, start with [the resource report](../../../docs/scn-resource-inventory.md). The five research reports describe preparation and future work, not a trained Sicilian pipeline.
