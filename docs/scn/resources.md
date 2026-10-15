# Sicilian resources

The [source catalog](../../resources/scn/sources/catalog.json) retains pinned acquisition metadata, source-specific rights, orthographic evidence and all reviewed leads. The [inventory](audits/inventory.md) lists them; the [DOS24 audit](audits/dos24.md) reports measured material. Machine evidence and search logs live beside those reports. Raw text, media, indexes and review queues remain in `.cache/scn`.

## Reproduce the source audit

```sh
.venv/bin/python scripts/scn_resources.py acquire --dry-run
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

For an empty cache, omit `--offline` from acquisition. Downloads require reviewed collection permission and pass robots/redirect checks. `--record` records a newly reviewed source's first acquisition; it does not authorize replacing mismatched bytes. `audit-access --refresh` repeats bounded network observations. Pinned viewer responses and HTTP snapshots may no longer be downloadable byte-for-byte; retain the original cache for replay.

`extract` handles legacy formats; `audit` adds current MediaWiki/RDF identities and bounded PDF text blocks, screens every retained unit, and applies [hashed review decisions](../../resources/scn/audit/reviews.json). A changed source, text, rule or implementation invalidates its evidence. Rule silence and publisher affiliation never independently verify conformity. The historical `analyze`/`probe-lm` commands retain their exploratory role; their cohort labels do not supersede the contextual audit or the [trained POC](README.md).

## Orthography and review

The target is Cademia Siciliana's DOS24 v2.2 (9 April 2024), distinct from its 2017 proposal. Both are reference documents written in Italian, not Sicilian training prose. Their printed NC-ND declaration conflicts with the DOS24 legalcode hyperlink's omission of NC; the catalog preserves this discrepancy.

DOS24 spans varieties rather than prescribing one pronunciation. Grave accents indicate stress/disambiguation; circumflexes indicate contraction. Dental `dd` differs from retroflex `ḍḍ`; blanket replacement corrupts legitimate words. Some *aviri* accents are optional. Lexical apostrophes, regional alternatives and exceptions must survive normalization. Written `e/o` do not establish a universal regional realization; initial intrinsic length and external doubling are not determined by letter doubling alone.

Earlier accent and syllabification conventions need contextual review. NFC/case/apostrophe normalization is distinct from respelling. The POC uses explicitly reviewed replacements, retaining original text and source hashes; it does not modernize whole historical collections. Source date and Cademia affiliation support provenance, not unit-level certification. Native-speaker certification has not been established.

## Pronunciation and ZIPA

Pinned WikiPron provides 1,599 broad rows/1,397 spellings and 1,187 narrow rows/801 spellings; the overlapping counts must not be added as independent vocabulary. These crowdsourced observations lack consistent regional/DOS24 labels. With concatenation and `ɡ→g` only, 1,431 broad and 919 narrow rows round-trip through ZIPA. This tests representation, not acoustic or linguistic accuracy; see [compatibility evidence](audits/evidence/lexicon/ipa-compatibility.json).

The implemented lexicon combines compatible observations with bounded spelling rules. Stress defaults, `z`, hiatus, loans and regional reflexes remain provisional. Keep linguistic IPA separate from its model projection: ZIPA has 127 fixed labels, no stress labels, and cannot represent every diacritic. Unsupported or non-round-tripping projections are rejected. Italian G2P is not a fallback.

Independent validation should cover stressed `/a ɛ i ɔ u/`, native unstressed `/a i u/`, regional vowel outcomes, palatal/retroflex sequences, nasal assimilation, rhotics, initial length and external doubling. Locality-tagged exceptions matter more than expanding variants indiscriminately. The catalog retains Bonner, Privitera, Cipolla, ALS/CSFLS, LFSAG and SoPhISM research leads; unacquired references are not treated as inspected evidence.

## Permissions and evidence

Collection, local analysis, corpus preparation, training and redistribution have independent assessments. `planned` means unacquired metadata, not necessarily prohibited or unavailable. A failed HEAD or robots denial is an access observation, not a copyright ruling. Source repository code licenses do not automatically cover corpora. Preserve MPL, CC, IGO and GPL/EUPL source partitions and attribution; no general permission to distribute channel media, transcripts or derived models is established.

The 8 October feasibility probes preceded contextual DOS24 review and actual KenLM training. Their empty verified cohort is historical, not a current claim that Sicilian training is impossible. They remain available as dated [diagnostics](audits/evidence/lm/diagnostics.json); current behavior and results are documented in the [pipeline guide](README.md).
