# Sicilian resource preparation

Language code: **`scn`**. Orthographic target: **Cademia Siciliana DOS24**.

This initialization creates locations and provenance placeholders only. No Sicilian reference materials, corpora, vocabularies, pronunciations, or models have been downloaded, scraped, or generated.

| Future resource | Location | Work reserved for a later task |
| --- | --- | --- |
| DOS24 reference materials | `resources/scn/orthography/` | Identify the exact authoritative edition and permitted use; record citation, source, version, and checksum |
| Pronunciation lexicon | `resources/scn/lexicon/` | Develop a reviewed lexicon aligned with DOS24 and an explicit IPA inventory |
| Written vocabulary | `resources/scn/lexicon/` | Curate accepted DOS24 forms separately from pronunciations |
| Trigram LM | `resources/scn/lm/` | Document approved corpus sources, vocabulary and tokenization, then plan model training separately |
| Corpus ingestion | Not implemented | Establish source permissions, attribution, versioning, normalization, and evaluation separation before building ingestion tools |

The manifest has separate IDs for DOS24, lexicon, vocabulary, and LM. Actual artifact paths are null; directories are not evidence that data exists. No empty `.arpa`, lexicon, or vocabulary file stands in for a real resource.

`scn.yaml` references these placeholders. `ita_scn.yaml` references both Italian and Sicilian resources, but does not implement language identification or switching. G2P development is explicitly excluded, including automatic Sicilian pronunciation generation through another library or Italian resource.
