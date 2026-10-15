# DOS24 source audit

This bounded audit measures acquired material and preserves source-specific uncertainty. The [inventory](inventory.md) lists every lead; [reproduction commands](../resources.md#reproduce-the-source-audit) regenerate the measurements below.

<!-- BEGIN GENERATED -->

Source evidence checked: **2026-10-11**. 111 catalog entries; 294,469 screened units / 5,100,005 tokens; 73 context-reviewed units / 543 tokens.

## Measured compatible material

| Category | Raw tokens | Exact dedup | Prose | UI | Lexical | Reviewed units |
| --- | --- | --- | --- | --- | --- | --- |
| verified_conforming | 166 | 153 | 102 | 44 | 20 | 35 |
| supported_inferred | 88,271 | 81,948 | 54,102 | 33,488 | 681 | 1 |
| mixed_nonconforming | 133 | 127 | 121 | 0 | 12 | 16 |
| unresolved | 5,000,731 | 4,319,088 | 4,935,462 | 27,957 | 37,312 | 19 |
| not_applicable | 10,704 | 5,605 | 10,676 | 28 | 0 | 2 |

Verified text excluding lexical entries: **146 raw / 146 deduplicated tokens**. Current cohort excluding the older Wikipedia snapshot and documented RDF reproductions: **2,680,802 raw / 2,599,381 deduplicated tokens**.

## Rights intersection

| Category | Conditional-inclusive training tokens | Exact dedup | Unconditionally permitted tokens |
| --- | --- | --- | --- |
| verified_conforming | 166 | 153 | 0 |
| supported_inferred | 88,271 | 81,948 | 0 |
| mixed_nonconforming | 133 | 127 | 0 |
| unresolved | 5,000,731 | 4,319,088 | 0 |
| not_applicable | 10,704 | 5,605 | 0 |

## Current Wikipedia

| Article category | Articles |
| --- | --- |
| unresolved | 25800 |
| mixed_nonconforming | 3 |
| verified_conforming | 1 |

Verified passages within mixed articles: 1 units / 10 tokens.

## LiITA measurements

| Source | Units | Lemmas | Forms | Variants | Lexical tokens | UD overlap units | Novel dedup tokens |
| --- | --- | --- | --- | --- | --- | --- | --- |
| scn_liita_lemmabank | 14,529 | 9,768 | 14,529 | 4,768 | 14,811 | 1,083 | 13,079 |
| scn_liita_glossary | 14,529 | 9,768 | 14,529 | 4,768 | 14,811 | 1,083 | 13,079 |
| scn_liita_stb_amarasapicatania2024 | 246 | 0 | 0 | 0 | 0 | 221 | 0 |
| scn_liita_stb_colapisciborgetto1904 | 7 | 0 | 0 | 0 | 0 | 7 | 0 |
| scn_liita_stb_colapisciganzirri1904 | 7 | 0 | 0 | 0 | 0 | 7 | 0 |
| scn_liita_stb_colapiscimessina1904 | 23 | 0 | 0 | 0 | 0 | 11 | 0 |
| scn_liita_stb_colapiscimessinaepalermo1904 | 12 | 0 | 0 | 0 | 0 | 10 | 0 |
| scn_liita_stb_colapiscipalermo1904_version1_ | 5 | 0 | 0 | 0 | 0 | 5 | 0 |
| scn_liita_stb_colapiscipalermo1904_version2_ | 32 | 0 | 0 | 0 | 0 | 31 | 0 |
| scn_liita_stb_colapisciroccalumera1904 | 5 | 0 | 0 | 0 | 0 | 5 | 0 |
| scn_liita_stb_colapiscisant_agatadimilitello1904 | 9 | 0 | 0 | 0 | 0 | 9 | 0 |
| scn_liita_stb_colapiscitermini1904 | 5 | 0 | 0 | 0 | 0 | 5 | 0 |

<!-- END GENERATED -->

## Interpretation

Verified figures are a reviewed lower bound, not an estimate of the whole corpus. Supported/inferred material has positive project evidence without unresolved screening flags, but is not all contextually reviewed. Unresolved does not mean nonconforming or unlicensed. UI strings are fragments; lexical forms are not prose. Targeted project reviews do not support population-level conformity percentages or native-speaker certification.

The [checklist](../../../resources/scn/audit/rules.json) cites DOS24 v2.2 pages and separates mechanical candidates from stress, morphology and lexical judgments. [Decisions](../../../resources/scn/audit/reviews.json) bind source bytes and exact text. Legitimate dental `friddu`, retroflex `cavaḍḍu`, optional *aviri* accents and foreign names illustrate why blanket replacements fail. Date, source affiliation and absence of a pattern never independently verify spelling.

Conditional grants retain attribution, share-alike/source obligations, noncommercial restrictions and exceptions. Unknown grants remain unresolved. The measured intersection is not an unconditional training or model-distribution grant. The catalog holds independent collection, analysis, preparation, training and redistribution fields.

## Source-specific findings

The audited current Wikipedia dump is 2026-10-01, obtained from Wikimedia's listed Your.org mirror after the primary endpoint returned 403. Checksums bind those bytes. Main-namespace nonredirect article bodies exclude rendered templates, tables, media captions and reference sections. Entire articles are verified only when every retained unit has a contextual decision; one counterexample establishes mixed/nonconforming, not its percentage of bad text. A verified Alivu proverb can coexist with a nonconforming adjacent paraphrase. Cademia article copula spelling required contextual review despite no mechanical flag.

The 2023-11-01 snapshot remains separate and lacks current revision identities. Global exact dedup catches shared passages only when extraction boundaries match; it does not reconstruct edit history or remove every near duplicate.

LiITA's lemma bank/glossary declare CC BY-SA 4.0 within RDF, despite no repository-root license. Its treebank wrapper declares CC BY-NC-SA 4.0 while pinned upstream UD declares CC BY-SA 4.0; both scopes remain recorded. Only Sicilian lemma literals count as lexical forms. Italian translations, RDF labels and annotations are excluded. The glossary references the same bank forms and adds no distinct form payload to their union. All ten treebank documents reproduce existing UD material and contribute zero new prose by provenance, including punctuation/segmentation differences. Lexical overlap against UD is type overlap, not proof that individual words were copied. Publisher figures are not forced to match measured pinned-file counts.

ARTESIA's 737 texts (2021) and the speech archive's over 4,500 sound documents are publisher figures, not measured downloadable corpus sizes. Historical works, modern editions and database rights are separate. Popular-writing collections primarily contain *italiano popolare*; donor forms are not exports. ASES/ATOS had live consultation sites at the audit date. Request-a-copy restrictions record observed routes, not universal prohibitions. The inventory retains Pitrè/Gutenberg for Sicilian spans within Italian works; I.Sicily concerns ancient epigraphic languages and DOS24 is inapplicable.

FLORES+ declares CC BY-SA but has access acceptance and an evaluation-only policy against training. No gate was accepted. Acquired viewer responses may contain expiring audio URLs; no audio-byte acquisition is implied by transcript metadata.

## Extraction and evidence

NFC, lowercase and apostrophe equivalence affect measurement/dedup keys, while source spelling is preserved. UNESCO uses independent PDF blocks with page coordinates and original linebreaks. Short headings are not inferred to be prose; printed linebreak hyphens are flagged rather than repaired. Three PDF pages were visually checked for the recorded decisions; other blocks remain unreviewed unless listed.

Android escapes follow resource syntax. Repeated localization locators retain distinct physical occurrences. An invalid whitespace-bearing LiITA annotation URI remains as supplied; warnings do not justify inventing replacement data. Language-specific RDF extraction excludes annotation text.

Exact dedup runs independently within source/category/eligibility scopes: an unresolved or ineligible copy cannot erase a verified eligible representative. Therefore dedup subtotals need not add; raw totals do reconcile. Possible-Italian/switching flags are review candidates, not language certification. Heuristic sentence counts are not linguistic gold.

Evidence: [statistics](evidence/audit/statistics.json), [extraction](evidence/audit/extraction.json), [catalog](../../../resources/scn/sources/catalog.json), [access observations](evidence/sources/access-audit.json), and [search log](evidence/sources/expanded-search-log.json). Full text, page/article indexes and review queues stay in `.cache/scn/audit`. Source judgments retain their original evidence dates even when implementation is rerun.
