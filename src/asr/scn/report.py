"""Render the measured audit and full inventory from checked machine artifacts."""

import json


def write_report(root, catalog, stats):
    def n(value):
        return f"{value:,}"

    sources, total = catalog["sources"], stats["total"]
    access = json.loads(
        (root / "src/asr/scn/resources/sources/access-audit.json").read_text()
    )["sources"]
    lines = [
        "# Sicilian source and DOS24 audit",
        "",
        f"Evidence review: {catalog['checked_on']} (America/New_York; HTTP receipts use UTC). "
        f"All 83 original inventory entries remain, with {len(sources) - 83} additions: **{len(sources)} entries**. "
        "This is a bounded source audit, not an exhaustive census of the web.",
        "",
        "## Measured compatible material",
        "",
        "The figures below describe the complete acquired archive, including both Wikipedia snapshots. "
        "The verified figures are a **reviewed lower bound**, not an estimate of the entire corpus. "
        "Supported/inferred material has positive Cademia authorship or project evidence and no unresolved screening flag; "
        "it has not all received contextual review. Unresolved material is neither an orthographic failure nor an unlicensed corpus.",
        "",
        "| Present DOS24 category | Raw tokens | Exact dedup tokens | Prose tokens | UI tokens | Lexical tokens | Context-reviewed units |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for category, s in stats["categories"].items():
        lines.append(
            f"| {category} | {n(s['tokens'])} | {n(s['deduplicated_tokens'])} | {n(s['prose_tokens'])} | {n(s['localization_tokens'])} | {n(s['lexical_tokens'])} | {n(s['reviewed_records'])} |"
        )
    verified = stats["text_categories"]["verified_conforming"]
    lines += [
        "",
        f"Excluding lexical entries, verified written text contains **{n(verified['tokens'])} raw / {n(verified['deduplicated_tokens'])} deduplicated tokens**. "
        "UI strings remain fragments; they are not counted as sentences. Lexical forms cannot be reported as a prose corpus.",
        "",
        f"All **{n(total['records'])} extracted units / {n(total['tokens'])} tokens** were screened. "
        f"**{n(total['reviewed_records'])} units / {n(total['reviewed_tokens'])} tokens** received contextual decisions; "
        f"**{n(total['unreviewed_records'])} units remain unreviewed**. The targeted sample does not support extrapolating a conformity percentage. "
        "Reviewer: Codex; no external/native-speaker certification is claimed. ",
        "",
        f"The **current cohort**, keeping the 2023 Wikipedia comparison snapshot and all ten RDF reproductions of existing UD documents out of the primary pool, contains "
        f"**{n(stats['current_cohort']['total']['tokens'])} raw / {n(stats['current_cohort']['total']['deduplicated_tokens'])} exact-deduplicated tokens**. "
        "Its category breakdown is preserved separately in the statistics; historical Sicilian composition remains included when it is an independent source.",
        "",
        "The [versioned checklist](../resources/audit/rules.json) cites DOS24 v2.2 PDF pages, keeps mechanical candidates separate from stress, morphology and lexical judgment, "
        "and permits contextual exceptions. The [decisions](../resources/audit/reviews.json) pin source bytes and exact extracted text. "
        "Neither historical date nor absence of a detected pattern verifies spelling. "
        "Examples include legitimate dental `friddu`, the prescribed retroflex spelling `cavaḍḍu`, optional accents in forms of *aviri*, "
        "and foreign names that legitimately retain letters outside the ordinary alphabet. "
        "[DOS24 v2.2](https://cademiasiciliana.org/wp-content/uploads/Cademia-Siciliana-DOS24-v.2.2.pdf).",
        "",
        "## Rights intersection",
        "",
        "Collection, local analysis, corpus preparation, model training, commercial use, derived-data distribution and model distribution have independent fields. "
        "The following intersection includes **conditional** license permissions for research/corpus preparation/training. "
        "It is not an unconditional model-training or model-distribution grant. No models or decoding graphs were built.",
        "",
        "| Category | Research/training raw tokens | Exact dedup | Unconditionally permitted raw tokens |",
        "|---|---:|---:|---:|",
    ]
    for category, s in stats["research_training_intersection"].items():
        unconditional = stats["unconditional_training_intersection"][category]["tokens"]
        lines.append(
            f"| {category} | {n(s['tokens'])} | {n(s['deduplicated_tokens'])} | {n(unconditional)} |"
        )
    lines += [
        "",
        "Conditional grants retain attribution, applicable share-alike/source obligations, noncommercial restrictions and source-specific exceptions. "
        "An unknown grant remains unresolved. A robots exclusion applies to the checked acquisition route; it is not a determination of copyright status. "
        "The manifest carries the same independent assessments as the catalog. Distribution of a particular trained model remains unresolved separately from input reuse.",
        "",
        "LiITA’s lemma bank and glossary declare **CC BY-SA 4.0 within the RDF**, although the GitHub repository has no top-level license declaration. "
        "Its treebank wrapper instead declares **CC BY-NC-SA 4.0**; the pinned upstream UD `LICENSE.txt` declares **CC BY-SA 4.0**. "
        "Both scopes are retained, and the wrapper is not treated as changing the original grant. "
        "[LiITA source files](https://github.com/LiITA-LOD/LocalVarieties/tree/87775784e285ca081385cf372fd2066c5befe25e/Siciliano/rdf).",
        "",
        "FLORES+ declares CC BY-SA 4.0 but also an access-acceptance gate and an evaluation-only policy against training. "
        "The catalog records collection as conditional, conditions unsatisfied, and training as prohibited by that usage policy; no gate was accepted. "
        "[Dataset card](https://huggingface.co/datasets/openlanguagedata/flores_plus).",
        "",
        "## Current Wikipedia and older snapshot",
        "",
        "The current audited dump is **2026-10-01**, obtained from the Your.org mirror listed by Wikimedia after the primary dump endpoint returned HTTP 403. "
        "The mirror reports the articles job complete; original size, MD5 and SHA1 were checked against its published completion metadata, with SHA256 recorded locally. "
        "[Official mirror directory](https://meta.wikimedia.org/wiki/Mirroring_Wikimedia_project_XML_dumps), "
        "[dump completion metadata](https://dumps.wikimedia.your.org/scnwiki/20261001/dumpstatus.json).",
        "",
        f"The dump contains {n(sum(stats['wikipedia']['page_exclusions'].values()))} page records. Exclusions and retained bodies: "
        + ", ".join(
            f"{k}: {n(v)}" for k, v in stats["wikipedia"]["page_exclusions"].items()
        )
        + ".",
        "",
        "Article-body extraction means static prose in namespace 0, excluding redirects, templates, tables, media, references, bibliography, external-link and related-page sections. "
        "It is not a complete rendering with template expansion. Full source XML remains unchanged. Page ID, revision ID, timestamp, URL and raw-wikitext hash are preserved in ignored indexes.",
        "",
        "| Article-body category | Articles |",
        "|---|---:|",
    ]
    for category, count in stats["wikipedia"]["article_categories"].items():
        lines.append(f"| {category} | {n(count)} |")
    mixed = stats["wikipedia"]["verified_passages_in_mixed_articles"]
    lines += [
        "",
        f"Verified passages inside explicitly mixed/nonconforming articles contribute **{n(mixed['tokens'])} tokens in {n(mixed['records'])} passage(s)**. "
        "An article is verified only when every retained body unit has a contextual verified decision. "
        "The Alivu proverb is separately verified while the adjacent paraphrase remains nonconforming; "
        "the Cademia article’s `e'` copula was found by contextual review despite no mechanical flag. "
        "Article-wide nonconformity identifies a counterexample, not the fraction of bad text. "
        "The much larger unresolved count is not a count of nonconforming articles.",
        "",
        "The **2023-11-01** Hugging Face snapshot remains a separate source. Its extraction lacks revision IDs, so no revision identity is invented. "
        "Global exact dedup catches shared passages where extraction boundaries match; it does not reconstruct edit histories or remove all near duplicates.",
        "",
        "## LiITA measurements",
        "",
        "Only `ontolex:writtenRep` literals explicitly tagged `@scn` from Sicilian lemma subjects are lexical forms. "
        "The glossary follows only Sicilian entry-to-lemma links; Italian translations, RDF labels and annotation markup are excluded. "
        "The glossary references the same bank forms and adds **no distinct lexical-form payload** to their union.",
        "",
        "| Dataset | Lemma identities | Written-form occurrences | Variants | Lexical tokens | Sentences | Prose tokens | Dedup tokens | Dedup after UD surface overlap |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in ["scn_liita_lemmabank", "scn_liita_glossary"]:
        s = stats["sources"][name]["measured"]
        lines.append(
            f"| {name} | {n(s['lemmas'])} | {n(s['lexical_forms'])} | {n(s['variant_forms'])} | {n(s['lexical_tokens'])} | {n(s['sentence_units'])} | {n(s['prose_tokens'])} | {n(s['deduplicated_tokens'])} | {n(s['deduplicated_novel_tokens'])} |"
        )
    tree = [
        s["measured"]
        for k, s in stats["sources"].items()
        if k.startswith("scn_liita_stb_")
    ]
    lines += [
        "",
        f"The ten RDF treebank documents contain {n(sum(s['records'] for s in tree))} source sentence records and "
        f"{n(sum(s['tokens'] for s in tree))} lexical tokens in reconstructed surface text. "
        f"{n(sum(s['treebank_overlap_records'] for s in tree))} sentence records match UD after apostrophe-boundary reconciliation. "
        "**All ten documents are reproductions of material already in UD and contribute zero new prose by provenance**, "
        "including records with punctuation or segmentation differences. Sentence IDs, RDF terminal order and surface strings are retained.",
        "",
        "Lexical overlap conservatively removes single-word forms already present in UD surface vocabulary from the “after overlap” column. "
        "This is type overlap, not a claim that every common word was copied from UD. Each remaining form still needs linguistic review. "
        "The RDF’s publisher description says “more than 10.2k” canonical forms; the measured pinned file has the lemma count shown above. "
        "Publisher figures and observed file counts are intentionally not forced to agree. "
        "[LiITA data page](https://www.liita.it/?page_id=69), [Ciallabacialla paper](https://aclanthology.org/2025.clicit-1.103/).",
        "",
        "## Expanded institutional coverage",
        "",
        "| Collection | Actual linguistic scope | Route and unresolved issue |",
        "|---|---|---|",
    ]
    for name in [
        "scn_artesia",
        "scn_vsm",
        "scn_vocabolario",
        "scn_als",
        "scn_parlate",
        "scn_vams",
        "scn_dases",
        "scn_ases",
        "scn_datos",
        "scn_atos",
        "scn_scritture_popolari",
        "scn_galloitalici",
    ]:
        s = sources[name]
        lines.append(
            f"| [{s['name']}]({s['source_url']}) | {s['audit']['language_coverage']['value']} | {s['reason']} |"
        )
    lines += [
        "",
        "ARTESIA’s 737 texts (2021) and the speech archive’s over 4,500 sound documents are publisher-reported figures; no bulk text/token count is claimed for them. "
        "Historical works and modern editions/annotations/database rights are kept distinct. "
        "The popular-writing archive primarily collects **italiano popolare**; its donor/author form downloads are not corpus exports. "
        "ASES and ATOS now have live public consultation sites, superseding older “under construction” descriptions.",
        "",
        "The inventory also retains Cademia educational and commercial publications, UNESCO, localization, literary material, subtitles, transcriptions, web corpora and academic repositories. "
        "Pitrè/Gutenberg is retained for historical Sicilian spans within Italian works. I.Sicily is technically downloadable but concerns ancient epigraphic languages, so DOS24 is not applicable. "
        "Institutional request-a-copy restrictions for ALS/Alcamo/Camilleri publications are recorded as observed access conditions, not universal rights prohibitions.",
        "",
        "## Extraction, duplicate and language limits",
        "",
        f"Measured totals: {n(total['documents'])} source document identities, {n(total['pages'])} represented PDF/wiki pages, "
        f"{n(total['sentence_units'])} heuristic sentence units, {n(total['vocabulary'])} normalized vocabulary types, "
        f"{n(total['exact_duplicate_records'])} exact duplicate records. "
        f"{n(total['possible_italian'])} possible-Italian and {n(total['possible_switching'])} possible-switching units are **review candidates**, not confirmed language labels. "
        "Absence of marker matches does not establish Sicilian.",
        "",
        "NFC, lowercasing and apostrophe equivalence are used only for measurement/dedup keys. Source spelling is preserved. "
        "UNESCO is extracted as independent PDF text blocks with page coordinates, not interleaved page-layout sentences; "
        "original block linebreaks remain available. Short labels/credits are not inferred to be prose. "
        "Remaining printed linebreak hyphens are flagged, not automatically joined. Three PDF pages were visually checked for the cited review decisions; all other blocks remain unreviewed unless explicitly listed.",
        "",
        "Android resource escapes are decoded according to their string syntax; original strings and extraction-input text are retained. "
        "Fourteen repeated legacy localization locator occurrences receive distinct IDs. "
        "A whitespace-bearing LiITA annotation URI is retained as supplied; RDFLib warns about it and no repaired URI is substituted. "
        "A language-specific RDF parser avoids counting its annotation text as Sicilian.",
        "",
        "Exact dedup is performed independently in each source/category/eligibility scope: an unresolved or rights-ineligible copy cannot erase an eligible verified representative. "
        "Consequently source/category dedup counts need not add to global dedup. Raw totals reconcile. "
        "Deduplication is not exhaustive near-duplicate detection; earlier exploratory near-duplicate results remain separate from these exact measurements.",
        "",
        "## Reproduce and validate",
        "",
        "```sh",
        "uv sync --frozen --extra research --extra corpus",
        ".venv/bin/python scripts/scn_resources.py audit-access",
        ".venv/bin/python scripts/scn_resources.py acquire --offline",
        "# For an empty cache, use acquire (without --offline); hashes are already pinned.",
        ".venv/bin/python scripts/scn_resources.py extract",
        ".venv/bin/python scripts/scn_resources.py audit",
        ".venv/bin/python scripts/scn_resources.py audit-validate",
        ".venv/bin/python scripts/scn_resources.py audit-report",
        ".venv/bin/python scripts/scn_resources.py sync-manifest",
        ".venv/bin/python -m pytest tests/asr/scn tests/common",
        "```",
        "",
        "Original downloads, acquisition/robots receipts, full extracted text, screening results, Wikipedia page/article indexes and review queues are in ignored `.cache/scn/`. "
        "The pinned [catalog](../resources/sources/catalog.json), [access observations](../resources/sources/access-audit.json), "
        "[search/follow-up log](../resources/sources/expanded-search-log.json), [rules](../resources/audit/rules.json), "
        "[reviews](../resources/audit/reviews.json), [extraction receipt](../resources/audit/extraction.json) and "
        "[statistics](../resources/audit/statistics.json) back this report. "
        "Access probes are one HEAD per landing URL with robots/redirect checks; failed probes are retained and can be retried with `audit-access --refresh`. "
        "Search queries inspect one returned result batch; repository trees record truncation, and no unbounded search/API pagination is claimed.",
        "",
        "Validation checks evidence and dates, permission enums, unchanged source checksums, review hashes and contextual coverage, Unicode exceptions, "
        "RDF/MediaWiki extraction boundaries, duplicate handling and reconciliation of raw source/category/page/token totals. "
        "The Italian configuration, model/lexicon resources and decoder implementation are outside this change.",
        "",
        "## Prioritized next uses",
        "",
        "1. **Collectable now:** pinned Wikipedia XML, LiITA lexical exports, the existing licensed localization archives, UNESCO originals and UD data. "
        "Use the catalog’s exact routes and licenses; these are already acquired locally. Current Wiktionary’s full dump is an export lead beyond the acquired LiITA/WikiPron subsets.",
        "2. **DOS24-compatible material:** start with the hashed verified units, then the supported/inferred Cademia Firefox/Android/iOS/Common Voice and UNESCO material. "
        "Keep lexical forms separate from prose and keep conditional rights attached.",
        "3. **Further linguistic review:** current Wikipedia’s unreviewed bodies and mixed-article passages, LiITA variants, historical UD witnesses, transcript spelling, and UNESCO blocks with extraction flags. "
        "The screen misses contextual errors and cannot certify this pool.",
        "4. **Unresolved access or rights:** ARTESIA/VSM exports; Vocabolario digitization; ALS/parlate/VAMS/ASES/ATOS bulk permissions; "
        "Cademia course/library/Telegram exports; Arba Sicula and Dieli edition rights; web-corpus underlying licenses; subtitle scope and recording/transcript permissions; "
        "FLORES gate and evaluation policy. Preserve metadata and permitted observations until source-specific questions are resolved.",
        "",
    ]
    (root / "src/asr/scn/docs/dos24-audit.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )

    lines = [
        "# Sicilian resource inventory",
        "",
        f"Expanded audit: {catalog['checked_on']}. **{len(sources)} entries**, including all 83 originals. "
        "See [measured DOS24 results](dos24-audit.md). No Italian decoder changes or Sicilian model training.",
        "",
        "`planned` in the legacy manifest means unacquired metadata, not unavailable or prohibited. "
        "Current acquisition outcomes, rights and conformity are separate fields in catalog schema v2. "
        "Old orthography labels survive only as source-convention/provenance information.",
        "",
        "| Source | Acquisition | Bulk permission | Local analysis / training | Source DOS24 inference | Raw / dedup tokens | Reviewed units |",
        "|---|---|---|---|---|---:|---:|",
    ]
    for name, s in sources.items():
        a, m = s["audit"], stats["sources"][name]["measured"]
        amount = (
            f"{n(m['tokens'])} / {n(m['deduplicated_tokens'])}"
            if m["records"]
            else "not measured"
        )
        rights = a["research_and_training"]
        lines.append(
            f"| [{name}]({s['source_url']}) | {s['acquisition_status']} | {a['bulk_collection']['value']} | {rights['local_analysis']['value']} / {rights['model_training']['value']} | {a['dos24']['value']} | {amount} | {m['reviewed_records']} |"
        )
    lines += [
        "",
        "## Source findings and evidence",
        "",
        "Each entry below has a checked landing/evidence URL; full multi-dimensional evidence, dates, dialect and license scopes are in the machine catalog. "
        "A failed HEAD is an access observation, not an established prohibition. All bodies and form counts are local measurements; metadata-only rows have no claimed corpus size.",
        "",
    ]
    for name, s in sources.items():
        observation = access.get(name, {})
        a = s["audit"]
        lines += [
            f"- **{name}** — {s['name']}. {s['reason']} "
            f"Access check: {observation.get('checked_on', 'unresolved')}, {observation.get('outcome', 'unresolved')}. "
            f"{observation.get('explanation', '')} "
            f"[Source]({s['source_url']}); [license/scope evidence]({s['license_url']})."
        ]
    lines += [
        "",
        "Searches, followed links, failures and pagination bounds: [expanded search log](../resources/sources/expanded-search-log.json). "
        "Per-source/category counts and research/training intersections: [statistics](../resources/audit/statistics.json).",
        "",
    ]
    (root / "src/asr/scn/docs/resources.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    print("Wrote src/asr/scn/docs/dos24-audit.md and src/asr/scn/docs/resources.md")
