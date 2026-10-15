"""Reproducible DOS24 screening and evidence-bound review, separate from training.

All full text and record indexes stay in .cache/scn/audit. Committed artifacts
contain rules, source evidence, hashed review decisions and aggregate statistics.
"""

import bz2
import hashlib
import json
import re
import unicodedata
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

from asr.scn.acquisition import load_catalog
from asr.scn.corpus import language_flags, sentences, tokens
from common.errors import ResourceError
from common.resources import verify_file
from common.settings import SCN_CATALOG, SCN_EVIDENCE, SCN_RESOURCES
from common.utils import fingerprints, local_path, write_json

VERSION = "scn-dos24-audit-v1"
DATA = SCN_RESOURCES / "audit"
EVIDENCE = SCN_EVIDENCE / "audit"
CACHE = ".cache/scn/audit"
CATEGORIES = (
    "verified_conforming",
    "supported_inferred",
    "mixed_nonconforming",
    "unresolved",
    "not_applicable",
)
PERMISSIONS = {"permitted", "conditional", "prohibited", "unresolved"}


def digest(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def android_unescape(text):
    """Decode Android resource escapes, preserving unknown literal sequences."""
    table = {
        "n": "\n",
        "r": "\r",
        "t": "\t",
        "'": "'",
        '"': '"',
        "\\": "\\",
        "@": "@",
        "?": "?",
    }
    return re.sub(
        r"\\(u[0-9a-fA-F]{4}|[nrt'\"\\@?])",
        lambda m: chr(int(m[1][1:], 16)) if m[1].startswith("u") else table[m[1]],
        text,
    )


def read_json(root, name):
    return json.loads((root / name).read_text(encoding="utf-8"))


def screen(text, rules):
    normalized = unicodedata.normalize("NFC", text)
    return [
        {
            "rule": rule["id"],
            "matches": sorted(
                set(m.group() for m in re.finditer(rule["pattern"], normalized))
            ),
        }
        for rule in rules["rules"]
        if rule.get("pattern") and re.search(rule["pattern"], normalized)
    ]


def wiki_body(wikitext):
    """Extract static body prose; never join prose over deleted template content.

    No template expansion is performed. Tables, media, references, headings,
    categories and interlanguage links are outside this defined body measure.
    """
    import mwparserfromhell as mw

    code = mw.parse(wikitext)
    removed = Counter()
    for section in list(code.get_sections(include_lead=False, flat=False)):
        headings = section.filter_headings()
        if headings and re.match(
            r"(?i)^(bibliugraf|bibliograf|rifirim|riferim|not[ei]|nutanni|fonti|lij?ami|liami|vidi puru|talìa puru|vuci currilati|àutri pruggetti)",
            str(headings[0].title).strip(),
        ):
            try:
                code.replace(section, "\n\n")
                removed["reference_sections"] += 1
            except ValueError:
                pass
    for template in code.filter_templates(recursive=False):
        code.replace(template, "\n\n")
        removed["templates"] += 1
    for tag in list(code.filter_tags(recursive=True)):
        if str(tag.tag).lower() in {
            "ref",
            "references",
            "table",
            "gallery",
            "math",
            "score",
            "timeline",
            "syntaxhighlight",
            "source",
        }:
            try:
                code.replace(tag, "\n\n")
                removed["nonprose_tags"] += 1
            except ValueError:  # Already removed with its enclosing element.
                pass
    for link in list(code.filter_wikilinks(recursive=True)):
        title = str(link.title).strip()
        # Colon namespaces and language links are metadata, not article sentences.
        if ":" in title and not title.startswith(":"):
            try:
                code.replace(link, "\n\n")
                removed["namespace_links"] += 1
            except ValueError:
                pass
    for heading in code.filter_headings():
        code.replace(heading, "\n\n")
        removed["headings"] += 1
    plain = code.strip_code(normalize=True, collapse=True)
    plain = re.sub(r"(?m)^[*#;:]+\s*", "", plain)
    return [p.strip() for p in re.split(r"\n\s*\n", plain) if tokens(p)], dict(removed)


def wikipedia_records(path, pages):
    with bz2.open(path, "rb") as stream:
        iterator = ET.iterparse(stream, events=("start", "end"))
        _, root = next(iterator)
        ns = root.tag.split("}")[0] + "}"
        for event, page in iterator:
            if event != "end" or page.tag != ns + "page":
                continue
            get = lambda node, tag: node.findtext(ns + tag)
            page_id, title = get(page, "id"), get(page, "title")
            revision = page.find(ns + "revision")
            meta = {
                "page_id": page_id,
                "title": title,
                "namespace": get(page, "ns"),
                "revision_id": get(revision, "id") if revision is not None else None,
                "revision_timestamp": get(revision, "timestamp")
                if revision is not None
                else None,
            }
            meta["url"] = (
                f"https://scn.wikipedia.org/w/index.php?oldid={meta['revision_id']}"
            )
            if get(page, "ns") != "0":
                meta["exclusion"] = "non_article_namespace"
            elif page.find(ns + "redirect") is not None:
                meta["exclusion"] = "redirect"
            else:
                raw = get(revision, "text") or ""
                parts, removed = wiki_body(raw)
                meta.update(
                    wikitext_sha256=digest(raw),
                    removed=removed,
                    body_records=len(parts),
                    exclusion=None if parts else "no_static_body",
                )
                for i, text in enumerate(parts):
                    yield page_id, str(i), text, "paragraph", dict(meta)
            pages.append(meta)
            page.clear()
            root.clear()


def liita_forms(path):
    """Exactly ontolex:writtenRep @scn, with lemma identity and canonical label."""
    from rdflib import RDF, RDFS, Graph, Literal, Namespace

    graph = Graph().parse(path, format="turtle")
    ontolex = Namespace("http://www.w3.org/ns/lemon/ontolex#")
    lila = Namespace("http://lila-erc.eu/ontologies/lila/")
    for subject in sorted(set(graph.subjects(RDF.type, lila.Lemma)), key=str):
        label = str(graph.value(subject, RDFS.label) or "")
        for form in sorted(set(graph.objects(subject, ontolex.writtenRep)), key=str):
            if not isinstance(form, Literal) or form.language != "scn":
                continue
            yield (
                str(subject),
                str(form),
                {
                    "lemma_uri": str(subject),
                    "canonical_label": label,
                    "form_kind": "canonical" if str(form) == label else "variant",
                    "lemma_variant_uris": sorted(
                        str(o) for o in graph.objects(subject, lila.lemmaVariant)
                    ),
                },
            )


def liita_glossary(path, bank):
    from rdflib import Graph, Namespace

    graph = Graph().parse(path, format="turtle")
    ontolex = Namespace("http://www.w3.org/ns/lemon/ontolex#")
    for subject, lemma in sorted(
        graph.subject_objects(ontolex.canonicalForm), key=lambda p: tuple(map(str, p))
    ):
        if "/LexicalEntry/siciliano/" not in str(subject):
            continue
        for form, meta in bank.get(str(lemma), []):
            yield str(subject), form, {**meta, "lexical_entry_uri": str(subject)}


def liita_sentences(path):
    """Read source terminals in citation order, excluding lemma/translation labels."""
    from rdflib import Graph, Literal, Namespace

    graph = Graph().parse(path, format="turtle")
    corpus = Namespace("http://lila-erc.eu/ontologies/lila_corpora/")
    powla = Namespace("http://purl.org/powla/powla.owl#")
    for subject in sorted(
        graph.subjects(corpus.hasRefType, Literal("Sentence")), key=str
    ):
        children = set(graph.objects(subject, powla.hasChild))
        node, ordered, seen = graph.value(subject, corpus.first), [], set()
        while node is not None:
            if node in seen or node not in children:
                raise ResourceError(f"Invalid LiITA terminal chain at {subject}")
            seen.add(node)
            value = graph.value(node, powla.hasStringValue)
            if value is None:
                raise ResourceError(f"Missing LiITA terminal surface: {node}")
            ordered.append(str(value))
            node = graph.value(node, powla.next)
        if seen != children:
            raise ResourceError(f"Incomplete LiITA sentence: {subject}")
        yield (
            str(subject),
            " ".join(ordered),
            {
                "sentence_uri": str(subject),
                "source_terminals": ordered,
                "extraction_note": "RDF terminal spacing is reconstructed; original Turtle retained; not new Sicilian composition.",
            },
        )


def make_record(source, spec, document, locator, text, kind, meta=None):
    return {
        "id": source + ":" + digest(document + "\0" + locator)[:24],
        "source_id": source,
        "document_id": source + ":" + document,
        "locator": locator,
        "text": text,
        "original_text": text,
        "source_sha256": spec["sha256"],
        "unit_kind": kind,
        "data_role": spec["data_role"],
        "metadata": meta or {},
    }


def extract_records(root, catalog):
    """Reuse checksummed legacy extraction; replace PDF boundaries and add exports."""
    sources = catalog["sources"]
    legacy_path = root / ".cache/scn/corpus/records.jsonl"
    legacy_receipt = read_json(root, SCN_EVIDENCE / "corpus/extraction.json")
    if fingerprints(legacy_path)["sha256"] != legacy_receipt["records_sha256"]:
        raise ResourceError(
            "Legacy extraction checksum mismatch; run extract for the original catalog first"
        )
    acquired = {}
    for name, spec in sources.items():
        if spec.get("acquisition_status") != "acquired":
            continue
        acquired[name] = verify_file(local_path(root, spec["path"]), spec)
        if spec.get("publisher_sha1"):
            with local_path(root, spec["path"]).open("rb") as original:
                observed_sha1 = hashlib.file_digest(original, "sha1").hexdigest()
            if observed_sha1 != spec["publisher_sha1"]:
                raise ResourceError(f"Publisher SHA1 mismatch: {name}")
        if spec["data_role"] not in {"reference", "orthography"} and spec["audit"][
            "research_and_training"
        ]["local_analysis"]["value"] not in {"permitted", "conditional"}:
            raise ResourceError(f"Local analysis permission unresolved: {name}")
    out = root / CACHE
    out.mkdir(parents=True, exist_ok=True)
    path = out / "records.jsonl"
    temp = path.with_suffix(".part")
    counts, occurrences, pages = Counter(), Counter(), []
    with temp.open("w", encoding="utf-8") as stream:

        def emit(row):
            if not tokens(row["text"]):
                return
            # Repeated upstream localization keys have distinct physical occurrences.
            # Keep the legacy ID and expose the collision instead of dropping text.
            original_id = row["id"]
            occurrences[original_id] += 1
            if occurrences[original_id] > 1:
                row["legacy_id"] = original_id
                row["id"] += f":occurrence-{occurrences[original_id]}"
                row["metadata"]["locator_occurrence"] = occurrences[original_id]
            row["text_sha256"] = digest(row["text"])
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            counts[row["source_id"]] += 1

        with legacy_path.open(encoding="utf-8") as original:
            for line in original:
                row = json.loads(line)
                spec = sources[row["source_id"]]
                if row["source_sha256"] != spec.get("sha256"):
                    raise ResourceError("Stale legacy extraction source checksum")
                if spec["extractor"] == "pdf":
                    continue
                if row["unit_kind"] == "localization_fragment" and row[
                    "document_id"
                ].endswith(".xml"):
                    repaired = android_unescape(row["text"])
                    if repaired != row["text"]:
                        row["extraction_input_text"] = row["text"]
                        row["text"] = repaired
                        row["metadata"]["extraction_repairs"] = [
                            "Decode Android XML resource escape sequences; original_text unchanged"
                        ]
                emit(row)
        bank = defaultdict(list)
        if "scn_liita_lemmabank" in acquired:
            for lemma, form, meta in liita_forms(
                root / sources["scn_liita_lemmabank"]["path"]
            ):
                bank[lemma].append((form, meta))
        for name, spec in sources.items():
            if name not in acquired:
                continue
            parser = spec.get("extractor")
            source_path = root / spec["path"]
            if parser == "pdf":
                import pymupdf

                with pymupdf.open(source_path) as pdf:
                    for p, page in enumerate(pdf):
                        for block in page.get_text("blocks", sort=False):
                            if block[6] != 0:
                                continue
                            original = block[4]
                            # Only unwrap within a PDF text block; no dehyphenation or respelling.
                            text = re.sub(r"[ \t]*\n[ \t]*", " ", original).strip()
                            row = make_record(
                                name,
                                spec,
                                "issue",
                                f"page-{p + 1}:block-{block[5]}",
                                text,
                                "pdf_block",
                                {
                                    "page": p + 1,
                                    "bbox": list(block[:4]),
                                    "quality": "block_extraction_requires_review",
                                    "linebreak_hyphen": bool(
                                        re.search(r"\w-\s*\n\s*\w", original)
                                    ),
                                },
                            )
                            row["original_text"] = original
                            emit(row)
            elif parser == "wikipedia_xml":
                for doc, loc, text, kind, meta in wikipedia_records(source_path, pages):
                    emit(make_record(name, spec, doc, loc, text, kind, meta))
            elif parser == "liita_lemmas":
                for lemma, forms in bank.items():
                    for form, meta in forms:
                        emit(
                            make_record(
                                name,
                                spec,
                                lemma,
                                digest(form),
                                form,
                                "lexical_entry",
                                meta,
                            )
                        )
            elif parser == "liita_glossary":
                for entry, form, meta in liita_glossary(source_path, bank):
                    emit(
                        make_record(
                            name, spec, entry, digest(form), form, "lexical_entry", meta
                        )
                    )
            elif parser == "liita_treebank":
                for loc, text, meta in liita_sentences(source_path):
                    emit(make_record(name, spec, name, loc, text, "sentence", meta))
    temp.replace(path)
    write_json(out / "wikipedia-pages.json", {"pages": pages})
    receipt = {
        "version": VERSION,
        "records_sha256": fingerprints(path)["sha256"],
        "repeated_legacy_locator_occurrences": sum(n - 1 for n in occurrences.values()),
        "sources": dict(counts),
        "artifacts": acquired,
        "catalog_sha256": fingerprints(root / SCN_CATALOG)["sha256"],
        "legacy_records_sha256": legacy_receipt["records_sha256"],
    }
    write_json(root / EVIDENCE / "extraction.json", receipt)
    return path


def validate_catalog(catalog):
    original = catalog.get("original_source_ids", [])
    if (
        len(original) != 83
        or len(set(original)) != 83
        or set(original) - catalog["sources"].keys()
    ):
        raise ResourceError("All 83 original inventory entries must remain auditable")
    for name, spec in catalog["sources"].items():
        audit = spec.get("audit", {})

        def claim(c):
            if (
                not c.get("explanation")
                or not c.get("evidence_urls")
                or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", c.get("checked_on", ""))
            ):
                raise ResourceError(f"Missing evidence/date/explanation: {name}")
            if any(
                not u.startswith(("https://", "http://")) for u in c["evidence_urls"]
            ):
                raise ResourceError(f"Invalid evidence URL: {name}")

        for dimension in (
            "bulk_collection",
            "copyright",
            "dates",
            "dos24",
            "language_coverage",
        ):
            claim(audit.get(dimension, {}))
        if audit["bulk_collection"]["value"] not in PERMISSIONS:
            raise ResourceError(f"Invalid collection permission: {name}")
        if audit["dos24"]["value"] not in CATEGORIES:
            raise ResourceError(f"Invalid DOS24 category: {name}")
        if audit["copyright"]["value"] not in {
            "public_domain",
            "copyrighted",
            "mixed",
            "unknown",
        }:
            raise ResourceError(f"Invalid copyright category: {name}")
        for use in (
            "local_analysis",
            "corpus_preparation",
            "model_training",
            "commercial_use",
            "derived_data_distribution",
            "derived_model_distribution",
        ):
            c = audit.get("research_and_training", {}).get(use, {})
            claim(c)
            if c.get("value") not in PERMISSIONS:
                raise ResourceError(f"Invalid use permission: {name}/{use}")


def apply_review(row, decision, rules):
    if (
        decision["source_sha256"] != row["source_sha256"]
        or decision["text_sha256"] != row["text_sha256"]
    ):
        raise ResourceError(f"Stale review: {row['id']}")
    if decision.get("rules_version") != rules["version"]:
        raise ResourceError(f"Stale review rules: {row['id']}")
    if decision.get("category") not in CATEGORIES or not all(
        decision.get(k)
        for k in ("reviewer", "checked_on", "explanation", "evidence_urls", "context")
    ):
        raise ResourceError(f"Incomplete contextual decision: {row['id']}")
    if decision["category"] == "verified_conforming":
        required = {r["id"] for r in rules["rules"] if r["kind"] == "contextual"}
        if not required <= set(decision.get("checked_rules", [])):
            raise ResourceError(
                f"Verified decision lacks contextual checklist: {row['id']}"
            )
        if {f["rule"] for f in row["flags"]} - set(
            decision.get("flag_dispositions", {})
        ):
            raise ResourceError(f"Verified decision has unresolved flags: {row['id']}")
    row.update(
        category=decision["category"],
        reviewed=True,
        review_explanation=decision["explanation"],
    )


def summary(rows):
    """Dedup inside this exact scope; conditional eligibility never steals a representative."""
    seen, vocab, unique_vocab = set(), set(), set()
    count = Counter()
    docs, pages, lemmas = set(), set(), set()
    for row in rows:
        words = row["tokens"]
        count["records"] += 1
        count["tokens"] += len(words)
        vocab.update(words)
        docs.add(row["document_id"])
        meta = row.get("metadata", {})
        if meta.get("page") is not None or meta.get("page_id") is not None:
            pages.add((row["source_id"], meta.get("page", meta.get("page_id"))))
        if meta.get("lemma_uri"):
            lemmas.add(meta["lemma_uri"])
        kind = row["unit_kind"]
        count[
            "lexical_forms"
            if kind == "lexical_entry"
            else "localization_fragments"
            if kind == "localization_fragment"
            else "prose_units"
        ] += 1
        count[
            "lexical_tokens"
            if kind == "lexical_entry"
            else "localization_tokens"
            if kind == "localization_fragment"
            else "prose_tokens"
        ] += len(words)
        count["variant_forms"] += meta.get("form_kind") == "variant"
        if kind not in {"lexical_entry", "localization_fragment"}:
            count["sentence_units"] += len(sentences(row["text"]))
        count["reviewed_records"] += bool(row["reviewed"])
        count["reviewed_tokens"] += len(words) if row["reviewed"] else 0
        count["flagged_records"] += bool(row["flags"])
        count["possible_italian"] += (
            row["language_screen"]["screen"] == "possible_italian"
        )
        count["possible_switching"] += (
            row["language_screen"]["screen"] == "possible_switching"
        )
        count["treebank_overlap_records"] += bool(row.get("treebank_overlap"))
        count["treebank_overlap_tokens"] += (
            len(words) if row.get("treebank_overlap") else 0
        )
        count["provenance_duplicate_records"] += bool(row.get("provenance_duplicate"))
        key = ("lexicon" if kind == "lexical_entry" else "text", tuple(words))
        if key not in seen:
            count["deduplicated_tokens"] += len(words)
            count["deduplicated_records"] += 1
            unique_vocab.update(words)
            if not row.get("treebank_overlap") and not row.get("provenance_duplicate"):
                count["deduplicated_novel_tokens"] += len(words)
        seen.add(key)
    defaults = (
        "records tokens lexical_forms localization_fragments prose_units lexical_tokens localization_tokens prose_tokens variant_forms sentence_units reviewed_records reviewed_tokens flagged_records possible_italian possible_switching treebank_overlap_records treebank_overlap_tokens provenance_duplicate_records deduplicated_tokens deduplicated_records deduplicated_novel_tokens"
    ).split()
    result = {k: count[k] for k in defaults}
    result.update(
        documents=len(docs),
        pages=len(pages),
        lemmas=len(lemmas),
        vocabulary=len(vocab),
        deduplicated_vocabulary=len(unique_vocab),
        exact_duplicate_records=count["records"] - count["deduplicated_records"],
        unreviewed_records=count["records"] - count["reviewed_records"],
    )
    return result


def training_eligible(spec):
    p = spec["audit"]["research_and_training"]
    return all(
        p[k]["value"] in {"permitted", "conditional"}
        for k in ("local_analysis", "corpus_preparation", "model_training")
    )


def _surface_key(text):
    # RDF splits clitics from their host; only undo that boundary for overlap matching.
    return tuple(tokens(re.sub(r"(['’])\s+", r"\1", text)))


def classify_record(row, spec, decision, rules, ud, ud_words):
    row["tokens"] = tokens(row["text"])
    row["flags"] = screen(row["text"], rules)
    if row.get("metadata", {}).get("linebreak_hyphen"):
        row["flags"].append({"rule": "E01", "matches": ["linebreak hyphen retained"]})
    row["language_screen"] = language_flags(row["tokens"])
    category = spec["audit"]["dos24"]["value"]
    if category == "verified_conforming":
        category = "unresolved"  # A source label cannot replace individual review.
    # Source affiliation supports inference; machine silence alone does not.
    if category == "supported_inferred" and (
        row["flags"] or row["language_screen"]["screen"] != "unresolved"
    ):
        category = "unresolved"
    if row["unit_kind"] == "pdf_block" and len(row["tokens"]) < 8:
        category = "unresolved"  # Short headings, names and credits lack prose context.
    if all(not any(c.isalpha() for c in w) for w in row["tokens"]) or re.fullmatch(
        r"https?://\S+", row["text"].strip()
    ):
        category = "not_applicable"
    row.update(category=category, reviewed=False)
    if decision is not None:
        apply_review(row, decision, rules)
    if row["source_id"].startswith("scn_liita_stb_"):
        row["treebank_overlap"] = ud.get(_surface_key(row["text"]))
        row["provenance_duplicate"] = spec.get("overlaps_source")
    if (
        row["source_id"].startswith("scn_liita_")
        and row["unit_kind"] == "lexical_entry"
    ):
        row["treebank_overlap"] = (
            "UD surface vocabulary"
            if len(row["tokens"]) == 1 and row["tokens"][0] in ud_words
            else None
        )


def article_summary(documents):
    # Article verification requires every retained body unit to have contextual review.
    article_rows, article_counts = [], Counter()
    for doc, parts in documents.items():
        categories = {p["category"] for p in parts}
        category = (
            "verified_conforming"
            if categories == {"verified_conforming"}
            else "mixed_nonconforming"
            if "mixed_nonconforming" in categories
            else "unresolved"
        )
        article_counts[category] += 1
        article_rows.append(
            {
                "document_id": doc,
                "page_id": parts[0]["metadata"]["page_id"],
                "revision_id": parts[0]["metadata"]["revision_id"],
                "body_category": category,
                "body_records": len(parts),
                "reviewed_records": sum(p["reviewed"] for p in parts),
                "verified_passage_tokens": sum(
                    len(p["tokens"])
                    for p in parts
                    if p["category"] == "verified_conforming"
                ),
            }
        )
    return article_rows, article_counts


def audit(root):
    catalog = load_catalog(root)
    validate_catalog(catalog)
    rules, reviews = (
        read_json(root, DATA / "rules.json"),
        read_json(root, DATA / "reviews.json"),
    )
    path = extract_records(root, catalog)
    rows = [json.loads(line) for line in path.open(encoding="utf-8")]
    ids = {r["id"] for r in rows}
    if len(ids) != len(rows):
        raise ResourceError("Duplicate audit record IDs")
    decisions = {d["id"]: d for d in reviews["decisions"]}
    if len(decisions) != len(reviews["decisions"]) or decisions.keys() - ids:
        raise ResourceError("Duplicate or unknown review IDs")
    ud = {
        _surface_key(r["text"]): r["id"] for r in rows if r["source_id"] == "scn_ud_stb"
    }
    ud_words = {
        w for r in rows if r["source_id"] == "scn_ud_stb" for w in tokens(r["text"])
    }
    by_source, by_category, documents = (
        defaultdict(list),
        defaultdict(list),
        defaultdict(list),
    )
    for row in rows:
        classify_record(
            row,
            catalog["sources"][row["source_id"]],
            decisions.get(row["id"]),
            rules,
            ud,
            ud_words,
        )
        by_source[row["source_id"]].append(row)
        by_category[row["category"]].append(row)
        if row["source_id"] == "scn_wikipedia_20261001":
            documents[row["document_id"]].append(row)
    output = root / CACHE / "screened.jsonl"
    with output.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    article_rows, article_counts = article_summary(documents)
    write_json(root / CACHE / "wikipedia-articles.json", {"articles": article_rows})
    stats = {
        "version": VERSION,
        "rules_version": rules["version"],
        "checked_on": catalog["checked_on"],
        "implementation_sha256": fingerprints(Path(__file__))["sha256"],
        "catalog_sha256": fingerprints(root / SCN_CATALOG)["sha256"],
        "rules_sha256": fingerprints(root / DATA / "rules.json")["sha256"],
        "reviews_sha256": fingerprints(root / DATA / "reviews.json")["sha256"],
        "screened_sha256": fingerprints(output)["sha256"],
        "total": summary(rows),
        "sources": {
            name: {
                "measured": summary(by_source[name]),
                "categories": {
                    c: summary(r for r in by_source[name] if r["category"] == c)
                    for c in CATEGORIES
                },
                "acquisition_outcome": s["acquisition_status"],
                "publisher_reported": s.get("reported_size"),
                "measurement_status": "measured"
                if by_source[name]
                else "reference_not_counted"
                if s["data_role"] in {"reference", "orthography"}
                else "not_acquired_or_no_extracted_text",
            }
            for name, s in catalog["sources"].items()
        },
        "categories": {c: summary(by_category[c]) for c in CATEGORIES},
        "current_cohort": {
            "excluded_sources": [
                name
                for name, s in catalog["sources"].items()
                if name == "scn_wikipedia_20231101" or s.get("overlaps_source")
            ],
            "total": summary(
                r
                for r in rows
                if r["source_id"] != "scn_wikipedia_20231101"
                and not r.get("provenance_duplicate")
            ),
            "categories": {
                c: summary(
                    r
                    for r in by_category[c]
                    if r["source_id"] != "scn_wikipedia_20231101"
                    and not r.get("provenance_duplicate")
                )
                for c in CATEGORIES
            },
        },
        "text_categories": {
            c: summary(r for r in by_category[c] if r["unit_kind"] != "lexical_entry")
            for c in CATEGORIES
        },
        "research_training_intersection": {
            c: summary(
                r
                for r in by_category[c]
                if training_eligible(catalog["sources"][r["source_id"]])
            )
            for c in CATEGORIES
        },
        "unconditional_training_intersection": {
            c: summary(
                r
                for r in by_category[c]
                if all(
                    catalog["sources"][r["source_id"]]["audit"][
                        "research_and_training"
                    ][k]["value"]
                    == "permitted"
                    for k in ("local_analysis", "corpus_preparation", "model_training")
                )
            )
            for c in CATEGORIES
        },
        "wikipedia": {
            "body_articles": len(documents),
            "article_categories": dict(article_counts),
            "page_exclusions": dict(
                Counter(
                    p["exclusion"] or "retained_article"
                    for p in read_json(root, CACHE + "/wikipedia-pages.json")["pages"]
                )
            ),
            "verified_passages_in_mixed_articles": summary(
                p
                for parts in documents.values()
                if any(r["category"] == "mixed_nonconforming" for r in parts)
                for p in parts
                if p["category"] == "verified_conforming"
            ),
        },
        "rules_hits": dict(Counter(f["rule"] for r in rows for f in r["flags"])),
        "definitions": {
            "dedup": "Exact NFC/case/apostrophe-normalized token sequences, lexical and text roles separate. Dedup independently within each reported scope; category/source dedup totals need not add to global dedup.",
            "sentences": "Heuristic punctuation units for prose; UI fragments and lexical forms have zero sentence units. Not linguistically certified segmentation.",
            "pages": "PDF page or current MediaWiki page ID; zero means no page representation, not zero documents.",
            "eligibility": "Conditional intersection includes license obligations; not an unconditional training grant. Model distribution remains separately assessed.",
            "review": "All records screened, only explicit hashed decisions contextually reviewed. Rule silence never verifies.",
            "liita_overlap": "Treebank sentence equality after rejoining apostrophe clitics is measured separately. All ten documented RDF reproductions of existing UD documents are excluded from novel totals by provenance, including punctuation/segmentation differences. Lexical overlap conservatively matches single-word UD surface vocabulary. Similarity alone does not delete text.",
            "wiki": "Main-namespace nonredirect static article body; no rendered templates, tables, media captions or references. 2023 snapshot is a separate source, not current revisions.",
        },
    }
    write_json(root / EVIDENCE / "statistics.json", stats)
    # Deterministic contextual review queue, not a representative certification sample.
    queue = []
    for name, subset in by_source.items():
        chosen = sorted(subset, key=lambda r: (bool(r["flags"]), digest(r["id"])))[:8]
        chosen += sorted(
            (r for r in subset if r["flags"]), key=lambda r: digest(r["id"])
        )[:4]
        for row in chosen:
            queue.append(
                {
                    k: row[k]
                    for k in (
                        "id",
                        "source_id",
                        "text",
                        "source_sha256",
                        "text_sha256",
                        "metadata",
                        "flags",
                        "category",
                    )
                }
            )
    write_json(root / CACHE / "review-queue.json", {"records": queue})
    validate(root)
    print(
        json.dumps(
            {
                "total": stats["total"],
                "categories": {c: s["tokens"] for c, s in stats["categories"].items()},
            },
            indent=2,
        )
    )


def validate(root):
    catalog = load_catalog(root)
    validate_catalog(catalog)
    path = root / EVIDENCE / "statistics.json"
    if not path.exists():
        return
    stats = read_json(root, EVIDENCE / "statistics.json")
    if fingerprints(Path(__file__))["sha256"] != stats["implementation_sha256"]:
        raise ResourceError("Audit implementation changed; rerun audit")
    for relative, key in [
        (SCN_CATALOG, "catalog_sha256"),
        (DATA / "rules.json", "rules_sha256"),
        (DATA / "reviews.json", "reviews_sha256"),
        (CACHE + "/screened.jsonl", "screened_sha256"),
    ]:
        if fingerprints(root / relative)["sha256"] != stats[key]:
            raise ResourceError(f"Stale audit artifact: {relative}")
    if set(stats["sources"]) != set(catalog["sources"]):
        raise ResourceError("Missing source measurements")
    for field in ("records", "tokens", "reviewed_records", "reviewed_tokens"):
        if (
            sum(s["measured"][field] for s in stats["sources"].values())
            != stats["total"][field]
            or sum(s[field] for s in stats["categories"].values())
            != stats["total"][field]
        ):
            raise ResourceError(f"Audit reconciliation failed: {field}")
    if (
        sum(stats["wikipedia"]["article_categories"].values())
        != stats["wikipedia"]["body_articles"]
    ):
        raise ResourceError("Wikipedia page reconciliation failed")
    for s in stats["sources"].values():
        m = s["measured"]
        if (
            m["tokens"]
            != m["lexical_tokens"] + m["localization_tokens"] + m["prose_tokens"]
        ):
            raise ResourceError("Unit/token reconciliation failed")
        for field in ("records", "tokens", "reviewed_records", "reviewed_tokens"):
            if sum(c[field] for c in s["categories"].values()) != m[field]:
                raise ResourceError(f"Source category reconciliation failed: {field}")
        if (
            m["reviewed_records"] + m["unreviewed_records"] != m["records"]
            or m["deduplicated_tokens"] > m["tokens"]
        ):
            raise ResourceError("Review/duplicate coverage reconciliation failed")
    for field in ("records", "tokens"):
        if (
            sum(c[field] for c in stats["current_cohort"]["categories"].values())
            != stats["current_cohort"]["total"][field]
        ):
            raise ResourceError(f"Current cohort reconciliation failed: {field}")
    print("Audit evidence, hashes, source/category/page/token totals validated")


def report(root):
    validate(root)
    from asr.scn.report import write_report

    write_report(
        root, load_catalog(root), read_json(root, EVIDENCE / "statistics.json")
    )
