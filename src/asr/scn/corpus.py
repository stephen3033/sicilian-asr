"""Source-preserving SCN extraction and descriptive statistics; no DOS24 conversion."""

import csv
import hashlib
import json
import re
import tarfile
import unicodedata
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from html.parser import HTMLParser
from pathlib import Path

from asr.scn.acquisition import load_catalog
from common.errors import ResourceError
from common.resources import verify_file
from common.utils import fingerprints, local_path, write_json

RULES = "scn-nfc-apostrophes-preserve-diacritics-no-spelling-conversion-v1"
CATEGORIES = ("dos24", "cademia_pre2024", "other_sicilian", "uncertain_mixed")
BARRIER = "\ufff0"
APOSTROPHES = str.maketrans({c: "'" for c in "’‘ʼ＇"})


def tokens(text):
    """Keep lexical diacritics and elision apostrophes, including word-initial ones."""
    text = unicodedata.normalize("NFC", text).lower().translate(APOSTROPHES)
    words, current = [], []
    for i, char in enumerate(text):
        lexical = char.isalnum() or (
            unicodedata.category(char).startswith("M") and current
        )
        apostrophe = char == "'" and (
            (i + 1 < len(text) and text[i + 1].isalpha()) or bool(current)
        )
        if lexical or apostrophe:
            current.append(char)
        elif current:
            word = "".join(current)
            if any(c.isalnum() for c in word):
                words.append(word)
            current = []
    if current and any(c.isalnum() for c in current):
        words.append("".join(current))
    return words


class MarkupText(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts = []

    def handle_data(self, data):
        self.parts.append(data)

    def handle_starttag(self, tag, attrs):
        if tag in {"br", "p", "div", "li"}:
            self.parts.append(" ")

    def handle_endtag(self, tag):
        if tag in {"p", "div", "li"}:
            self.parts.append(" ")


def clean_fragments(text):
    """Do not create an n-gram across an unknown variable or an inline code token."""
    text = re.sub(
        r"\{[^{}]*\}|\$[A-Za-z_][\w-]*\$|%(?:\d+\$)?[-+\d.]*[@sdif]|&[A-Za-z][\w.-]*;",
        BARRIER,
        text,
    )
    parser = MarkupText()
    parser.feed(text)
    return [
        part.strip() for part in "".join(parser.parts).split(BARRIER) if tokens(part)
    ]


def fluent_records(text):
    from fluent.syntax import FluentParser, ast

    def patterns(pattern):
        paths = [""]
        for element in pattern.elements:
            if isinstance(element, ast.TextElement):
                options = [element.value]
            elif isinstance(element, ast.Placeable) and isinstance(
                element.expression, ast.SelectExpression
            ):
                options = [
                    p for v in element.expression.variants for p in patterns(v.value)
                ]
            else:
                options = [BARRIER]
            if len(paths) * len(options) > 128:
                raise ResourceError(
                    "Fluent select expansion exceeds 128 paths; review source"
                )
            paths = [a + b for a in paths for b in options]
        return paths

    resource = FluentParser(with_spans=True).parse(text)
    for entry in resource.body:
        if isinstance(entry, ast.Junk):
            raise ResourceError("Unparsed Fluent syntax; refuse silent extraction loss")
        if not isinstance(entry, ast.Message):
            continue
        variants = [("value", entry.value)] + [
            (a.id.name, a.value)
            for a in entry.attributes
            if a.id.name not in {"accesskey", "key", "keycode", "style"}
        ]
        for attribute, pattern in variants:
            if pattern is None:
                continue
            for number, value in enumerate(patterns(pattern)):
                for fragment, cleaned in enumerate(clean_fragments(value)):
                    yield (
                        f"{entry.id.name}.{attribute}.{number}.{fragment}",
                        cleaned,
                        text[pattern.span.start : pattern.span.end],
                        "localization_fragment",
                    )


def xml_text(node):
    output = node.text or ""
    for child in node:
        if child.tag.rsplit("}", 1)[-1] in {"x", "ph", "bpt", "ept", "it"}:
            output += BARRIER
        else:
            output += xml_text(child)
        output += child.tail or ""
    return output


def localization_records(name, text):
    suffix = Path(name).suffix
    if suffix == ".ftl":
        yield from fluent_records(text)
    elif suffix in {".xml", ".xliff", ".xlf"}:
        root = ET.fromstring(text)
        if suffix == ".xml":
            for entry in root:
                if entry.attrib.get("translatable") == "false":
                    continue
                leaves = (
                    list(entry) if entry.tag in {"plurals", "string-array"} else [entry]
                )
                for i, leaf in enumerate(leaves):
                    original = xml_text(leaf)
                    for j, part in enumerate(clean_fragments(original)):
                        yield (
                            f"{entry.get('name', '')}.{i}.{j}",
                            part,
                            original,
                            "localization_fragment",
                        )
        else:
            for entry in root.iter():
                if entry.tag.rsplit("}", 1)[-1] not in {"trans-unit", "segment"}:
                    continue
                children = {n.tag.rsplit("}", 1)[-1]: n for n in entry}
                if "target" not in children:
                    continue
                original = xml_text(children["target"])
                if "source" in children and original == xml_text(children["source"]):
                    continue  # untranslated source fallback, not confirmed Sicilian
                for j, part in enumerate(clean_fragments(original)):
                    yield (
                        f"{entry.get('id', '')}.{j}",
                        part,
                        original,
                        "localization_fragment",
                    )
    elif suffix == ".po":
        import polib

        for i, entry in enumerate(polib.pofile(text)):
            if entry.obsolete or "fuzzy" in entry.flags:
                continue
            for plural, original in (
                {"singular": entry.msgstr} | entry.msgstr_plural
            ).items():
                if not original or original in {entry.msgid, entry.msgid_plural}:
                    continue
                for j, part in enumerate(clean_fragments(original)):
                    yield f"{i}.{plural}.{j}", part, original, "localization_fragment"
    elif suffix == ".json":
        for key, value in json.loads(text).items():
            if isinstance(value, dict) and isinstance(value.get("message"), str):
                for j, part in enumerate(clean_fragments(value["message"])):
                    yield f"{key}.{j}", part, value["message"], "localization_fragment"
    elif suffix == ".properties":
        text = re.sub(r"\\\r?\n\s*", "", text)
        for i, line in enumerate(text.splitlines()):
            if not line.strip() or line.lstrip().startswith(("#", "!")):
                continue
            match = re.match(r"\s*([^:=\s]+)\s*[:=]\s*(.*)", line)
            if match:
                original = re.sub(
                    r"\\u([0-9a-fA-F]{4})", lambda m: chr(int(m[1], 16)), match[2]
                )
                original = original.replace("\\n", " ").replace("\\t", " ")
                for j, part in enumerate(clean_fragments(original)):
                    yield f"{match[1]}.{i}.{j}", part, match[2], "localization_fragment"
    elif suffix == ".dtd":
        for i, match in enumerate(
            re.finditer(r'<!ENTITY\s+([\w.-]+)\s+([\'"])(.*?)\2\s*>', text, re.S)
        ):
            if match[1].lower().endswith(("accesskey", "key", "style")):
                continue
            for j, part in enumerate(clean_fragments(match[3])):
                yield f"{match[1]}.{i}.{j}", part, match[3], "localization_fragment"


def conllu_records(text):
    doc, sentence = "document", 0
    for block in re.split(r"\n\s*\n", text):
        meta = dict(re.findall(r"^# ([^=\n]+?)\s*=\s*(.*)$", block, re.M))
        doc = meta.get("newdoc id", doc)
        if "text" not in meta:
            continue
        sentence += 1
        # Surface text is authoritative; syntactic rows split contractions differently.
        yield doc, meta.get("sent_id", str(sentence)), meta["text"], meta


def sentences(text):
    """Descriptive sentence units, not a claim of linguist-validated segmentation."""
    return [s.strip() for s in re.split(r"(?<=[.!?…])\s+|\n\s*\n", text) if tokens(s)]


def extract(root):
    catalog = load_catalog(root)
    output = root / ".cache/scn/corpus/records.jsonl"
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".part")
    report = {
        "rules": RULES,
        "sources": {},
        "errors": [],
        "catalog_sha256": fingerprints(
            root / "src/asr/scn/resources/sources/catalog.json"
        )["sha256"],
    }
    count = 0
    with temporary.open("w", encoding="utf-8") as stream:
        for source_id, spec in catalog["sources"].items():
            if spec.get("extractor") in {
                "wikipedia_xml",
                "liita_lemmas",
                "liita_glossary",
                "liita_treebank",
            }:
                # These need page/RDF identities and are extracted by the audit CLI.
                continue
            if spec.get("acquisition_status") != "acquired" or spec.get(
                "data_role"
            ) in {"reference", "orthography"}:
                continue
            if spec["permissions"].get("local_analysis") != "allowed":
                continue
            path = local_path(root, spec["path"])
            actual = verify_file(path, spec)
            info = {
                "records": 0,
                "artifacts_sha256": actual["sha256"],
                "members": {},
                "skipped_members": [],
            }
            report["sources"][source_id] = info

            def emit(doc, locator, text, original=None, kind="sentence", metadata=None):
                nonlocal count
                if not tokens(text):
                    return
                identity = f"{source_id}\0{doc}\0{locator}"
                record = {
                    "id": hashlib.sha256(identity.encode()).hexdigest()[:24],
                    "source_id": source_id,
                    "document_id": source_id + ":" + doc,
                    "locator": str(locator),
                    "text": text,
                    "original_text": original if original is not None else text,
                    "source_sha256": actual["sha256"],
                    "rules": RULES,
                    "orthography": spec["orthography"],
                    "orthography_evidence": spec["orthography_evidence"],
                    "data_role": spec["data_role"],
                    "unit_kind": kind,
                    "language_assessment": "source-declared scn; automatic screening is not verification",
                    "metadata": metadata or {},
                }
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                count += 1
                info["records"] += 1

            parser = spec["extractor"]
            if parser in {"localization_archive", "conllu_archive", "flores_archive"}:
                with tarfile.open(path, "r:gz") as archive:
                    for member in archive:
                        if not member.isfile():
                            continue
                        name = member.name.split("/", 1)[-1]
                        if parser == "localization_archive":
                            if not (
                                {"scn", "values-scn"} & set(name.split("/"))
                            ) or name.startswith("server/data/"):
                                continue
                            if Path(name).suffix not in {
                                ".ftl",
                                ".xml",
                                ".xliff",
                                ".xlf",
                                ".po",
                                ".json",
                                ".properties",
                                ".dtd",
                            }:
                                info["skipped_members"].append(name)
                                continue
                        elif parser == "conllu_archive" and not name.endswith(
                            ".conllu"
                        ):
                            continue
                        elif parser == "flores_archive" and "scn_Latn" not in name:
                            continue
                        # Read members in place: no archive path traversal or code execution.
                        data = archive.extractfile(member).read()
                        info["members"][name] = hashlib.sha256(data).hexdigest()
                        text = data.decode("utf-8-sig")
                        if parser == "localization_archive":
                            for (
                                location,
                                cleaned,
                                original,
                                kind,
                            ) in localization_records(name, text):
                                emit(name, location, cleaned, original, kind)
                        elif parser == "conllu_archive":
                            for doc, location, sentence, meta in conllu_records(text):
                                emit(
                                    name + ":" + doc, location, sentence, metadata=meta
                                )
                        else:
                            for i, line in enumerate(text.splitlines()):
                                emit(name + ":" + str(i), i, line)
            elif parser == "wikipedia_parquet":
                import pyarrow.parquet as pq

                for batch in pq.ParquetFile(path).iter_batches(batch_size=256):
                    for row in batch.to_pylist():
                        for i, sentence in enumerate(sentences(row["text"])):
                            emit(
                                str(row["id"]),
                                i,
                                sentence,
                                metadata={"url": row["url"], "title": row["title"]},
                            )
            elif parser == "napizia_csv":
                with path.open(encoding="utf-8-sig", newline="") as handle:
                    reader = csv.DictReader(handle)
                    field = next(
                        (
                            x
                            for x in reader.fieldnames
                            if x.lower() in {"scn", "sicilian", "sicilianu"}
                        ),
                        None,
                    )
                    if field is None:
                        raise ResourceError(
                            f"Unrecognized Sicilian CSV column: {reader.fieldnames}"
                        )
                    for i, row in enumerate(reader):
                        emit(
                            str(i),
                            i,
                            row[field],
                            metadata={k: v for k, v in row.items() if k != field},
                        )
            elif parser == "toponyms_csv":
                with path.open(encoding="utf-8-sig", newline="") as handle:
                    for i, row in enumerate(csv.DictReader(handle)):
                        for field in ("SCN", "LUCALI", "ABBITANTI"):
                            for j, form in enumerate(row[field].split(",")):
                                emit(
                                    str(i),
                                    f"{field}:{j}",
                                    form.strip(),
                                    kind="lexical_entry",
                                    metadata=row,
                                )
            elif parser == "smol_jsonl":
                for i, line in enumerate(path.read_text().splitlines()):
                    row = json.loads(line)
                    targets = row.get("trgs")
                    if (
                        row.get("tl") != "scn"
                        or not isinstance(targets, list)
                        or not all(isinstance(t, str) for t in targets)
                    ):
                        raise ResourceError(
                            f"Unrecognized Google translation record: {list(row)}"
                        )
                    for j, part in enumerate(targets):
                        emit(
                            str(row.get("id", i)),
                            j,
                            part,
                            kind="lexical_entry"
                            if spec["data_role"] == "lexicon"
                            else "sentence",
                            metadata={
                                "source_row": i,
                                "target_index": j,
                                "human_translation": True,
                            },
                        )
            elif parser == "ipa_tsv":
                for i, line in enumerate(path.read_text().splitlines()):
                    word, ipa = line.split("\t")
                    emit(
                        str(i),
                        i,
                        word,
                        kind="lexical_entry",
                        metadata={
                            "source_ipa": ipa,
                            "validation": "crowdsourced_unvalidated",
                        },
                    )
            elif parser == "hf_rows":
                payload = json.loads(path.read_text())
                if payload.get("partial"):
                    raise ResourceError("Incomplete dataset viewer response")
                for item in payload["rows"]:
                    row = item["row"]
                    if item.get("truncated_cells") or row["language"] != "scn_Latn":
                        raise ResourceError(
                            "Truncated or wrong-language transcript row"
                        )
                    urls = [a["src"] for a in row["audio"]]
                    if not urls or any(spec["revision"] not in u for u in urls):
                        raise ResourceError(
                            "Transcript viewer revision differs from pinned upstream revision"
                        )
                    for j, sentence in enumerate(sentences(row["raw_text"])):
                        emit(
                            row["speaker_id"] + ":" + row["prompt_id"],
                            f"{item['row_idx']}:{j}",
                            sentence,
                            metadata={
                                k: v
                                for k, v in row.items()
                                if k not in {"audio", "raw_text"}
                            },
                        )
            elif parser == "perl_lexicon":
                text = path.read_text()
                for i, match in enumerate(
                    re.finditer(
                        r'display_as\s*=>\s*([\'"])((?:\\.|(?!\1)[^\n])*)\1', text
                    )
                ):
                    form = re.sub(r"\\([\'\"\\])", r"\1", match[2])
                    emit(
                        str(i),
                        match.start(),
                        form,
                        original=match[2],
                        kind="lexical_entry",
                    )
            elif parser == "text_lines":
                for i, line in enumerate(path.read_text().splitlines()):
                    emit(str(i), i, line)
            elif parser == "tatoeba_bz2":
                import bz2

                with bz2.open(path, "rt", encoding="utf-8") as handle:
                    for row in csv.reader(handle, delimiter="\t"):
                        if len(row) >= 3 and row[1] == "scn":
                            emit(
                                row[0],
                                row[0],
                                row[2],
                                metadata={"attribution": row[3:]},
                            )
            elif parser == "pdf":
                from pypdf import PdfReader

                for i, page in enumerate(PdfReader(path).pages):
                    text = page.extract_text(extraction_mode="layout") or ""
                    for j, part in enumerate(sentences(text)):
                        emit(
                            "issue",
                            f"page-{i + 1}:{j}",
                            part,
                            metadata={
                                "page": i + 1,
                                "quality": "pdf_layout_unreviewed",
                            },
                        )
            else:
                raise ResourceError(f"Unknown SCN extractor: {parser}")
            print(f"EXTRACTED {source_id}: {info['records']} records", flush=True)
    temporary.replace(output)
    report.update(records=count, records_sha256=fingerprints(output)["sha256"])
    write_json(root / "src/asr/scn/resources/corpus/extraction.json", report)


ITALIAN_MARKERS = set(
    "della delle degli nella nelle questo questa sono essere puoi quindi perché viene vengono abbiamo anche quando quello quella degli della tuo tua vuoi clicca scegliere".split()
)
SICILIAN_MARKERS = set(
    "chiḍḍu chiḍḍi cchiù chistu chista sunnu picchì nta ntô dû pû cû macari accussì sicilianu nzinu nzumma siddu quannu avemu chiḍḍa agghicari staju scàrrica tìnniri".split()
)


def language_flags(words):
    it = sorted(set(words) & ITALIAN_MARKERS)
    scn = sorted(set(words) & SICILIAN_MARKERS)
    label = (
        "possible_switching"
        if len(it) >= 2 and len(scn) >= 2
        else "possible_italian"
        if len(it) >= 2 and not scn
        else "unresolved"
    )
    return {
        "screen": label,
        "italian_markers": it,
        "sicilian_markers": scn,
        "confirmed": False,
    }


class Groups:
    def __init__(self):
        self.parent = {}

    def find(self, value):
        self.parent.setdefault(value, value)
        if self.parent[value] != value:
            self.parent[value] = self.find(self.parent[value])
        return self.parent[value]

    def union(self, a, b):
        a, b = self.find(a), self.find(b)
        self.parent[max(a, b)] = min(a, b)


def near_duplicates(records, threshold=0.85):
    """Deterministic MinHash candidates, exact 5-token-shingle Jaccard confirmation."""
    buckets, shingles, pairs = defaultdict(list), {}, []
    prime = 2**61 - 1
    for index, row in enumerate(records):
        words = row["tokens"]
        if row["data_role"] != "text" or len(words) < 10 or row.get("duplicate_of"):
            continue
        spans = {tuple(words[i : i + 5]) for i in range(len(words) - 4)}
        hashes = [
            int.from_bytes(hashlib.sha256("\0".join(s).encode()).digest()[:8], "big")
            % prime
            for s in spans
        ]
        signature = [
            min(((j * 2 + 3) * v + (j + 1) * 104729) % prime for v in hashes)
            for j in range(16)
        ]
        candidates = set()
        keys = [
            (band, tuple(signature[band * 4 : (band + 1) * 4])) for band in range(4)
        ]
        for key in keys:
            candidates.update(buckets[key])
        for earlier in sorted(candidates):
            other = shingles[earlier]
            if min(len(spans), len(other)) / max(len(spans), len(other)) < threshold:
                continue
            score = len(spans & other) / len(spans | other)
            if threshold <= score < 1:
                pairs.append(
                    {"a": records[earlier]["id"], "b": row["id"], "jaccard": score}
                )
        shingles[index] = spans
        for key in keys:
            buckets[key].append(index)
    return pairs


def summary(rows):
    vocabulary = Counter(w for row in rows for w in row["tokens"])
    retained = [r for r in rows if not r.get("duplicate_of")]
    return {
        "documents": len({r["document_id"] for r in rows}),
        "records": len(rows),
        "sentence_units": sum(r["unit_kind"] == "sentence" for r in rows),
        "localization_fragments": sum(
            r["unit_kind"] == "localization_fragment" for r in rows
        ),
        "lexical_entries": sum(r["unit_kind"] == "lexical_entry" for r in rows),
        "tokens": sum(vocabulary.values()),
        "unique_word_forms": len(vocabulary),
        "exact_duplicate_records": len(rows) - len(retained),
        "deduplicated_tokens": sum(len(r["tokens"]) for r in retained),
        "screening": dict(Counter(r["language_screen"]["screen"] for r in rows)),
        "confirmed_italian_records": None,
        "confirmed_switching_records": None,
    }


def analyze(root):
    path = root / ".cache/scn/corpus/records.jsonl"
    extraction = json.loads(
        (root / "src/asr/scn/resources/corpus/extraction.json").read_text()
    )
    if fingerprints(path)["sha256"] != extraction["records_sha256"]:
        raise ResourceError("Extracted records differ from their receipt")
    rows, seen, groups = [], {}, Groups()
    for line in path.open():
        row = json.loads(line)
        row["tokens"] = tokens(row["text"])
        row["language_screen"] = language_flags(row["tokens"])
        # Lexical entries cannot steal a training sentence's deduplication representative.
        key = (row["data_role"], tuple(row["tokens"]))
        if key in seen:
            other = seen[key]
            row["duplicate_of"] = other["id"]
            # Short shared UI labels are removed globally but do not join whole publications.
            if len(row["tokens"]) >= 5:
                groups.union(row["document_id"], other["document_id"])
        else:
            seen[key] = row
        groups.find(row["document_id"])
        rows.append(row)
    pairs = near_duplicates(rows)
    lookup = {r["id"]: r for r in rows}
    for pair in pairs:
        groups.union(lookup[pair["a"]]["document_id"], lookup[pair["b"]]["document_id"])
    normalized = root / ".cache/scn/corpus/analyzed.jsonl"
    split_index = []
    with normalized.open("w") as stream:
        for row in rows:
            group = groups.find(row["document_id"])
            bucket = int(hashlib.sha256(group.encode()).hexdigest(), 16) % 10
            split = "development" if bucket == 0 else "test" if bucket == 1 else "train"
            row.update(split_group=group, split=split)
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            split_index.append(
                {
                    k: row[k]
                    for k in (
                        "id",
                        "document_id",
                        "source_id",
                        "orthography",
                        "split_group",
                        "split",
                    )
                }
            )
    out = root / "src/asr/scn/resources/corpus"
    write_json(
        out / "statistics.json",
        {
            "rules": RULES,
            "records_sha256": extraction["records_sha256"],
            "analyzed_sha256": fingerprints(normalized)["sha256"],
            "total": summary(rows),
            "text_only": summary([r for r in rows if r["data_role"] == "text"]),
            "categories": {
                c: summary(
                    [
                        r
                        for r in rows
                        if r["orthography"] == c and r["data_role"] == "text"
                    ]
                )
                for c in CATEGORIES
            },
            "sources": {
                s: summary([r for r in rows if r["source_id"] == s])
                for s in sorted({r["source_id"] for r in rows})
            },
            "near_duplicate_pairs": len(pairs),
            "near_duplicate_method": "16 deterministic MinHash values, 4 bands; verify 5-token Jaccard >=0.85; >=10 tokens; candidate search is not exhaustive",
            "sentence_method": "Source sentence IDs when available; otherwise punctuation/newline heuristic; UI fragments and lexical entries counted separately",
            "language_screen_method": "Versioned lexical markers identify review candidates only, never verified language labels",
        },
    )
    write_json(
        root / ".cache/scn/corpus/duplicates.json",
        {
            "exact": [
                {"id": r["id"], "duplicate_of": r["duplicate_of"]}
                for r in rows
                if r.get("duplicate_of")
            ],
            "near": pairs,
        },
    )
    # Full document and split indexes can be large, so remain derived, ignored artifacts.
    write_json(root / ".cache/scn/corpus/splits.json", {"records": split_index})
    review = []
    for source in sorted({r["source_id"] for r in rows}):
        subset = sorted(
            (r for r in rows if r["source_id"] == source), key=lambda r: r["id"]
        )
        chosen = {r["id"]: r for r in subset[:100]}
        chosen.update(
            {
                r["id"]: r
                for r in subset
                if r["language_screen"]["screen"] != "unresolved"
            }
        )
        review += [
            {
                "id": r["id"],
                "source_id": source,
                "text": r["text"],
                "screen": r["language_screen"],
                "review_status": "pending",
            }
            for r in chosen.values()
        ]
    write_json(root / ".cache/scn/corpus/review-queue.json", {"records": review})
    print(
        json.dumps(
            {
                "text": summary([r for r in rows if r["data_role"] == "text"]),
                "near_pairs": len(pairs),
            },
            indent=2,
        )
    )


def inventory_ipa(root):
    from ipa.inventory import inventory

    base = root / ".cache/models/zipa-cr-ns-large-800k"
    labels, tokenizer = inventory(base / "tokens.txt", base / "unigram_127.model")
    required = "a ɛ i ɔ u e o ɪ ʊ p b t d ɖ ʈ c ɟ k ɡ g f v s z ʃ ʒ ç ʝ x χ r ɾ ɽ ʂ ʐ m n ɲ ŋ ɳ ɱ l ʎ j w h ɣ β ð ː ˈ ˌ ͡ ̪ ̠ ̝ ̯ tʃ dʒ ts dz ʈɽ ɖɽ".split()
    rows = []
    for symbol in required:
        ids = tokenizer.encode(symbol, out_type=int)
        raw = ids[1:] if ids and ids[0] == 3 else ids
        rows.append(
            {
                "ipa": symbol,
                "token_ids": raw,
                "lossless": bool(raw)
                and all(i >= 4 for i in raw)
                and "".join(labels[i] for i in raw) == symbol,
                "projection": "g"
                if symbol == "ɡ"
                else "metadata_only"
                if symbol in {"ˈ", "ˌ"}
                else "requires_review"
                if 2 in raw
                else "identity",
            }
        )
    catalog = load_catalog(root)
    evidence = {}
    for name, spec in catalog["sources"].items():
        if (
            spec.get("extractor") != "ipa_tsv"
            or spec.get("acquisition_status") != "acquired"
        ):
            continue
        path = local_path(root, spec["path"])
        verify_file(path, spec)
        pairs = [line.split("\t") for line in path.read_text().splitlines()]
        unsupported = Counter()
        supported = 0
        for word, ipa in pairs:
            text = unicodedata.normalize("NFC", "".join(ipa.split())).replace("ɡ", "g")
            ids = tokenizer.encode(text, out_type=int)
            if ids and ids[0] == 3:
                ids = ids[1:]
            if (
                ids
                and all(i >= 4 for i in ids)
                and "".join(labels[i] for i in ids) == text
            ):
                supported += 1
            else:
                unsupported.update(c for c in text if c not in labels)
        evidence[name] = {
            "rows": len(pairs),
            "word_forms": len({w for w, _ in pairs}),
            "representable_after_g_projection": supported,
            "unsupported_characters": dict(unsupported),
            "validated_authoritative_pairs": 0,
        }
    write_json(
        root / "src/asr/scn/resources/lexicon/ipa-compatibility.json",
        {
            "labels_sha256": fingerprints(base / "tokens.txt")["sha256"],
            "candidate_inventory": rows,
            "source_lexicons": evidence,
            "acoustic_accuracy_validated": False,
            "scope": "Candidate broad and regional phones; review cited linguistic descriptions before implementing pronunciation rules.",
        },
    )
