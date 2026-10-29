"""Pinned lexical analyses and sentence-preserving DOS24 corpus admission."""

import json
import hashlib
import re
import tarfile
from collections import Counter, defaultdict

from asr.scn.dataset import digest, passage_hashes
from asr.scn.dos24 import Converter, key, rule_manifest
from asr.scn.orthography import normalize
from common.artifacts import read, seal, sha, verified_files
from common.errors import ResourceError
from common.utils import write_json


def build_index(root, bundle, output):
    from rdflib import RDF, RDFS, Graph, Literal, Namespace

    catalog = read(root / "resources/scn/sources/catalog.json")["sources"]
    names = ("scn_liita_lemmabank", "scn_ud_stb")
    identities = {}
    for name in names:
        spec = catalog[name]
        identities[name] = sha(root / spec["path"])
        if identities[name] != spec["sha256"]:
            raise ResourceError(f"Lexical source changed: {name}")
    identity = {
        "sources": identities,
        "channel": sha(bundle / "channel.json"),
        "implementation": sha(__file__),
    }
    if (output / "index.receipt.json").exists():
        receipt = read(output / "index.receipt.json")
        if receipt["key"] != digest(identity):
            raise ResourceError("Lexical index inputs changed")
        verified_files(output, receipt)
        return read(output / "index.json")["analyses"]
    index = defaultdict(list)
    graph = Graph().parse(root / catalog[names[0]]["path"], format="turtle")
    ontolex = Namespace("http://www.w3.org/ns/lemon/ontolex#")
    lila = Namespace("http://lila-erc.eu/ontologies/lila/")
    pos = {
        "noun": "NOUN",
        "adjective": "ADJ",
        "verb": "VERB",
        "adverb": "ADV",
        "pronoun": "PRON",
        "conjunction": "CCONJ",
        "preposition": "ADP",
        "interjection": "INTJ",
    }
    for subject in sorted(set(graph.subjects(RDF.type, lila.Lemma)), key=str):
        lemma = str(graph.value(subject, RDFS.label) or "")
        lexical_pos = str(graph.value(subject, lila.hasPOS) or "").rsplit("/", 1)[-1]
        features = {
            str(p).rsplit("/", 1)[-1]: str(o).rsplit("/", 1)[-1]
            for p, o in graph.predicate_objects(subject)
            if str(p).startswith(str(lila)) and p != lila.lemmaVariant
        }
        variants = sorted(str(o) for o in graph.objects(subject, lila.lemmaVariant))
        for form in sorted(set(graph.objects(subject, ontolex.writtenRep)), key=str):
            if (
                isinstance(form, Literal)
                and form.language == "scn"
                and " " not in str(form)
            ):
                index[key(str(form))].append(
                    {
                        "lemma": lemma,
                        "pos": pos.get(lexical_pos, "X"),
                        "features": features,
                        "variant_uris": variants,
                        "source": names[0],
                        "evidence": str(subject),
                        "source_sha256": identities[names[0]],
                        "surface": str(form),
                    }
                )
    del graph
    with tarfile.open(root / catalog[names[1]]["path"]) as archive:
        for member in sorted(archive.getmembers(), key=lambda m: m.name):
            if not member.name.endswith(".conllu"):
                continue
            document = sentence = ""
            for line in archive.extractfile(member).read().decode().splitlines():
                if line.startswith("# newdoc id = "):
                    document = line.split(" = ", 1)[1]
                if line.startswith("# sent_id = "):
                    sentence = line.split(" = ", 1)[1]
                fields = line.split("\t")
                if len(fields) != 10 or not fields[0].isdigit() or fields[3] == "PUNCT":
                    continue
                features = dict(
                    p.split("=", 1) for p in fields[5].split("|") if "=" in p
                )
                row = {
                    "lemma": fields[2],
                    "pos": fields[3],
                    "features": features,
                    "source": names[1],
                    "source_sha256": identities[names[1]],
                    "surface": fields[1],
                    "evidence": f"{member.name}:{document}:{sentence}:{fields[0]}",
                    "dependency": {"head": fields[6], "relation": fields[7]},
                }
                index[key(fields[1])].append(row)
    # Channel evidence supplements lexical analyses. It is neither a whitelist
    # nor permission to promote contextual corrections to universal mappings.
    for record in read(bundle / "channel.json")["recordings"]:
        if record["split"] != "train":
            continue
        for word in sorted(set(normalize(record["text"]).split())):
            index[word].append(
                {
                    "lemma": word,
                    "pos": "X",
                    "features": {},
                    "source": "channel-training",
                    "evidence": record["recording_id"],
                    "source_sha256": record["source_sha256"],
                }
            )
    # Compact repeated UD token analyses while preserving all provenance locators.
    compact = {}
    for word, rows in sorted(index.items()):
        unique = {}
        for row in rows:
            ident = digest(
                {k: v for k, v in row.items() if k not in {"evidence", "dependency"}}
            )
            if ident in unique:
                unique[ident].setdefault("additional_evidence", []).append(
                    row["evidence"]
                )
            else:
                unique[ident] = row
        compact[word] = list(unique.values())
    write_json(
        output / "index.json",
        {
            "identity": identity,
            "analyses": compact,
            "policy": "Analyses and variant relations only; canonical labels never authorize spelling replacement.",
        },
    )
    seal(output, "index.receipt.json", identity, ["index.json"])
    return compact


def sentence_spans(text):
    """Retain offsets and original boundaries; excluded spans never join."""
    for match in re.finditer(r"\S[\s\S]*?(?:[.!?](?=\s|$)|$)", text):
        if match.group().strip():
            yield match.start(), match.end()


class DuplicateFilter:
    """Exact and verified 5-shingle Jaccard dedup; deterministic inverted index."""

    def __init__(self, threshold=0.9):
        self.threshold, self.exact, self.rows, self.postings = (
            threshold,
            {},
            [],
            defaultdict(list),
        )

    def check(self, text, identifier, *, add=True):
        normalized = normalize(text)
        if normalized in self.exact:
            return {
                "reason": "exact_duplicate",
                "representative": self.exact[normalized],
            }
        words = normalized.split()
        shingles = {tuple(words[i : i + 5]) for i in range(len(words) - 4)}
        possible = set()
        for shingle in shingles:
            possible.update(self.postings.get(shingle, []))
        for i in sorted(possible):
            other_id, other = self.rows[i]
            if len(shingles & other) / len(shingles | other) >= self.threshold:
                return {"reason": "near_duplicate", "representative": other_id}
        if add:
            self.exact[normalized] = identifier
            if shingles:
                idx = len(self.rows)
                self.rows.append((identifier, shingles))
                for shingle in shingles:
                    self.postings[shingle].append(idx)
        return None


def leakage(text, forbidden, exact):
    value = normalize(text)
    return value in exact or bool(passage_hashes(value.split()) & forbidden)


def prepare_corpus(root, bundle, output, settings):
    output.mkdir(parents=True, exist_ok=True)
    index = build_index(root, bundle, output)
    converter = Converter(index)
    sources = read(root / "resources/scn/sources/catalog.json")["sources"]
    source = sources[settings["source_id"]]
    if sha(root / source["path"]) != source["sha256"]:
        raise ResourceError("Pinned Wikipedia dump changed")
    records_path = root / settings["records"]
    identity = {
        "source_id": settings["source_id"],
        "source_sha256": source["sha256"],
        "extraction_sha256": sha(records_path),
        "index_sha256": sha(output / "index.json"),
        "rules": rule_manifest(),
        "implementation": sha(__file__),
        "converter_implementation": sha(root / "src/asr/scn/dos24.py"),
        "development": sha(bundle / "recordings.json"),
        "settings": settings,
    }
    if (output / "corpus.receipt.json").exists():
        saved = read(output / "corpus.receipt.json")
        if saved["key"] != digest(identity):
            raise ResourceError("Corpus inputs changed; use a new experiment output")
        verified_files(output, saved)
        return read(output / "summary.json")
    dev = [r["reference"] for r in read(bundle / "recordings.json")]
    forbidden = set().union(*(passage_hashes(normalize(t).split()) for t in dev))
    exact = {normalize(t) for t in dev}
    channel = read(bundle / "channel.json")
    channel_words = {
        w
        for r in channel["recordings"]
        if r["split"] == "train"
        for w in normalize(r["text"]).split()
    }
    channel_text = [normalize(r["text"]) for r in channel["training"]]
    channel_ngrams = {
        n: {
            tuple(ws[i : i + n])
            for t in channel_text
            for ws in [t.split()]
            for i in range(len(ws) - n + 1)
        }
        for n in (2, 3)
    }
    duplicates = [
        DuplicateFilter(settings.get("near_duplicate_threshold", 0.9)) for _ in range(2)
    ]
    counts, reasons, rules, words, articles = (
        Counter(),
        Counter(),
        Counter(),
        set(),
        set(),
    )
    additional_ngrams = {2: set(), 3: set()}
    collisions, article_support, examples = defaultdict(set), defaultdict(set), []
    paths = {
        name: output / name
        for name in (
            "corpus.jsonl",
            "review-queue.jsonl",
            "admitted.jsonl",
            "channel-preservation.jsonl",
        )
    }
    streams = {name: path.open("w") for name, path in paths.items()}

    def emit(name, row):
        if "analyses" in row:
            row = {**row, "analyses": [
                {**token, "analyses": [
                    ({"index_reference": a["index_reference"], **{k: v for k, v in a.items() if k in {"derivation", "elision"}}}
                     if "index_reference" in a else a)
                    for a in token["analyses"]]}
                for token in row["analyses"]]}
        streams[name].write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    try:
        for rec in channel["training"]:
            converted = converter.convert(rec["text"])
            emit("channel-preservation.jsonl", {"id": rec["id"], **converted})
            counts["channel_sentences"] += 1
            counts["channel_changed"] += bool(converted["edits"])
            counts["channel_unresolved"] += bool(converted["unresolved"])
        with records_path.open() as stream:
            for line in stream:
                record = json.loads(line)
                if record["source_id"] != settings["source_id"]:
                    continue
                if (
                    record["source_sha256"] != source["sha256"]
                        or hashlib.sha256(record["text"].encode("utf-8")).hexdigest() != record["text_sha256"]
                ):
                    raise ResourceError("Immutable source record checksum mismatch")
                for start, end in sentence_spans(record["text"]):
                    original = record["text"][start:end]
                    identifier = record["id"] + f":{start}-{end}"
                    counts["sentences"] += 1
                    converted = converter.convert(original)
                    target = converted["converted_text"]
                    exclusion = None
                    if record["unit_kind"] != "paragraph" or record["metadata"].get(
                        "exclusion"
                    ):
                        exclusion = {"reason": "nonprose"}
                    elif not re.search(r"[.!?][\"»”]?\s*$", original):
                        exclusion = {"reason": "unterminated_or_nonprose_fragment"}
                    elif leakage(original, forbidden, exact) or leakage(
                        target, forbidden, exact
                    ):
                        exclusion = {"reason": "development_passage_overlap"}
                    elif len(normalize(original).split()) < settings.get(
                        "minimum_words", 4
                    ):
                        exclusion = {"reason": "short_fragment"}
                    elif converted["unresolved"]:
                        exclusion = {"reason": "unresolved_analysis"}
                    else:
                        # Dedup only admissible records: an unresolved first copy
                        # must not suppress a later independently supported one.
                        for filt, text in zip(duplicates, (original, target)):
                            exclusion = filt.check(text, identifier, add=False)
                            if exclusion:
                                break
                    row = {
                        "id": identifier,
                        "source_record_id": record["id"],
                        "document_id": record["document_id"],
                        "source_sha256": record["source_sha256"],
                        "source_record_sha256": digest(record),
                        "paragraph_span": [start, end],
                        "metadata": record["metadata"],
                        "license": source["license"],
                        "attribution": record["metadata"].get("url"),
                        **converted,
                        "lm_eligible": exclusion is None,
                        "exclusion": exclusion,
                    }
                    emit("corpus.jsonl", row)
                    for edit in converted["edits"]:
                        rules.update(edit["rules"])
                        collisions[edit["replacement"]].add(edit["original"])
                        if (
                            key(edit["replacement"]) not in channel_words
                            and len(examples) < 30
                        ):
                            examples.append({"id": identifier, **edit})
                    if exclusion:
                        reasons[exclusion["reason"]] += 1
                        if converted["unresolved"]:
                            emit("review-queue.jsonl", row)
                            reasons.update(
                                "unresolved:" + x["reason"]
                                for x in converted["unresolved"]
                            )
                        continue
                    for filt, text in zip(duplicates, (original, target)):
                        filt.check(text, identifier)
                    raw, text = normalize(original), normalize(target)
                    emit(
                        "admitted.jsonl",
                        {
                            "id": identifier,
                            "document_id": record["document_id"],
                            "raw": raw,
                            "converted": text,
                        },
                    )
                    counts["admitted_sentences"] += 1
                    counts["admitted_tokens"] += len(text.split())
                    counts["admitted_changed"] += bool(converted["edits"])
                    counts["admitted_unchanged"] += not converted["edits"]
                    words.update(text.split())
                    articles.add(record["document_id"])
                    for word in set(text.split()):
                        article_support[word].add(record["document_id"])
                    ws = text.split()
                    for n in (2, 3):
                        additional_ngrams[n].update(
                            tuple(ws[i : i + n]) for i in range(len(ws) - n + 1)
                        )
    finally:
        for stream in streams.values():
            stream.close()
    replacements = read(root / "resources/scn/poc_reviews.json").get("replacements", {})
    write_json(
        output / "replacement-audit.json",
        {
            "policy": "No previous replacement dictionary imported into the converter.",
            "entries": [
                {"form": word, **value, "disposition": "context_only_not_reused"}
                for word, value in sorted(replacements.items())
            ],
        },
    )
    summary = {
        "counts": dict(counts),
        "exclusions": dict(reasons),
        "edit_rules": dict(rules),
        "unique_articles": len(articles),
        "unique_words": len(words),
        "additional_words": len(words - channel_words),
        "additional_ngrams": {
            str(n): len(v - channel_ngrams[n]) for n, v in additional_ngrams.items()
        },
        "conversion_collisions": {
            k: sorted(v) for k, v in collisions.items() if len(v) > 1
        },
        "productive_examples_absent_from_subtitles": examples,
        "correctness": "Coverage of automated support, not an estimate of conversion accuracy or human certification.",
    }
    write_json(output / "summary.json", summary)
    write_json(
        output / "article-support.json",
        {w: sorted(v) for w, v in sorted(article_support.items())},
    )
    write_json(output / "rules.json", rule_manifest())
    seal(
        output,
        "corpus.receipt.json",
        identity,
        [
            *paths,
            "summary.json",
            "article-support.json",
            "rules.json",
            "replacement-audit.json",
        ],
    )
    return summary
