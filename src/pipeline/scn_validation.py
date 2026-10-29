"""Cross-stage validation of a completed DOS24 experiment and its portable result."""

import json
from collections import defaultdict
from pathlib import Path

from asr.scn.dataset import passage_hashes
from asr.scn.orthography import normalize
from asr.wfst.arpa import read_unigrams
from common.artifacts import digest, read, sha, verified_files
from common.errors import ResourceError
from common.utils import write_json
from pipeline.experiment import SPECIAL, aggregate


def require(condition, message):
    if not condition:
        raise ResourceError(message)


def validate(root, bundle, output, recipe):
    verified_files(bundle, read(bundle / "bundle.json"))
    verified_files(output / "review", read(output / "review/review.receipt.json"))
    for name in ("index.receipt.json", "corpus.receipt.json"):
        verified_files(output / "corpus", read(output / "corpus" / name))
    recordings = read(bundle / "recordings.json")
    videos = {r["video"] for r in recordings}
    forbidden = set().union(
        *(passage_hashes(normalize(r["reference"]).split()) for r in recordings)
    )
    exact = {normalize(r["reference"]) for r in recordings}
    admitted = [json.loads(line) for line in (output / "corpus/admitted.jsonl").open()]
    indexed = {r["id"]: r for r in admitted}
    require(len(indexed) == len(admitted), "Repeated admitted sentence identity")
    found, article_support, source_spans = set(), defaultdict(set), defaultdict(list)
    with (output / "corpus/corpus.jsonl").open() as stream:
        for line in stream:
            row = json.loads(line)
            if row["id"] not in indexed:
                require(
                    not row["lm_eligible"], "Eligible sentence absent from LM corpus"
                )
                continue
            require(
                row["lm_eligible"] and not row["unresolved"] and not row["exclusion"],
                "Unsupported sentence in admitted corpus",
            )
            paired = indexed[row["id"]]
            require(
                paired["document_id"] == row["document_id"], "Article identity changed"
            )
            for name, text_key in (
                ("raw", "original_text"),
                ("converted", "converted_text"),
            ):
                text = normalize(row[text_key])
                require(paired[name] == text, "Matched passage content changed")
                require(
                    text not in exact and not passage_hashes(text.split()) & forbidden,
                    "Development passage leaked into LM corpus",
                )
            reconstructed, cursor = [], 0
            for edit in row["edits"]:
                start, end = edit["start"], edit["end"]
                require(
                    cursor <= start < end
                    and row["original_text"][start:end] == edit["original"],
                    "Invalid conversion edit span",
                )
                reconstructed.extend(
                    (row["original_text"][cursor:start], edit["replacement"])
                )
                cursor = end
            reconstructed.append(row["original_text"][cursor:])
            require(
                "".join(reconstructed) == row["converted_text"],
                "Edit trace does not reproduce converted text",
            )
            source_spans[row["source_record_id"]].append(row)
            for word in set(paired["converted"].split()):
                article_support[word].add(row["document_id"])
            found.add(row["id"])
    require(found == set(indexed), "Missing admitted provenance record")
    found_sources = set()
    with (root / recipe["corpus"]["records"]).open() as stream:
        for line in stream:
            source = json.loads(line)
            if source["id"] not in source_spans:
                continue
            for row in source_spans[source["id"]]:
                start, end = row["paragraph_span"]
                require(
                    source["text"][start:end] == row["original_text"]
                    and digest(source) == row["source_record_sha256"],
                    "Source span/provenance mismatch",
                )
            found_sources.add(source["id"])
    require(found_sources == set(source_spans), "Missing immutable source record")
    require(
        {w: sorted(v) for w, v in article_support.items()}
        == read(output / "corpus/article-support.json"),
        "Article support inventory mismatch",
    )

    study = read(output / "lm/models.json")
    require(
        study["status"] == "complete" and len(study["models"]) == 14,
        "Fourteen completed LMs required",
    )
    vocabulary = set(read(bundle / "graph/lexicon_selected.json")) - SPECIAL
    models = {m["id"]: m for m in study["models"]}
    for model in models.values():
        directory = Path(model["directory"])
        receipt = read(directory / "model.json")
        verified_files(directory, receipt)
        require(
            sha(directory / "model.json") == model["receipt_sha256"],
            "LM receipt changed",
        )
        require(
            set(read_unigrams(directory / "model.arpa")) - SPECIAL == vocabulary,
            "Fixed vocabulary changed",
        )
        require(
            receipt["identity"]["prune"] == study["shared_prune"],
            "Unmatched pruning policy",
        )
        require(
            sum(v for k, v in model["ngram_counts"].items() if int(k) > 1)
            <= recipe["lm"]["maximum_context_ngrams"],
            "N-gram budget exceeded",
        )
        if model["kind"] == "channel":
            continue
        require(
            receipt["identity"]["wiki"] == digest([r[model["kind"]] for r in admitted]),
            "LM passage content mismatch",
        )
        require(
            model["source_passage_ids_sha256"] == digest([r["id"] for r in admitted]),
            "LM source selection mismatch",
        )
        require(
            abs(model["weighting"]["effective_wikipedia_share"] - model["share"])
            < 0.001,
            "Effective mixture share outside tolerance",
        )
        if model["kind"] == "converted":
            raw = models[model["id"].replace("converted", "raw")]
            require(
                model["weighting"] == raw["weighting"],
                "Raw and converted mixture weights differ",
            )

    lm = read(output / "lm-results.json")
    pron = read(output / "pronunciation-results.json")
    vocab = read(output / "vocabulary-results.json")
    selection = read(output / "selection.json")
    require(
        all(r["status"] == "complete" for r in (lm, pron, vocab, selection)),
        "Incomplete experiment stage",
    )
    require(
        len(lm["results"]) == 15 and len(pron["results"]) == 3,
        "Incomplete controlled decode matrix",
    )
    rows = (
        lm["results"]
        + pron["results"]
        + [vocab["result"]]
        + selection["combined"]
        + selection["retuned"]
    )
    base_lexicon = read(bundle / "graph/lexicon_selected.json")
    base_tokens = read(bundle / "graph/compile_input.json")["tokens"]
    for row in rows:
        graph = Path(row["graph"])
        verified_files(graph, read(graph / "graph.json"))
        require(
            read(graph / "compile_input.json")["tokens"] == base_tokens,
            "Acoustic inventory changed",
        )
        if row in lm["results"] or row in pron["results"]:
            require(
                read(graph / "lexicon_selected.json") == base_lexicon,
                "Controlled pronunciation paths or vocabulary changed",
            )
        result_path = Path(row["result"])
        verified_files(
            result_path.parent, read(result_path.parent / "result.receipt.json")
        )
        result = read(result_path)
        require(
            {r["video"] for r in result["recordings"]} == videos
            and len(result["recordings"]) == len(videos),
            "Partial recording evaluation",
        )
        require(
            aggregate(result["recordings"]) == row["metrics"] == result["metrics"],
            "Aggregate scores do not match recording errors",
        )
    analysis = read(output / "pronunciation/analysis.json")
    require(len(analysis["folds"]) == 6, "Six pronunciation validation folds required")
    for fold in analysis["folds"]:
        require(
            set(fold["training_recordings"])
            == set(analysis["recordings"]) - {fold["heldout"]},
            "Pronunciation fold leakage",
        )
    verified_files(output / "best", read(output / "best/candidate.json"))
    require(
        read(output / "best/vocabulary.json")["words"]
        == sorted(read(output / "best/graph/lexicon_selected.json")),
        "Portable candidate vocabulary mismatch",
    )
    replay = read(output / "selected-fresh-replay/result.json")
    verified_files(
        output / "selected-fresh-replay",
        read(output / "selected-fresh-replay/result.receipt.json"),
    )
    require(
        replay["metrics"] == selection["selected"]["metrics"],
        "Selected fresh replay mismatch",
    )
    require(
        read(output / "baseline-replay/verification.json")["status"] == "verified",
        "Baseline independence not verified",
    )
    require(
        read(output / "cleanup/cleanup.json")["status"] == "complete",
        "Cleanup incomplete",
    )
    summary = {
        "status": "verified",
        "admitted_sentences": len(admitted),
        "source_records": len(found_sources),
        "language_models": len(models),
        "complete_recording_candidates": len(rows),
        "recordings_per_candidate": len(videos),
        "pronunciation_folds": len(analysis["folds"]),
        "checks": [
            "immutable_source_spans",
            "traceable_conversion_edits",
            "no_excluded_span_adjacency",
            "development_passage_exclusion",
            "article_support",
            "matched_lm_passages_and_weights",
            "common_vocabulary_and_pruning",
            "frozen_acoustic_inventory",
            "complete_recording_scores",
            "training_only_folds",
            "portable_candidate",
            "baseline_replay",
            "cleanup_receipts",
        ],
        "inputs": {
            str(p.relative_to(output)): sha(p)
            for p in (
                output / "study.json",
                output / "corpus/corpus.receipt.json",
                output / "lm/models.json",
                output / "lm-results.json",
                output / "pronunciation-results.json",
                output / "vocabulary-results.json",
                output / "selection.json",
                output / "best/candidate.json",
                output / "baseline-replay/verification.json",
                output / "cleanup/cleanup.json",
            )
        },
        "implementation_sha256": sha(__file__),
    }
    write_json(output / "validation.json", summary)
    return summary
