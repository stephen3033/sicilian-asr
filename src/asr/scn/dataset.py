"""Build a reviewed corpus; held-out text supplies only overlap fingerprints."""

import hashlib
import json
import re

from asr.scn.orthography import RULES, corrected
from common.errors import ResourceError
from common.settings import SCN_RESOURCES
from common.utils import fingerprints, write_json


def digest(value):
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True).encode()
    ).hexdigest()


def read_jsonl(path):
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_jsonl(path, rows):
    path.write_text(
        "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in rows)
    )


def reviews(root):
    path = root / SCN_RESOURCES / "poc_reviews.json"
    return (
        json.loads(path.read_text())
        if path.exists()
        else {"decisions": {}, "replacements": {}}
    )


def reviewed_text(record, review):
    decision = review.get("decisions", {}).get(record["id"])
    if not decision or decision.get("decision") != "include":
        return None
    if (
        decision.get("text_sha256")
        != hashlib.sha256(record["text"].encode()).hexdigest()
    ):
        raise ResourceError(f"Stale contextual review: {record['id']}")
    if not decision.get("reason") or not decision.get("reviewer"):
        raise ResourceError(f"Missing contextual review evidence: {record['id']}")
    raw = apply_review(record, decision)
    text, changes = corrected(raw, review.get("replacements", {}))
    return text, changes


def apply_review(record, decision):
    raw = decision.get("corrected_text", record["text"])
    if "cues" in record and decision.get("excluded_cues"):
        raw = " ".join(
            c["text"]
            for i, c in enumerate(record["cues"])
            if i not in decision["excluded_cues"]
        )
    for edit in decision.get("edits", []):
        pattern = re.escape(edit["old"])
        if edit.get("mode") == "word":
            pattern = r"(?<!\w)" + pattern + r"(?!\w)"
        raw, count = re.subn(pattern, lambda _: edit["new"], raw)
        if not count:
            raise ResourceError(f"Stale text correction: {record['id']}: {edit['old']}")
    return raw


def video_record(root, video):
    path = root / ".cache/scn/youtube" / video
    receipt = json.loads((path / "receipt.json").read_text())
    for name, expected in receipt["artifacts"].items():
        if fingerprints(path / name)["sha256"] != expected:
            raise ResourceError(f"Changed acquired artifact: {video}/{name}")
    cues = json.loads((path / "cues.json").read_text())
    return {
        "id": "youtube:" + video,
        "source_id": "youtube:" + video,
        "document_id": video,
        "text": " ".join(c["text"] for c in cues),
        "cues": cues,
        "source_sha256": receipt["artifacts"][receipt["caption"]],
    }


def sentences(text):
    # Newlines within captions are not sentence boundaries.
    return [
        part.strip()
        for part in re.split(r"(?<=[.!?])\s+", text.replace("\n", " "))
        if part.strip()
    ]


def passage_hashes(words, width=12):
    return {digest(words[i : i + width]) for i in range(max(0, len(words) - width + 1))}


def test_overlap_fingerprints(root, videos, replacements):
    """Use held-out captions only as a fixed duplicate denylist, never as LM text."""
    hashes, sources = set(), {}
    for video in videos:
        record = video_record(root, video)
        text, _ = corrected(record["text"], replacements)
        hashes.update(passage_hashes(text.split()))
        sources[video] = record["source_sha256"]
    return hashes, sources


def prepare_corpus(root, output, recipe):
    review = reviews(root)
    split_sets = [set(v) for v in recipe["splits"].values()]
    if sum(map(len, split_sets)) != len(set.union(*split_sets)):
        raise ResourceError("Video partitions overlap")
    approved, pending, development = [], [], []
    # Final references are not reviewed, token-counted or supplied to the trainer.
    for split in ("train", "development"):
        for video in recipe["splits"][split]:
            record = video_record(root, video)
            value = reviewed_text(record, review)
            if value is None:
                pending.append(
                    {
                        **record,
                        "text_sha256": hashlib.sha256(
                            record["text"].encode()
                        ).hexdigest(),
                    }
                )
                continue
            text, changes = value
            if split == "development":
                development.append(
                    {**record, "normalized": text, "corrections": changes}
                )
                continue
            decision = review["decisions"][record["id"]]
            raw = apply_review(record, decision)
            for i, sentence in enumerate(sentences(raw)):
                normalized, edits = corrected(sentence, review.get("replacements", {}))
                if normalized:
                    approved.append(
                        {
                            "id": f"{record['id']}:{i}",
                            "source_id": record["source_id"],
                            "document_id": video,
                            "text": normalized,
                            "tokens": normalized.split(),
                            "corrections": edits,
                            "cohort": "channel",
                            "source_sha256": record["source_sha256"],
                        }
                    )
    screened = root / ".cache/scn/audit/screened.jsonl"
    allowed_sources = set(recipe["preferred_sources"] + recipe["expansion_sources"])
    for record in read_jsonl(screened):
        if (
            record.get("data_role") != "text"
            or record["source_id"] not in allowed_sources
        ):
            continue
        if record.get("metadata", {}).get("split") in {"test", "dev", "validation"}:
            continue
        value = reviewed_text(record, review)
        if value is None:
            if record["category"] in {"supported_inferred", "verified_conforming"}:
                pending.append(record)
            continue
        normalized, changes = value
        for i, sentence in enumerate(
            sentences(apply_review(record, review["decisions"][record["id"]]))
        ):
            text, edits = corrected(sentence, review.get("replacements", {}))
            if text:
                approved.append(
                    {
                        "id": f"{record['id']}:{i}",
                        "source_id": record["source_id"],
                        "document_id": record["document_id"],
                        "text": text,
                        "tokens": text.split(),
                        "corrections": edits,
                        "cohort": "background",
                        "source_sha256": record["source_sha256"],
                    }
                )
    # Exact sentence duplicates and long shared passages in development cannot train.
    dev_words = [r["normalized"].split() for r in development]
    dev_shingles = set().union(*(passage_hashes(words) for words in dev_words))
    test_shingles, held_out_sources = test_overlap_fingerprints(
        root, recipe["splits"]["test"], review.get("replacements", {})
    )
    seen, retained, exclusions = set(), [], []
    for row in approved:
        words = row["tokens"]
        reason = None
        if row["text"] in seen:
            reason = "duplicate_sentence"
        elif len(words) > 250:
            reason = "unresolved_sentence_boundary"
        elif passage_hashes(words) & dev_shingles:
            reason = "development_passage_overlap"
        elif passage_hashes(words) & test_shingles:
            reason = "held_out_passage_overlap"
        if reason:
            exclusions.append({"id": row["id"], "reason": reason})
        else:
            seen.add(row["text"])
            retained.append(row)
    write_jsonl(output / "review-queue.jsonl", pending)
    write_jsonl(output / "training.jsonl", retained)
    write_json(output / "development.json", development)
    receipt = {
        "normalization": RULES,
        "recipe": recipe,
        "reviews_sha256": digest(review),
        "training_sha256": fingerprints(output / "training.jsonl")["sha256"],
        "development_sha256": fingerprints(output / "development.json")["sha256"],
        "records": len(retained),
        "tokens": sum(len(r["tokens"]) for r in retained),
        "pending_review_records": len(pending),
        "held_out_duplicate_screen": {
            "policy": "Remove training sentences sharing 12 normalized consecutive words with held-out captions; expose only fingerprints, never vocabulary or reference text to training.",
            "source_sha256": held_out_sources,
        },
        "exclusions": exclusions,
    }
    write_json(output / "corpus.json", receipt)
    if not retained or len(development) != len(recipe["splits"]["development"]):
        raise ResourceError(
            "Training/development reviews incomplete; inspect review-queue.jsonl and record contextual decisions"
        )
    return receipt
