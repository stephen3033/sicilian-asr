"""Rank unresolved linguistic facts for inspection without changing admission."""

import json
from collections import Counter

from asr.scn.orthography import normalize
from common.artifacts import digest, read, seal, sha, verified_files


def ranked_candidates(candidates, frequency):
    return sorted(
        (
            {"text": text, "channel_training_count": frequency[normalize(text)]}
            for text in candidates
        ),
        key=lambda row: (-row["channel_training_count"], row["text"]),
    )


def prioritize(bundle, corpus, output):
    identity = {
        "queue": sha(corpus / "review-queue.jsonl"),
        "channel": sha(bundle / "channel.json"),
        "implementation": sha(__file__),
    }
    receipt = output / "review.receipt.json"
    if receipt.exists() and read(receipt)["key"] == digest(identity):
        verified_files(output, read(receipt))
        return
    frequency = Counter(
        word
        for recording in read(bundle / "channel.json")["recordings"]
        if recording["split"] == "train"
        for word in normalize(recording["text"]).split()
    )
    facts = {}
    with (corpus / "review-queue.jsonl").open() as stream:
        for line in stream:
            sentence = json.loads(line)
            for issue in sentence["unresolved"]:
                identifier = digest(
                    {key: value for key, value in issue.items() if key != "span"}
                )
                if identifier not in facts:
                    candidates = ranked_candidates(
                        issue.get("candidates", []), frequency
                    )
                    facts[identifier] = {
                        "id": identifier,
                        "original": issue.get("original"),
                        "reason": issue["reason"],
                        "missing": issue["missing"],
                        "ranked_candidates": candidates,
                        "occurrences": 0,
                        "examples": [],
                        "priority_count": max(
                            [
                                frequency[normalize(issue.get("original", ""))],
                                *[r["channel_training_count"] for r in candidates],
                            ]
                        ),
                        "resolution": "unresolved",
                        "policy": "Frequency determines inspection order only; it does not establish spelling or admission.",
                    }
                fact = facts[identifier]
                fact["occurrences"] += 1
                if len(fact["examples"]) < 3:
                    fact["examples"].append(
                        {"sentence_id": sentence["id"], "span": issue["span"]}
                    )
    output.mkdir(parents=True, exist_ok=True)
    with (output / "priority.jsonl").open("w") as stream:
        for fact in sorted(
            facts.values(),
            key=lambda row: (-row["priority_count"], -row["occurrences"], row["id"]),
        ):
            stream.write(json.dumps(fact, ensure_ascii=False, sort_keys=True) + "\n")
    seal(output, receipt.name, identity, ["priority.jsonl"])
