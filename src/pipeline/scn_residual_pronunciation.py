"""Training-only pronunciation coverage and recurrence evidence for residuals."""

from collections import Counter, defaultdict

from common.artifacts import digest, read, sha
from common.errors import ResourceError
from common.utils import write_json
from pipeline.scn_pronunciation import analyze
from pipeline.scn_word_labels import recurring_word_labels


def audit(bundle, output, recipe):
    directory = output / "pronunciation"
    thresholds = {
        k: recipe["pronunciation"][k]
        for k in ("minimum_count", "minimum_fraction", "minimum_recordings")
    }
    analysis = analyze(bundle, directory, thresholds)
    observations = read(bundle / "zipa-word-sequences.json")["observations"]
    allowed = set(analysis["recordings"])
    if any(r["video"] not in allowed for r in observations):
        raise ResourceError("Nontraining word pronunciation evidence")
    fraction = recipe["pronunciation"]["whole_word_fraction"]

    def accepted(rows):
        return [
            r
            for r in recurring_word_labels(rows)
            if r["count"] / r["total"] >= fraction
        ]

    folds = []
    for video in sorted(allowed):
        training = [r for r in observations if r["video"] != video]
        validation = [r for r in observations if r["video"] == video]
        paths = {(r["word"], tuple(r["labels"])) for r in accepted(training)}
        folds.append(
            {
                "heldout": video,
                "training_recordings": sorted(allowed - {video}),
                "accepted_paths": len(paths),
                "validation_observations": len(validation),
                "supported_validation_observations": sum(
                    (r["word"], tuple(r["labels"])) in paths for r in validation
                ),
                "training_sha256": digest(training),
            }
        )
    lexical = read(bundle / "graph/lexicon_selected.json")
    inputs = read(bundle / "graph/compile_input.json")
    settings, tokens = inputs["settings"], inputs["tokens"]
    eligible = accepted(observations)
    missing = []
    for row in eligible:
        labels = [tokens.index(t) for t in row["labels"]]
        word = row["word"]
        options = settings.get("pronunciation_settings", {}).get(
            word, [{}] * len(lexical.get(word, []))
        )
        present = any(
            path == labels and p.get("exact")
            for path, p in zip(lexical.get(word, []), options)
        )
        row["already_in_baseline"] = present
        if not present:
            missing.append(row)
    counts, recordings = Counter(), defaultdict(set)
    for row in observations:
        counts[row["word"]] += 1
        recordings[row["word"]].add(row["video"])
    evidence = {}
    for word, paths in lexical.items():
        costs = settings.get("pronunciation_settings", {}).get(word, [{}] * len(paths))
        evidence[word] = {
            "lexicon_paths": [
                {
                    "labels": p,
                    "tokens": [tokens[t] for t in p],
                    "cost": c.get("cost", 0),
                    "exact": c.get("exact", False),
                }
                for p, c in zip(paths, costs)
            ],
            "training_observations": counts[word],
            "training_recordings": sorted(recordings[word]),
            "accepted_channel_paths": [r for r in eligible if r["word"] == word],
            "provenance": "Frozen baseline lexicon/compile settings, including retained historical public paths; new recurrence evidence only from six channel training recordings.",
        }
    result = {
        "status": "complete",
        "analysis": str(directory / "analysis.json"),
        "thresholds": {
            **thresholds,
            "whole_word_count": 3,
            "whole_word_fraction": fraction,
        },
        "whole_word_folds": folds,
        "accepted_paths": eligible,
        "missing_accepted_paths": missing,
        "recordings": sorted(allowed),
        "observation_count": len(observations),
        "context_passes_all_training_folds": all(
            f["metrics"]["context"]["negative_log_likelihood"]
            < f["metrics"]["positional"]["negative_log_likelihood"]
            for f in analysis["folds"]
        ),
        "word_provenance": evidence,
        "bundle_sha256": sha(bundle / "bundle.json"),
        "boundary_policy": analysis["boundary_policy"],
        "policy": "No development word, forced path, or acoustic frame is used to learn a pronunciation. Existing historical paths remain controls. Missing recurring training paths, if any, are eligible; lack of training evidence leaves lexical/acoustic attribution unresolved.",
    }
    write_json(directory / "coverage.json", result)
    return result
