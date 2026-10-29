"""Training-recording-only label contexts, with mappings rebuilt in every fold."""

import math
from collections import Counter, defaultdict

from asr.scn.dataset import digest
from asr.scn.lexicon import g2p, project
from asr.scn.orthography import normalize
from common.errors import ResourceError
from common.utils import write_json
from pipeline.experiment import read, sha, verified_files


def observations(bundle):
    import sentencepiece as spm

    verified_files(bundle, read(bundle / "bundle.json"))
    tokens = read(bundle / "graph/compile_input.json")["tokens"]
    tokenizer = spm.SentencePieceProcessor(model_file=str(bundle / "tokenizer.model"))
    directory = bundle / "training-observations"
    manifest = read(directory / "identity.json")
    allowed = {
        r["recording_id"]
        for r in read(bundle / "channel.json")["recordings"]
        if r["split"] == "train"
    }
    result, records = [], []
    for i, source in enumerate(manifest["rows"]):
        if (
            source["cohort"] != "channel"
            or source["split"] != "train"
            or source["recording_id"] not in allowed
        ):
            raise ResourceError("Nontraining recording in pronunciation evidence")
        record = read(directory / f"{i:03}.json")
        if record["id"] != source["id"] or record["video"] != source["recording_id"]:
            raise ResourceError("Misidentified channel observation")
        records.append(record)
        words = normalize(source["text"]).split()
        paths = [
            project(g2p(word, repair_soft_geminates=True)[0], tokens, tokenizer)
            for word in words
        ]
        flat = [t for path in paths for t in path]
        if flat != source["reference_ids"]:
            raise ResourceError("G2P changed since the cached training alignment")
        owners = [
            (wi, offset, len(path))
            for wi, path in enumerate(paths)
            for offset in range(len(path))
        ]
        cursor = 0
        for pair in record["alignment"]:
            if pair["reference"] is None:
                continue
            wi, offset, length = owners[cursor]
            if tokens[flat[cursor]] != pair["reference"]:
                raise ResourceError("Alignment/lexicon seed labels disagree")
            cursor += 1
            result.append(
                {
                    "video": record["video"],
                    "cue": record["id"],
                    "word": words[wi],
                    "seed": pair["reference"],
                    "observed": pair["hypothesis"],
                    "offset": offset,
                    "length": length,
                    "position": "initial"
                    if offset == 0
                    else "final"
                    if offset == length - 1
                    else "medial",
                    "previous_word": words[wi - 1] if wi else None,
                    "next_word": words[wi + 1] if wi + 1 < len(words) else None,
                    "written_context": words[wi],
                }
            )
        if cursor != len(flat):
            raise ResourceError("Incomplete training alignment")
    return result, records, tokens


def coordinate(row, mode):
    if mode == "global":
        return (row["seed"],)
    if mode == "positional":
        return (
            row["seed"],
            "final" if row["offset"] == row["length"] - 1 else "nonfinal",
        )
    return (row["seed"], row["word"], row["length"], row["offset"])


def estimate(rows, mode, thresholds):
    counts, totals, support = defaultdict(Counter), Counter(), defaultdict(set)
    for row in rows:
        k = coordinate(row, mode)
        totals[k] += 1
        if row["observed"] is not None:
            counts[k][row["observed"]] += 1
            support[k, row["observed"]].add(row["video"])
    accepted = {}
    for k, total in totals.items():
        alternatives = [
            label
            for label, count in counts[k].items()
            if label != k[0]
            and count >= thresholds["minimum_count"]
            and count / total >= thresholds["minimum_fraction"]
            and len(support[k, label]) >= thresholds["minimum_recordings"]
        ]
        if alternatives:
            accepted[k] = {
                label: counts[k][label] for label in [k[0], *sorted(alternatives)]
            }
    return accepted, counts, totals, support


def heldout_loss(rows, models, tokens, mode):
    total = loss = covered = 0
    for row in rows:
        if row["observed"] not in tokens[4:]:
            continue
        selected = None
        for fallback in (
            ["context", "positional", "global"]
            if mode == "context"
            else ["positional", "global"]
            if mode == "positional"
            else ["global"]
        ):
            selected = models[fallback][0].get(coordinate(row, fallback))
            if selected:
                covered += fallback == mode
                break
        selected = selected or {row["seed"]: 1}
        probability = (selected.get(row["observed"], 0) + 0.5) / (
            sum(selected.values()) + 0.5 * len(tokens[4:])
        )
        loss -= math.log(probability)
        total += 1
    return {
        "negative_log_likelihood": loss / total if total else None,
        "labels": total,
        "conditioned_labels": covered,
    }


def analyze(bundle, output, thresholds):
    rows, records, tokens = observations(bundle)
    videos = sorted({r["video"] for r in rows})
    if len(videos) != 6:
        raise ResourceError(
            "Pronunciation study requires the six frozen training recordings"
        )
    folds = []
    for heldout in videos:
        training = [r for r in rows if r["video"] != heldout]
        validation = [r for r in rows if r["video"] == heldout]
        models = {
            mode: estimate(training, mode, thresholds)
            for mode in ("global", "positional", "context")
        }
        folds.append(
            {
                "heldout": heldout,
                "training_recordings": [v for v in videos if v != heldout],
                "training_sha256": digest(training),
                "validation_sha256": digest(validation),
                "metrics": {
                    mode: heldout_loss(validation, models, tokens, mode)
                    for mode in models
                },
                "accepted_counts": {
                    mode: len(value[0]) for mode, value in models.items()
                },
            }
        )
    from pipeline.scn_word_labels import recurring_word_labels
    word_observations = read(bundle / "zipa-word-sequences.json")["observations"]
    if any(r["video"] not in videos for r in word_observations):
        raise ResourceError("Whole-word evidence contains a nontraining recording")
    for fold in folds:
        training_words = [r for r in word_observations if r["video"] != fold["heldout"]]
        heldout_words = [r for r in word_observations if r["video"] == fold["heldout"]]
        accepted_words = recurring_word_labels(training_words)
        accepted_paths = {(r["word"], tuple(r["labels"])) for r in accepted_words}
        fold["whole_word_control"] = {"accepted_paths": len(accepted_paths),
            "heldout_observations": len(heldout_words),
            "heldout_supported_paths": sum((r["word"], tuple(r["labels"])) in accepted_paths for r in heldout_words),
            "training_sha256": digest(training_words)}
    mapping, evidence = {}, []
    from pipeline.scn_label_mapping import empirical_label_costs, positional_label_costs

    global_alternatives, global_seed_costs, global_evidence = empirical_label_costs(
        records, tokens
    )
    positional_rows = [
        {
            "seed": r["seed"],
            "observed": r["observed"],
            "count": 1,
            "recordings": [r["video"]],
            "seed_label_offset": r["offset"],
            "seed_label_length": r["length"],
        }
        for r in rows
    ]
    positional_costs, positional_evidence = positional_label_costs(
        positional_rows, tokens
    )
    accepted, _counts, totals, support = estimate(rows, "context", thresholds)
    lexicon = read(bundle / "graph/lexicon_selected.json")
    for k, labels in sorted(accepted.items()):
        seed, word, length, offset = k
        if word not in lexicon or seed not in tokens[4:]:
            continue
        token = tokens.index(seed)
        if not any(len(p) == length and p[offset] == token for p in lexicon[word]):
            continue
        maximum = max(labels.values())
        options = [
            {
                "token": tokens.index(label),
                "cost": math.log((maximum + 0.5) / (count + 0.5)),
            }
            for label, count in labels.items()
            if label in tokens[4:]
        ]
        mapping.setdefault(word, {})[f"{length}:{offset}:{token}"] = options
        evidence.append(
            {
                "word": word,
                "written_context": word,
                "seed": seed,
                "length": length,
                "offset": offset,
                "total": totals[k],
                "labels": [
                    {
                        "label": label,
                        "count": count,
                        "recordings": sorted(support[k, label]),
                    }
                    for label, count in labels.items()
                ],
            }
        )
    boundary = Counter(
        (r["previous_word"], r["word"], r["seed"], r["observed"])
        for r in rows
        if r["offset"] == 0
    )
    summary = {
        "status": "complete",
        "thresholds": thresholds,
        "folds": folds,
        "phone_context_costs": mapping,
        "global_alternatives": global_alternatives,
        "global_seed_costs": global_seed_costs,
        "positional_costs": positional_costs,
        "global_evidence": global_evidence,
        "positional_evidence": positional_evidence,
        "evidence": evidence,
        "observations": len(rows),
        "recordings": videos,
        "boundary_observations": [
            {
                "previous_word": k[0],
                "word": k[1],
                "seed": k[2],
                "observed": k[3],
                "count": count,
            }
            for k, count in boundary.most_common()
        ],
        "boundary_policy": "Neighboring words are measured only. Greedy cue alignments do not identify syntactic doubling independently; no new boundary-dependent pronunciation arcs are inferred.",
        "context_policy": "Whole written word and seed-label coordinate; no invented grapheme offset or collapse of e/è, ci/cci, ni/nni. Existing exact whole-word paths retain their labels.",
        "bundle_sha256": sha(bundle / "bundle.json"),
        "implementation_sha256": sha(__file__),
    }
    write_json(output / "analysis.json", summary)
    return summary
