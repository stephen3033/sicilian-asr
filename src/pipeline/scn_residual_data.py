"""Stable residual error identities and diagnostic-only word counterfactuals."""

from collections import Counter

from asr.scn.orthography import normalize
from common.artifacts import digest
from common.errors import ResourceError
from evaluation.metrics import edits


def suffix_distances(reference, hypothesis):
    import numpy as np

    n, m = len(reference), len(hypothesis)
    d = np.zeros((n + 1, m + 1), dtype=np.int32)
    d[n, :] = range(m, -1, -1)
    d[:, m] = range(n, -1, -1)
    for i in range(n - 1, -1, -1):
        for j in range(m - 1, -1, -1):
            d[i, j] = min(
                int(d[i + 1, j]) + 1,
                int(d[i, j + 1]) + 1,
                int(d[i + 1, j + 1]) + (reference[i] != hypothesis[j]),
            )
    return d


def inventory(recordings, decoded, lexicon):
    decoded = {r["video"]: r for r in decoded}
    errors, cases, segments, controls = [], {}, [], []
    for recording in recordings:
        video = recording["video"]
        result = decoded[video]
        reference = normalize(recording["reference"]).split()
        hypothesis = normalize(result["text"]).split()
        metrics, alignment = edits(reference, hypothesis)
        if metrics["errors"] != result["metrics"]["wer"]["errors"]:
            raise ResourceError("Residual alignment differs from frozen baseline")
        distances = suffix_distances(reference, hypothesis)
        owners, starts = [], []
        for segment in result["segments"]:
            starts.append(len(owners))
            ws = normalize(segment["text"]).split()
            owners += [(segment["index"], i) for i in range(len(ws))]
            segments.append(
                {
                    "video": video,
                    **segment,
                    "diagnostic_id": f"{video}:{segment['index']:03}",
                }
            )
        if len(owners) != len(hypothesis):
            raise ResourceError("Segment word ownership is incomplete")
        cue_words, cue_owners = [], []
        for cue in recording["cues"]:
            ws = normalize(cue["text"]).split()
            cue_words += ws
            cue_owners += [
                {k: cue[k] for k in ("start_seconds", "end_seconds")} for _ in ws
            ]
        cue_map, ri, ci = {}, 0, 0
        for pair in edits(reference, cue_words)[1]:
            if pair["operation"] == "correct":
                cue_map[ri] = cue_owners[ci]
            ri += pair["reference"] is not None
            ci += pair["hypothesis"] is not None
        ri = hi = 0
        annotated = []
        for position, pair in enumerate(alignment):
            op, ref, hyp = pair["operation"], pair["reference"], pair["hypothesis"]
            choices = []
            if ri < len(reference):
                choices.append(int(distances[ri + 1, hi]) + 1)
            if hi < len(hypothesis):
                choices.append(int(distances[ri, hi + 1]) + 1)
            if ri < len(reference) and hi < len(hypothesis):
                choices.append(
                    int(distances[ri + 1, hi + 1]) + (reference[ri] != hypothesis[hi])
                )
            uncertain = sum(c == int(distances[ri, hi]) for c in choices) > 1
            if hyp is not None:
                si, local = owners[hi]
                boundary = False
            elif hi < len(owners):
                si, local = owners[hi]
                boundary = bool(hi and owners[hi - 1][0] != si)
            elif owners:
                si, last = owners[-1]
                local = last + 1
                boundary = False
            else:
                si, local, boundary = 0, 0, True
            segment = result["segments"][si]
            row = {
                "id": f"{video}:alignment-{position:04}",
                "video": video,
                "alignment_index": position,
                **pair,
                "reference_index": ri if ref is not None else None,
                "reference_anchor": ri,
                "hypothesis_index": hi if hyp is not None else None,
                "segment": si,
                "local_start": local,
                "local_end": local + (hyp is not None),
                "target": op != "correct" and (op == "insertions" or ref in lexicon),
                "reference_in_vocabulary": ref in lexicon if ref else None,
                "alignment_ambiguous": uncertain,
                "cross_boundary_deletion": boundary,
                "caption_interval": cue_map.get(ri) if ref else None,
                "segment_interval": [
                    segment["source_start_seconds"],
                    segment["source_end_seconds"],
                ],
                "forced_cut": segment["forced_cut"],
                "reference_context": reference[max(0, ri - 4) : ri + 5],
                "hypothesis_context": hypothesis[max(0, hi - 4) : hi + 5],
                "near_boundary": local <= 1
                or local >= len(segment["text"].split()) - 2,
            }
            annotated.append(row)
            (controls if op == "correct" else errors).append(row)
            ri += ref is not None
            hi += hyp is not None

        def case(rows, kind):
            targets = [r for r in rows if r["target"]]
            if not targets:
                return
            si = targets[0]["segment"]
            if any(r["segment"] != si for r in rows):
                return
            before = normalize(result["segments"][si]["text"]).split()
            start, end = (
                min(r["local_start"] for r in rows),
                max(r["local_end"] for r in rows),
            )
            replacement = []
            for r in rows:
                if r["target"]:
                    if r["reference"] is not None:
                        replacement.append(r["reference"])
                elif r["hypothesis"] is not None:
                    replacement.append(r["hypothesis"])
            after = before[:start] + replacement + before[end:]
            if set(after) - set(lexicon):
                raise ResourceError("Counterfactual introduced an OOV word")
            ident = digest([video, si, after])[:24]
            if ident not in cases:
                cases[ident] = {
                    "id": ident,
                    "video": video,
                    "segment": si,
                    "original": before,
                    "corrected": after,
                    "error_ids": [],
                    "kinds": [],
                    "reference_constrained": True,
                    "training_eligible": False,
                }
            for r in targets:
                if r["id"] not in cases[ident]["error_ids"]:
                    cases[ident]["error_ids"].append(r["id"])
            if kind not in cases[ident]["kinds"]:
                cases[ident]["kinds"].append(kind)

        run = []
        for row in annotated:
            if row["operation"] == "correct":
                case(run, "joint")
                run = []
            else:
                if run and row["segment"] != run[-1]["segment"]:
                    case(run, "joint")
                    run = []
                run.append(row)
                if row["target"]:
                    case([row], "individual")
        case(run, "joint")
    counts = Counter(r["operation"] for r in errors if r["target"])
    return {
        "errors": errors,
        "controls": controls,
        "cases": list(cases.values()),
        "segments": segments,
        "target_counts": dict(counts),
        "total_errors": len(errors),
        "policy": "Stable baseline edit identities. Insertions are included; reference-OOV edits remain outside the target set. Counterfactuals are never training evidence.",
    }


def candidate_effects(ledger, baseline_recordings, candidate_recordings, references):
    """Compare reference-position errors; insertion effects use reference gaps."""

    def positions(text, reference):
        position, bad, insertions = 0, set(), Counter()
        for pair in edits(normalize(reference).split(), normalize(text).split())[1]:
            if pair["operation"] == "insertions":
                insertions[position] += 1
            elif pair["operation"] != "correct":
                bad.add(position)
            position += pair["reference"] is not None
        return bad, insertions

    actual = {
        r["video"]: positions(r["text"], references[r["video"]])
        for r in candidate_recordings
    }
    baseline = {
        r["video"]: positions(r["text"], references[r["video"]])
        for r in baseline_recordings
    }
    fixed, pending_insertions = [], Counter()
    for row in ledger:
        if not row["target"]:
            continue
        bad, ins = actual[row["video"]]
        if row["operation"] == "insertions":
            coordinate = row["video"], row["reference_anchor"]
            pending_insertions[coordinate] += 1
            removed = max(
                0, baseline[row["video"]][1][coordinate[1]] - ins[coordinate[1]]
            )
            if pending_insertions[coordinate] <= removed:
                fixed.append(row["id"])
        elif row["reference_index"] not in bad:
            fixed.append(row["id"])
    new_errors = sum(
        len(bad - baseline[v][0])
        + sum(max(0, n - baseline[v][1][i]) for i, n in ins.items())
        for v, (bad, ins) in actual.items()
    )
    return {
        "repaired_target_ids": fixed,
        "new_reference_position_errors_and_insertions": new_errors,
        "policy": "Deterministic reference-position comparison; ambiguous baseline alignments remain flagged. This is an outcome association, not a causal label.",
    }
