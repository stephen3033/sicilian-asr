"""Evidence-bearing error classifications, intervention outcomes, and validation."""

import json
import math
from collections import Counter, defaultdict
from pathlib import Path

from asr.scn.orthography import normalize
from common.artifacts import digest, read, seal, sha, verified_files
from common.errors import ResourceError
from common.utils import write_json
from pipeline.experiment import aggregate, improves

SOLUTIONS = {
    "lm_preference": {
        "mechanism": "The LM term reverses the local decision, or chooses between acoustically tied alternatives.",
        "solution": "Repair corpus boundary construction and conditional probabilities; calibrate LM scale and word bonus; retain written distinctions.",
        "missing": "A trained candidate that resolves this context without aggregate regression; independent reference validation.",
    },
    "pronunciation_cost": {
        "mechanism": "Removing lexical path penalties reverses the corrected-versus-decoded decision.",
        "solution": "Recalibrate pronunciation penalties or add independently recurring channel label paths, preserving fallback paths.",
        "missing": "Training recordings supporting the preferred lexical cost or path; validation of any residual word-boundary ambiguity.",
    },
    "mixed_lm_pronunciation": {
        "mechanism": "Both LM and pronunciation-cost interventions can reverse the constrained score comparison.",
        "solution": "Compare independent LM and lexical-cost repairs, then combine only demonstrated improvements.",
        "missing": "A controlled deployable repair distinguishing the interacting factors.",
    },
    "search": {
        "mechanism": "A better corrected path exists in the compiled graph but was missed by the baseline search.",
        "solution": "Increase search capacity and verify full-recording error counts.",
        "missing": "A wider-search decode that recovers the path without offsetting errors.",
    },
    "alignment_or_segmentation_unresolved": {
        "mechanism": "The reference caption interval or deletion alignment is not securely assigned to the tested segment.",
        "solution": "Rerun frozen ZIPA across the boundary; compare general segmentation; verify reference wording and timing against audio.",
        "missing": "Independent word timing and verbatim reference verification; joined-waveform outcomes alone do not prove caption correctness.",
    },
    "ambiguous_edit_alignment": {
        "mechanism": "Multiple minimum-edit alignments exist at this position; the individual edit attribution is ambiguous.",
        "solution": "Inspect the complete adjacent-error group and its audio/caption interval before changing a word-specific rule.",
        "missing": "Word-level alignment and a verified local reference; all score mechanisms remain attached as secondary evidence.",
    },
    "acoustic_or_pronunciation_support_unresolved": {
        "mechanism": "Even with pronunciation penalties removed, current legal reference paths fit the frozen acoustic posteriors worse.",
        "solution": "Check independently observed training pronunciations and reference timing; distinguish missing lexical support from ZIPA evidence conflict.",
        "missing": "Independent phone/word annotation or recurring channel examples of this pronunciation. Forced development paths are not evidence for adding a path.",
    },
    "mixed_or_unresolved": {
        "mechanism": "The measured factors do not establish one causal component for this edit.",
        "solution": "Use joint counterfactuals and controlled intervention outcomes; inspect the reference/audio when attribution remains ambiguous.",
        "missing": "Independent reference/phone evidence or a repair that isolates the responsible factor.",
    },
}


def classify(error, cases, proven_search):
    individual = [c for c in cases if "individual" in c["kinds"]]
    case = min(individual or cases, key=lambda c: (len(c["error_ids"]), c["id"]))
    flags = set(case["mechanistic_flags"])
    caption = error["caption_interval"]
    start, end = error["segment_interval"]
    outside = caption is not None and (
        caption["end_seconds"] <= start or caption["start_seconds"] >= end
    )
    if error["cross_boundary_deletion"] or outside:
        category = "alignment_or_segmentation_unresolved"
    elif error["alignment_ambiguous"]:
        category = "ambiguous_edit_alignment"
    elif any(c["id"] in proven_search for c in cases):
        category = "search"
    elif {"lm_score_reversal", "pronunciation_cost_reversal"} <= flags:
        category = "mixed_lm_pronunciation"
    elif "pronunciation_cost_reversal" in flags:
        category = "pronunciation_cost"
    elif "lm_score_reversal" in flags or (
        "acoustically_tied_legal_sequences" in flags and case["delta"]["lm_score"] < 0
    ):
        category = "lm_preference"
    elif "acoustic_evidence_conflict" in flags:
        category = "acoustic_or_pronunciation_support_unresolved"
    else:
        category = "mixed_or_unresolved"
    return category, case, outside


def report(root, bundle, output, recipe):
    validation, cleanup = (
        read(output / "validation.json"),
        read(output / "cleanup.json"),
    )
    if validation["status"] != "verified" or cleanup["status"] != "complete":
        raise ResourceError(
            "Complete validation and diagnostic retirement before final reporting"
        )
    inventory, scores, trials = (
        read(output / n)
        for n in ("inventory.json", "scores.json", "interventions.json")
    )
    if scores["status"] != "complete" or trials["status"] != "complete":
        raise ResourceError(
            "Residual reporting requires all scores and intervention outcomes"
        )
    verified_files(bundle, read(bundle / "bundle.json"))
    verified_files(output, read(output / "inventory.receipt.json"))
    verified_files(output / "lm", read(output / "lm/models.receipt.json"))
    verified_files(
        output / "research-candidate",
        read(output / "research-candidate/candidate.json"),
    )
    verify_scores(output, scores, inventory)
    baseline = read(output / "baseline-replay/result.json")
    coverage = read(output / "pronunciation/coverage.json")
    boundary = read(output / "boundary-outcomes.json")
    language_models = read(output / "lm/models.json")
    calibration = read(output / "calibration.json")
    historical_lms = read(root / recipe["corpus_study"] / "lm-results.json")["results"]
    parent_repairs = []
    for trial in calibration["results"]:
        if (
            not trial["family"].endswith("-support")
            or trial["decoder"].get("word_bonus", 0) != 0
        ):
            continue
        parent = next(
            (
                r
                for r in historical_lms
                if r["lm"] == trial["family"].removesuffix("-support")
                and r["decoder"]["lm_scale"] == trial["decoder"]["lm_scale"]
            ),
            None,
        )
        if parent:
            parent_repairs.append(
                {
                    "parent": parent["id"],
                    "repair": trial["id"],
                    "parent_metrics": parent["metrics"],
                    "repair_metrics": trial["metrics"],
                    "improves_matched_parent": improves(
                        trial["metrics"], parent["metrics"]
                    ),
                }
            )
    by_error, repaired, repaired_without_regression, boundary_repaired = (
        defaultdict(list),
        defaultdict(list),
        defaultdict(list),
        defaultdict(list),
    )
    for segment in scores["segments"]:
        for case in segment["cases"]:
            for ident in case["error_ids"]:
                by_error[ident].append(case)
    for row in trials["results"]:
        directory = Path(row["result"]).parent
        verified_files(directory, read(directory / "result.receipt.json"))
        result = read(row["result"])
        if (
            aggregate(result["recordings"]) != row["metrics"]
            or len(result["recordings"]) != 3
        ):
            raise ResourceError("Incomplete or inconsistent intervention scores")
        for ident in row["effects"]["repaired_target_ids"]:
            repaired[ident].append(row["id"])
            if row["improves_baseline"]:
                repaired_without_regression[ident].append(row["id"])
    for probe in boundary["pairs"]:
        for ident in probe["effects"]["repaired_target_ids"]:
            boundary_repaired[ident].append(probe["pair"]["id"])
    selected_fixed = set(trials["selected"]["effects"]["repaired_target_ids"])
    proven = {r["case"] for r in trials["search"]["cases"] if r["search_error"]}
    training_counts, training_recordings = Counter(), defaultdict(set)
    for cue in read(bundle / "channel.json")["training"]:
        for word in normalize(cue["text"]).split():
            training_counts[word] += 1
            training_recordings[word].add(cue["recording_id"])
    ledger, review, categories, mechanisms = [], [], Counter(), Counter()
    for error in inventory["errors"]:
        if not error["target"]:
            ledger.append(
                {
                    **error,
                    "classification": "reference_oov_outside_requested_target",
                    "included_in_full_recording_scores": True,
                }
            )
            continue
        cases = by_error[error["id"]]
        if not cases:
            raise ResourceError("Target error has no counterfactual evidence")
        category, individual, outside = classify(error, cases, proven)
        categories[category] += 1
        for flag in individual["mechanistic_flags"]:
            mechanisms[flag] += 1
        lexical = {
            word: coverage["word_provenance"].get(word)
            for word in (error["reference"], error["hypothesis"])
            if word is not None
        }
        row = {
            **error,
            "classification": category,
            "evidence_status": "unresolved_attribution"
            if "unresolved" in category or category == "ambiguous_edit_alignment"
            else "controlled_score_mechanism",
            "mechanism_and_solution": SOLUTIONS[category],
            "caption_outside_tested_segment": outside,
            "primary_counterfactual": individual["id"],
            "counterfactuals": [
                {
                    k: c[k]
                    for k in (
                        "id",
                        "kinds",
                        "error_ids",
                        "delta",
                        "unweighted_acoustic_delta",
                        "lattice",
                        "mechanistic_flags",
                    )
                }
                for c in cases
            ],
            "pronunciation_provenance": lexical,
            "reference_training_tokens": training_counts[error["reference"]],
            "reference_training_recordings": sorted(
                training_recordings[error["reference"]]
            ),
            "repaired_in_trials": repaired[error["id"]],
            "repaired_in_trials_improving_both_metric_criteria": repaired_without_regression[
                error["id"]
            ],
            "repaired_by_selected_candidate": error["id"] in selected_fixed,
            "repaired_in_joined_audio_diagnostics": boundary_repaired[error["id"]],
            "training_eligible": False,
            "attribution_policy": "Mechanistic score reversal and observed repair are distinct. Caption references are not independently verified speech annotations. No counterfactual is used to propose training vocabulary, pronunciation paths, or conversion rules.",
        }
        ledger.append(row)
        if (
            row["evidence_status"] == "unresolved_attribution"
            or not repaired_without_regression[error["id"]]
        ):
            review.append(
                {
                    "id": error["id"],
                    "video": error["video"],
                    "caption_interval": error["caption_interval"],
                    "segment_interval": error["segment_interval"],
                    "reference": error["reference"],
                    "hypothesis": error["hypothesis"],
                    "classification": category,
                    "missing_facts": SOLUTIONS[category]["missing"],
                    "training_eligible": False,
                }
            )
    if (
        sum(categories.values()) != 1184
        or len(ledger) != 1429
        or len({r["id"] for r in ledger}) != len(ledger)
    ):
        raise ResourceError(
            "Residual report does not account for the frozen error inventory"
        )
    for name, rows in (("error-ledger.jsonl", ledger), ("review-queue.jsonl", review)):
        with (output / name).open("w") as stream:
            stream.writelines(
                json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in rows
            )
    write_json(
        output / "counterfactuals.json",
        {
            "status": "complete",
            "training_eligible": False,
            "cases": [c for s in scores["segments"] for c in s["cases"]],
        },
    )
    controls = []
    score_segments = {r["segment"]: r for r in scores["segments"]}
    for label, rows in (
        ("correct", inventory["controls"]),
        ("target_errors", [r for r in inventory["errors"] if r["target"]]),
    ):
        terms = []
        for row in rows:
            if row["hypothesis"] is not None:
                terms.append(
                    score_segments[f"{row['video']}:{row['segment']:03}"][
                        "factor_parity"
                    ]["lm_terms"][row["local_start"]]
                )
        controls.append(
            {
                "group": label,
                "positions": len(rows),
                "hypothesis_lm_orders": dict(Counter(t["ngram_order"] for t in terms)),
                "lm_order_source": "KenLM context metadata; score attribution uses the compiled grammar",
                "reference_seen_in_training": sum(
                    training_counts[r["reference"]] > 0 for r in rows
                ),
                "near_boundary": sum(r["near_boundary"] for r in rows),
                "ambiguous_alignment": sum(r["alignment_ambiguous"] for r in rows),
            }
        )
    families = defaultdict(list)
    for row in trials["results"]:
        families[row["family"]].append(row)
    from pipeline.scn_experiment import rank

    best_by_family = [min(rows, key=rank) for rows in families.values()]
    confusing = Counter(
        (r["reference"], r["hypothesis"], r["classification"])
        for r in ledger
        if r["target"]
    )
    class_outcomes = {
        category: {
            "errors": count,
            "repaired_in_any_trial": sum(
                r["classification"] == category and bool(r.get("repaired_in_trials"))
                for r in ledger
            ),
            "repaired_by_selected": sum(
                r["classification"] == category
                and r.get("repaired_by_selected_candidate", False)
                for r in ledger
            ),
        }
        for category, count in categories.items()
    }
    summary = {
        "status": "complete",
        "target_errors": 1184,
        "other_oov_reference_errors": 245,
        "fixed_vocabulary_wer_floor": 245
        / baseline["metrics"]["wer"]["reference_units"],
        "maximum_known_word_errors_for_wer_below_10_percent": math.ceil(
            0.1 * baseline["metrics"]["wer"]["reference_units"]
        )
        - 1
        - 245,
        "baseline_metrics": baseline["metrics"],
        "selected": trials["selected"],
        "classification_counts": dict(categories),
        "classification_outcomes": class_outcomes,
        "representative_cases": [
            next(r for r in ledger if r["classification"] == category)
            for category in categories
        ],
        "lm_construction": {
            "baseline_audit": language_models["baseline_lm_audit"],
            "normalized_baseline_audit": read(
                output / "baseline-normalized/audit.json"
            ),
            "original_control_audits": language_models["original_lm_audit"],
            "training_selected_floor_mass": language_models["selected_floor_mass"],
            "matched_raw_converted": calibration["matched_controls"],
            "matched_parent_repairs": parent_repairs,
            "support_model_real_wikipedia_shares": {
                model["id"]: model["statistics"][
                    "effective_wikipedia_share_without_prior"
                ]
                for model in language_models["models"]
                if model.get("statistics")
            },
        },
        "primary_mechanistic_flags": dict(mechanisms),
        "score_parity_segments": len(scores["segments"]),
        "counterfactual_cases": scores["cases"],
        "maximum_factor_parity_error": max(
            abs(s["factor_parity"]["residual"]) for s in scores["segments"]
        ),
        "compiled_kenlm_score_difference_segments": sum(
            abs(s["factor_parity"]["compiled_minus_kenlm"]) > 0.001
            for s in scores["segments"]
        ),
        "search": trials["search"],
        "correct_word_controls": controls,
        "intervention_count": len(trials["results"]),
        "best_by_family": best_by_family,
        "matched_conversion_controls": calibration["matched_controls"],
        "known_recurring_channel_paths_missing_from_baseline": len(
            coverage["missing_accepted_paths"]
        ),
        "target_errors_with_any_trial_repair": sum(bool(v) for v in repaired.values()),
        "target_errors_repaired_by_selected": len(selected_fixed),
        "target_errors_repaired_in_joined_audio_probes": sum(
            bool(v) for v in boundary_repaired.values()
        ),
        "review_queue_count": len(review),
        "production_changed": False,
        "validation": validation,
        "cleanup": cleanup,
        "targets_met": trials["selected"]["metrics"]["wer"]["rate"] < 0.1
        and trials["selected"]["metrics"]["cer"]["rate"] < 0.05,
        "remaining_gap_percentage_points": {
            k: max(0, (trials["selected"]["metrics"][k]["rate"] - target) * 100)
            for k, target in (("wer", 0.1), ("cer", 0.05))
        },
        "confusions": [
            {"reference": r, "hypothesis": h, "classification": c, "count": n}
            for (r, h, c), n in confusing.most_common()
        ],
        "scope": "Development diagnosis and finite controlled repairs, not a global accuracy ceiling or independent acceptance.",
    }
    write_json(output / "report.json", summary)
    text = markdown(summary, output, root)
    (output / "report.md").write_text(text)
    (root / "docs/scn/residual-diagnosis.md").write_text(text)
    names = [
        "report.json",
        "report.md",
        "error-ledger.jsonl",
        "review-queue.jsonl",
        "counterfactuals.json",
    ]
    seal(
        output,
        "report.receipt.json",
        {
            "recipe": recipe,
            "baseline": sha(bundle / "bundle.json"),
            "scores": sha(output / "scores.json"),
            "interventions": sha(output / "interventions.json"),
            "implementation": sha(__file__),
        },
        names,
    )
    write_json(
        output / "completion.json",
        {
            "schema_version": 1,
            "status": "implemented_and_verified",
            "scope": "Frozen-ZIPA development diagnosis and controlled repairs; no production promotion or independent fresh-recording acceptance.",
            "target_errors_accounted_for": sum(categories.values()),
            "classification_counts": dict(categories),
            "counterfactuals": scores["cases"],
            "complete_recording_comparisons": len(trials["results"]),
            "selected": trials["selected"]["id"],
            "metrics": trials["selected"]["metrics"],
            "targets_met": summary["targets_met"],
            "production_unchanged": validation[
                "production_and_historical_selection_unchanged"
            ],
            "python_tests_passed": validation["python_tests_passed"],
            "preceding_phase": {
                "path": str(root / recipe["corpus_study"] / "completion.json"),
                "sha256": sha(root / recipe["corpus_study"] / "completion.json"),
            },
            "evidence": {
                name: sha(output / name)
                for name in (
                    "study.json",
                    "inventory.receipt.json",
                    "scores.json",
                    "lm/models.receipt.json",
                    "calibration.json",
                    "repair-results.json",
                    "interventions.json",
                    "selected-fresh-replay/result.receipt.json",
                    "research-candidate/candidate.json",
                    "experimental-scn.yaml",
                    "production-inputs.json",
                    "validation.json",
                    "cleanup.json",
                    "report.receipt.json",
                )
            },
        },
    )
    return summary


def verify_scores(output, scores, inventory):
    if (
        len(scores["segments"]) != len(inventory["segments"])
        or not scores["factor_parity_verified"]
    ):
        raise ResourceError("Incomplete diagnostic factor parity")
    expected = {r["diagnostic_id"] for r in inventory["segments"]}
    if {r["segment"] for r in scores["segments"]} != expected:
        raise ResourceError("Diagnostic segment inventory mismatch")
    if {c["id"] for s in scores["segments"] for c in s["cases"]} != {
        c["id"] for c in inventory["cases"]
    }:
        raise ResourceError("Missing counterfactual scores")
    for segment in scores["segments"]:
        matching = []
        for path in (output / "scores").glob(
            f"*/{segment['segment'].replace(':', '-')}/result.receipt.json"
        ):
            receipt = read(path)
            if (path.parent / "result.json").exists() and digest(
                read(path.parent / "result.json")
            ) == digest(segment):
                verified_files(path.parent, receipt)
                matching.append(path)
        if not matching or not segment["factor_parity"]["verified"]:
            raise ResourceError("Unverified diagnostic segment")
        if any(c["training_eligible"] for c in segment["cases"]):
            raise ResourceError(
                "Diagnostic references were marked as training evidence"
            )


def markdown(summary, output, root):
    baseline, selected = summary["baseline_metrics"], summary["selected"]
    lines = [
        "# Residual word-error diagnosis",
        "",
        "All **1,184 target errors** are accounted for: 805 known-word substitutions, 248 deletions, and 131 insertions. The other 245 OOV-reference errors remain in every complete-recording score.",
        "",
        f"Baseline: **{baseline['wer']['rate']:.2%} WER / {baseline['cer']['rate']:.2%} CER**. Research selection: **{selected['metrics']['wer']['rate']:.2%} WER / {selected['metrics']['cer']['rate']:.2%} CER** (`{selected['id']}`). Production remains unchanged.",
        "",
        f"Exact factor reconstruction passed on all {summary['score_parity_segments']} segments (maximum residual {summary['maximum_factor_parity_error']:.8g}). {summary['counterfactual_cases']} individual/joint reference counterfactuals were scored on the original frozen frames. Compiled-grammar backoff scores differed slightly from KenLM on {summary['compiled_kenlm_score_difference_segments']} segments; the diagnostic decomposition uses the actual grammar.",
        "",
        "Each counterfactual holds the surrounding baseline hypothesis fixed and uses existing legal pronunciation paths. Joint adjacent-error corrections are also tested. Attribution is conditional on that context; these scores are not a fully correct transcript, an acoustic accuracy ceiling, or an independent annotation of what was spoken.",
        "",
        "## Classification",
        "",
        "These are evidence-supported mechanisms or explicit unresolved cases, not independently certified speech-error causes. Alignment ambiguity takes precedence over word-specific attribution; secondary mechanisms remain in the ledger.",
        "",
        "| Classification | Errors | Repaired in any trial | Repaired by selection | Tested or required solution |",
        "|---|---:|---:|---:|---|",
    ]
    for category, count in sorted(
        summary["classification_counts"].items(), key=lambda r: -r[1]
    ):
        lines.append(
            f"| {category.replace('_', ' ')} | {count} | {summary['classification_outcomes'][category]['repaired_in_any_trial']} | {summary['classification_outcomes'][category]['repaired_by_selected']} | {SOLUTIONS[category]['solution']} |"
        )
    lines += [
        "",
        "## Representative score comparisons",
        "",
        "Deltas compare the corrected sequence with the decoded sequence on the same frames. Positive values favor the correction. Acoustic ties can still be resolved incorrectly by lexical penalties or LM context. These examples are diagnostic constraints, never proposed training sentences.",
        "",
        "| Reference | Hypothesis | Classification | Acoustic delta | Pronunciation delta | LM delta |",
        "|---|---|---|---:|---:|---:|",
    ]
    for row in summary["representative_cases"]:
        case = next(
            c
            for c in row["counterfactuals"]
            if c["id"] == row["primary_counterfactual"]
        )
        delta = case["delta"]
        lines.append(
            f"| {row['reference'] or '∅'} | {row['hypothesis'] or '∅'} | {row['classification'].replace('_', ' ')} | {delta['acoustic_score']:.4f} | {delta['pronunciation_score']:.4f} | {delta['lm_score']:.4f} |"
        )
    lines += [
        "",
        "## Controlled outcomes",
        "",
        "Each row below is the lowest-WER candidate in its family, with CER breaking ties. A family winner may still be a tradeoff; research selection applies the joint WER/CER acceptance rule.",
        "",
        "| Candidate (best WER per family) | WER | CER | Target errors repaired | New reference-position errors/insertions |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in summary["best_by_family"]:
        lines.append(
            f"| {row['id']} | {row['metrics']['wer']['rate']:.2%} | {row['metrics']['cer']['rate']:.2%} | {len(row['effects']['repaired_target_ids'])} | {row['effects']['new_reference_position_errors_and_insertions']} |"
        )
    lines += [
        "",
        f"Converted text outperforms matched raw text under the joint WER/CER rule in {sum(row['conversion_improves'] for row in summary['matched_conversion_controls'])} of {len(summary['matched_conversion_controls'])} repaired-model settings. These comparisons use the same source passages, real-token weighting, vocabulary, and decoder setting; neither family surpasses the historical baseline.",
        "",
        "The construction repair also has matched before/after controls. These quantify an improvement to the newly trained models even when the historical winning baseline remains stronger.",
        "",
        "| Original model/scale | Original word/character errors | Repaired word/character errors |",
        "|---|---:|---:|",
    ]
    for row in summary["lm_construction"]["matched_parent_repairs"]:
        before, after = row["parent_metrics"], row["repair_metrics"]
        lines.append(
            f"| {row['parent']} | {before['wer']['errors']} / {before['cer']['errors']} | {after['wer']['errors']} / {after['cer']['errors']} |"
        )
    lines += [
        "",
        "Repair counts use stable baseline reference positions and insertion gaps. They are outcome associations, not causal attribution; flagged edit-alignment ambiguity remains relevant.",
        "",
        f"{summary['search']['proven_search_errors']} compiled-path search failures were demonstrated. {summary['known_recurring_channel_paths_missing_from_baseline']} recurring channel pronunciation paths satisfying the retained thresholds were missing from the baseline. More permissive paths cannot be justified by forced development alignments.",
        "",
        f"Joined-waveform diagnostics repaired {summary['target_errors_repaired_in_joined_audio_probes']} distinct target positions in at least one probe. These overlapping local probes are not one deployable candidate. The 30/36-second segmentation row is the full-recording control, using newly computed posteriors from the unchanged ZIPA checkpoint.",
        "",
        "## Remaining limitations",
        "",
        f"The channel-only control had 32,515 artificial one-word vocabulary sentences alongside 695 real training cues. The support-floor repair removes those artificial contexts and recomputes normalized backoff weights. The floor mass ({summary['lm_construction']['training_selected_floor_mass']:g}) was selected by six training-recording folds. Raw and converted repairs retain exactly matched source passages and repetition counts. The historical baseline unigram mass is {summary['lm_construction']['baseline_audit']['unigram_mass']:.6f}; normalization is tested separately.",
        "",
        f"The `w50` repair names identify their original parents. After removing the artificial vocabulary sentences, Wikipedia supplies {summary['lm_construction']['support_model_real_wikipedia_shares']['converted-w50-o3-support']:.2%} of real tokens in both matched repairs; their real channel/Wikipedia counts remain unchanged. The selected support mass was the highest of the three tested values, not an established optimum.",
        "",
        f"The fixed vocabulary imposes a {summary['fixed_vocabulary_wer_floor']:.2%} WER floor from 245 reference tokens it cannot emit. Reaching WER below 10% with that vocabulary would leave room for at most {summary['maximum_known_word_errors_for_wer_below_10_percent']} known-word errors, compared with the baseline 1,184. This is a vocabulary-specific bound, not a bound on an expanded system.",
        "",
        "The six training recordings provide limited independent pronunciation recurrence and domain context. Where the unpenalized reference paths fit the posteriors worse, these measurements cannot separate missing lexical variants, acoustic-model mismatch, subtitle differences, and timing error without independent phone/word annotation. Acoustically tied written forms require grammatical context. Artificial vocabulary sentences and unnormalized ARPA weights are measured construction properties; their removal is a tested hypothesis, not an assumed accuracy gain.",
        "",
        f"The selected result remains {summary['remaining_gap_percentage_points']['wer']:.2f} WER percentage points and {summary['remaining_gap_percentage_points']['cer']:.2f} CER percentage points above the 10%/5% targets. Independent acceptance on fresh recordings remains separate from development selection.",
        "",
        "## Reproduction and artifacts",
        "",
        f"Validation passed {summary['validation']['python_tests_passed']} Python tests, exact Italian and Sicilian production regressions, portable-candidate preparation/transcription, and wheel packaging. Retirement removed {summary['cleanup']['removed_file_bytes'] / 1024**2:.2f} MiB of superseded diagnostic attempts after verifying their replacement. Current controlled candidate graphs remain available for replay; the earlier repository cleanup is recorded separately.",
        "",
        "Run `.venv/bin/python -m pipeline scn-improve --config configs/scn-residual.json --stage all`. Each stage is resumable and checksummed; native workers enforce the existing resource limit. Diagnostic word-trie queries build a small local C++ helper, independently checked against k2 products.",
        "",
    ]
    for name in (
        "error-ledger.jsonl",
        "review-queue.jsonl",
        "counterfactuals.json",
        "calibration.json",
        "interventions.json",
        "lm/models.json",
        "pronunciation/coverage.json",
        "boundary-outcomes.json",
        "research-candidate/candidate.json",
        "experimental-scn.yaml",
        "validation.json",
        "cleanup.json",
        "completion.json",
    ):
        lines.append(f"- [{name}](../../{(output / name).relative_to(root)})")
    lines += [
        "",
        "Diagnostic corrections are never training evidence. Current production configuration and the historical winning bundle are retained unchanged.",
        "",
    ]
    return "\n".join(lines)
