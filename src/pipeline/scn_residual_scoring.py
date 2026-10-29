"""Resumable native workers for residual-error counterfactual scoring."""

import sys
from pathlib import Path

from common.artifacts import digest, read, run_native, seal, sha, verified_files
from common.errors import ResourceError
from common.utils import write_json


def worker(job):
    import gc

    import numpy as np
    import torch

    from asr.wfst.cache import load_graph
    from asr.wfst.diagnostics import (
        SequenceScorer,
        retained_lattice,
        sequences_in_lattice,
        tolerance,
    )
    from ipa.types import AcousticOutput

    torch.set_num_threads(4)
    bundle, output = Path(job["bundle"]), Path(job["output"])
    segment = job["segment"]
    inputs = read(bundle / "graph/compile_input.json")
    settings = {**inputs["settings"], **read(bundle / "decoder.json")}
    with np.load(
        bundle / "posteriors" / segment["video"] / segment["path"], allow_pickle=False
    ) as saved:
        if saved["tokens"].tolist() != inputs["tokens"]:
            raise ResourceError("Residual acoustic label inventory mismatch")
        values = torch.from_numpy(saved["log_probs"].copy()).unsqueeze(0)
    acoustic = AcousticOutput(
        inputs["tokens"], 0, {}, values, torch.tensor([values.shape[1]])
    )
    scorer = SequenceScorer(
        read(bundle / "graph/lexicon_selected.json"),
        inputs["tokens"],
        settings,
        bundle / "graph/restricted.arpa",
        compiled_grammar=bundle / "graph/G.fst.txt",
    )
    original = segment["text"].split()
    decomposition = scorer.decompose(original, segment)
    if not decomposition["verified"]:
        raise ResourceError(
            f"Baseline factor parity failed: {decomposition['residual']}"
        )
    exact = scorer.score(original, acoustic)
    zero = scorer.score(original, acoustic, pronunciation_scale=0)
    graph, _ = load_graph(bundle / "graph")
    lattice = retained_lattice(graph, acoustic, settings)
    memberships = sequences_in_lattice(
        lattice, [original, *[c["corrected"] for c in job["cases"]]], scorer.word_ids
    )
    membership = memberships[0]
    if not membership["present"] or abs(
        membership["score"] - segment["score"]
    ) > tolerance(segment["score"]):
        raise ResourceError("Retained lattice failed baseline path replay")
    rows = []
    del graph, lattice
    gc.collect()
    for case, member in zip(job["cases"], memberships[1:]):
        scored = scorer.score(case["corrected"], acoustic)
        unweighted = scorer.score(case["corrected"], acoustic, pronunciation_scale=0)
        delta = {
            key: scored[key] - exact[key]
            for key in (
                "acoustic_score",
                "pronunciation_score",
                "lm_score",
                "word_bonus_score",
                "score",
            )
        }
        acoustic_delta = unweighted["acoustic_score"] - zero["acoustic_score"]
        epsilon = tolerance(exact["score"])
        flags = []
        if scored["score"] > segment["score"] + epsilon:
            flags.append("factor_score_advantage_requires_compiled_path_check")
        if delta["score"] < -epsilon and delta["score"] - delta["lm_score"] > epsilon:
            flags.append("lm_score_reversal")
        if (
            delta["score"] < -epsilon
            and acoustic_delta + delta["lm_score"] + delta["word_bonus_score"] > epsilon
        ):
            flags.append("pronunciation_cost_reversal")
        if acoustic_delta < -epsilon:
            flags.append("acoustic_evidence_conflict")
        if abs(acoustic_delta) <= epsilon:
            flags.append("acoustically_tied_legal_sequences")
        rows.append(
            {
                **case,
                "status": "scored",
                "corrected_scores": scored,
                "corrected_zero_pronunciation": unweighted,
                "delta": delta,
                "unweighted_acoustic_delta": acoustic_delta,
                "lattice": member,
                "mechanistic_flags": flags,
                "policy": "Score reversals are diagnostic mechanisms, not a demonstrated deployable repair or a gold speech annotation.",
            }
        )
        write_json(
            output / "progress.json",
            {"completed": len(rows), "total": len(job["cases"])},
        )
        gc.collect()
    write_json(
        output / "result.json",
        {
            "status": "complete",
            "segment": segment["diagnostic_id"],
            "factor_parity": decomposition,
            "baseline_exact": exact,
            "baseline_zero_pronunciation": zero,
            "baseline_lattice": membership,
            "cases": rows,
        },
    )


def score_all(root, bundle, output, inventory):
    source = root / "src"
    implementation = {
        name: sha(source / name)
        for name in (
            "pipeline/scn_residual_scoring.py",
            "asr/wfst/diagnostics.py",
            "asr/wfst/trie_scores.py",
            "asr/wfst/graph.py",
            "asr/wfst/compact.py",
        )
    }
    implementation["resources/native/lattice_trie_scores.cpp"] = sha(
        root / "resources/native/lattice_trie_scores.cpp"
    )
    results = []
    for segment in inventory["segments"]:
        cases = [
            c
            for c in inventory["cases"]
            if c["video"] == segment["video"] and c["segment"] == segment["index"]
        ]
        directory = (
            output
            / "scores"
            / digest(implementation)[:12]
            / segment["diagnostic_id"].replace(":", "-")
        )
        identity = {
            "bundle": sha(bundle / "bundle.json"),
            "segment": segment,
            "cases": cases,
            "implementation": implementation,
        }
        receipt = directory / "result.receipt.json"
        if receipt.exists():
            saved = read(receipt)
            if saved["key"] != digest(identity):
                raise ResourceError(f"Scoring inputs changed: {directory}")
            verified_files(directory, saved)
        else:
            directory.mkdir(parents=True, exist_ok=True)
            write_json(
                directory / "job.json",
                {
                    "bundle": str(bundle),
                    "output": str(directory),
                    "segment": segment,
                    "cases": cases,
                },
            )
            print("Residual segment", segment["diagnostic_id"], len(cases), flush=True)
            run_native(
                [
                    sys.executable,
                    "-m",
                    "pipeline.scn_residual_scoring",
                    str(directory / "job.json"),
                ],
                directory / "worker.log",
            )
            seal(directory, receipt.name, identity, ["result.json"])
        results.append(read(directory / "result.json"))
        write_json(
            output / "scoring-progress.json",
            {"segments": len(results), "total": len(inventory["segments"])},
        )
    report = {
        "status": "complete",
        "segments": results,
        "cases": sum(len(r["cases"]) for r in results),
        "factor_parity_verified": all(r["factor_parity"]["verified"] for r in results),
    }
    write_json(output / "scores.json", report)
    return report


if __name__ == "__main__":
    worker(read(Path(sys.argv[1])))
