"""Distinguish factor-score improvements from paths in the compiled graph."""

import sys
from pathlib import Path

from common.artifacts import digest, read, run_native, seal, sha, verified_files
from common.errors import ResourceError
from common.utils import write_json


def constrained_compiled_score(graph, acoustic, sequence, scorer):
    import k2
    import torch

    from asr.wfst.compact import compact_frames

    words = k2.linear_fsa([scorer.word_ids[w] for w in sequence])
    # The pinned k2 build requires a true stride-one singleton score tensor.
    words.scores = torch.tensor(words.scores.tolist(), dtype=torch.float32)
    matched = k2.connect(
        k2.intersect(k2.arc_sort(k2.invert(graph[0])), k2.arc_sort(words))
    )
    if matched.num_arcs == 0:
        return {"present": False, "score": None}
    constrained = k2.arc_sort(k2.invert(matched))
    frames, lengths = compact_frames(
        acoustic, scorer.settings.get("acoustic_scale", 1.0)
    )
    dense = k2.DenseFsaVec(
        frames, torch.tensor([[0, 0, int(lengths[0])]], dtype=torch.int32)
    )
    result = k2.intersect_dense(
        k2.create_fsa_vec([constrained]),
        dense,
        output_beam=1000000,
        max_states=15000000,
        max_arcs=100000000,
    )
    total = float(result.get_tot_scores(use_double_scores=True, log_semiring=False)[0])
    return {"present": bool(torch.isfinite(torch.tensor(total))), "score": total}


def worker(job):
    import numpy as np
    import torch

    from asr.wfst.cache import load_graph
    from asr.wfst.diagnostics import SequenceScorer, tolerance
    from ipa.types import AcousticOutput

    torch.set_num_threads(4)
    bundle = Path(job["bundle"])
    inputs = read(bundle / "graph/compile_input.json")
    settings = {**inputs["settings"], **read(bundle / "decoder.json")}
    scorer = SequenceScorer(
        read(bundle / "graph/lexicon_selected.json"),
        inputs["tokens"],
        settings,
        bundle / "graph/restricted.arpa",
        bundle / "graph/G.fst.txt",
    )
    graph, _ = load_graph(bundle / "graph")
    rows = []
    for case in job["cases"]:
        segment = case["segment_data"]
        with np.load(
            bundle / "posteriors" / case["video"] / segment["path"], allow_pickle=False
        ) as saved:
            values = torch.from_numpy(saved["log_probs"].copy()).unsqueeze(0)
        acoustic = AcousticOutput(
            inputs["tokens"], 0, {}, values, torch.tensor([values.shape[1]])
        )
        result = constrained_compiled_score(graph, acoustic, case["corrected"], scorer)
        rows.append(
            {
                "case": case["id"],
                "compiled_path": result,
                "search_error": result["present"]
                and result["score"] > segment["score"] + tolerance(segment["score"]),
                "factor_compiled_difference": case["corrected_scores"]["score"]
                - result["score"]
                if result["present"]
                else None,
            }
        )
    write_json(
        Path(job["output"]) / "result.json", {"status": "complete", "cases": rows}
    )


def audit(bundle, output):
    scores, inventory = read(output / "scores.json"), read(output / "inventory.json")
    segments = {(r["video"], r["index"]): r for r in inventory["segments"]}
    candidates = [
        {**c, "segment_data": segments[c["video"], c["segment"]]}
        for s in scores["segments"]
        for c in s["cases"]
        if "factor_score_advantage_requires_compiled_path_check"
        in c["mechanistic_flags"]
    ]
    directory = output / "search"
    directory.mkdir(parents=True, exist_ok=True)
    identity = {
        "bundle": sha(bundle / "bundle.json"),
        "scores": sha(output / "scores.json"),
        "implementation": sha(__file__),
    }
    receipt = directory / "result.receipt.json"
    if receipt.exists():
        saved = read(receipt)
        if saved["key"] != digest(identity):
            raise ResourceError("Search audit inputs changed")
        verified_files(directory, saved)
        return read(directory / "result.json")
    if candidates:
        write_json(
            directory / "job.json",
            {"bundle": str(bundle), "output": str(directory), "cases": candidates},
        )
        run_native(
            [
                sys.executable,
                "-m",
                "pipeline.scn_residual_search",
                str(directory / "job.json"),
            ],
            directory / "worker.log",
        )
        result = read(directory / "result.json")
    else:
        result = {
            "status": "complete",
            "cases": [],
            "reason": "No corrected factor sequence outscored the decoded baseline; a wider beam cannot select these lower-scoring corrections under unchanged factors.",
        }
    result["tested_counterfactuals"] = scores["cases"]
    result["proven_search_errors"] = sum(r["search_error"] for r in result["cases"])
    write_json(directory / "result.json", result)
    seal(directory, receipt.name, identity, ["result.json"])
    return result


if __name__ == "__main__":
    worker(read(Path(sys.argv[1])))
