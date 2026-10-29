"""Shared, receipt-bound WFST experiment scoring and worker orchestration."""

import sys
from pathlib import Path

from asr.scn.orthography import normalize
from asr.wfst.build import compile_graph as compile_graph
from asr.wfst.cached import decode_video
from common.artifacts import digest, read, run_native, seal, sha, verified_files
from common.errors import ResourceError
from common.utils import local_path, write_json
from evaluation.metrics import edits

SPECIAL = {"<s>", "</s>", "<unk>"}


def score(text, reference):
    ref, hyp = normalize(reference), normalize(text)
    wer, alignment = edits(ref.split(), hyp.split())
    cer, _ = edits(list(ref.replace(" ", "")), list(hyp.replace(" ", "")))
    return {"wer": wer, "cer": cer, "word_alignment": alignment}


def aggregate(rows):
    result = {}
    for name in ("wer", "cer"):
        counts = {
            key: sum(r["metrics"][name][key] for r in rows)
            for key in (
                "errors",
                "reference_units",
                "substitutions",
                "deletions",
                "insertions",
                "correct",
            )
        }
        counts["rate"] = (
            counts["errors"] / counts["reference_units"]
            if counts["reference_units"]
            else None
        )
        result[name] = counts
    return result


def improves(candidate, baseline):
    cw, cc = (candidate[n]["rate"] for n in ("wer", "cer"))
    bw, bc = (baseline[n]["rate"] for n in ("wer", "cer"))
    return (
        cw is not None
        and cc is not None
        and ((cw < bw and cc <= bc) or (cw == bw and cc < bc))
    )


def decode_bundle(bundle, graph, output, settings, *, fresh=False):
    """Score complete frozen recordings. Relative bundle paths are the only inputs."""
    bundle, graph, output = map(lambda p: Path(p).resolve(), (bundle, graph, output))
    manifest = read(bundle / "bundle.json")
    verified_files(bundle, manifest)
    verified_files(graph, read(graph / "graph.json"))
    source = Path(__file__).parents[1]
    identity = {
        "bundle": sha(bundle / "bundle.json"),
        "graph": sha(graph / "graph.json"),
        "settings": settings,
        "runtime": {
            name: sha(source / name)
            for name in (
                "pipeline/experiment.py",
                "asr/wfst/cached.py",
                "asr/wfst/compact.py",
                "asr/wfst/cache.py",
                "asr/wfst/storage.py",
                "asr/scn/orthography.py",
                "common/artifacts.py",
                "evaluation/metrics.py",
            )
        },
    }
    receipt = output / "result.receipt.json"
    if receipt.exists() and not fresh:
        saved = read(receipt)
        if saved["key"] != digest(identity):
            raise ResourceError(f"Decode inputs changed: {output}")
        verified_files(output, saved)
        return read(output / "result.json")
    output.mkdir(parents=True, exist_ok=True)
    job = {
        "bundle": str(bundle),
        "graph": str(graph),
        "output": str(output),
        "settings": settings,
    }
    write_json(output / "job.json", job)
    run_native(
        [sys.executable, "-m", "pipeline.experiment", str(output / "job.json")],
        output / "decode.log",
    )
    seal(output, receipt.name, identity, ["result.json"])
    return read(output / "result.json")


def worker(job):
    import torch

    torch.set_num_threads(4)
    bundle, graph = Path(job["bundle"]), Path(job["graph"])
    rows = []
    for rec in read(bundle / "recordings.json"):
        acoustic_dir = local_path(bundle, rec["acoustic_dir"])
        acoustic = read(acoustic_dir / "acoustic.json")
        verified_files(acoustic_dir, acoustic)
        decoded = decode_video(graph, acoustic_dir, acoustic, job["settings"])
        rows.append(
            {
                "video": rec["video"],
                **decoded,
                "metrics": score(decoded["text"], rec["reference"]),
            }
        )
        print(rec["video"], rows[-1]["metrics"]["wer"]["rate"], flush=True)
    write_json(
        Path(job["output"]) / "result.json",
        {"status": "complete", "metrics": aggregate(rows), "recordings": rows},
    )


if __name__ == "__main__":
    worker(read(Path(sys.argv[1])))
