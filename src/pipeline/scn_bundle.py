"""Export and replay a portable winning WFST, before retiring its parent studies."""

import shutil
from pathlib import Path

from common.errors import ResourceError
from common.utils import write_json
from pipeline.experiment import (
    compile_graph,
    decode_bundle,
    read,
    seal,
    sha,
    verified_files,
)


def export_baseline(root, destination, legacy):
    if (destination / "bundle.json").exists():
        verified_files(destination, read(destination / "bundle.json"))
        return destination
    replay = read(legacy / "selected-replay-audit.json")
    if replay["status"] != "verified":
        raise ResourceError("Baseline requires a verified historical replay")
    graph = Path(replay["replay"]["graph"])
    verified_files(graph, read(graph / "graph.json"))
    job = read(next((Path(replay["directory"]) / "jobs").glob("*.json")))
    destination.mkdir(parents=True, exist_ok=True)
    shutil.copytree(graph, destination / "graph", dirs_exist_ok=True)
    recordings = []
    channel = read(legacy / "channel.json")
    for rec in job["recordings"]:
        audio_dir = root / rec["acoustic_dir"]
        verified_files(audio_dir, read(audio_dir / "acoustic.json"))
        target = "posteriors/" + rec["video"]
        shutil.copytree(audio_dir, destination / target, dirs_exist_ok=True)
        recordings.append(
            {
                "video": rec["video"],
                "split": rec["split"],
                "reference": rec["reference"],
                "acoustic_dir": target,
                "cues": rec["cues"],
            }
        )
    for rec in channel["recordings"]:
        source = Path(rec["audio"])
        if sha(source) != rec["audio_sha256"]:
            raise ResourceError("Channel audio changed before export")
        target = destination / "audio" / source.name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        rec["audio"] = "audio/" + source.name
    for rec in channel["training"]:
        rec["audio"] = "audio/" + Path(rec["audio"]).name
    write_json(destination / "channel.json", channel)
    write_json(destination / "recordings.json", recordings)
    write_json(destination / "decoder.json", job["settings"])
    config = read(root / "outputs/scn_diagnosis/study.json")["identity"]["config"]
    write_json(destination / "config.json", config)
    original_rows = []
    for rec in replay["replay"]["recordings"]:
        row = read(rec["result"])
        original_rows.append(
            {"video": rec["video"], "text": row["text"], "metrics": rec["metrics"]}
        )
    write_json(
        destination / "expected.json",
        {"metrics": replay["original_metrics"], "recordings": original_rows},
    )
    for name in (
        "zipa-word-sequences.json",
        "zipa-word-evidence-full.json",
        "zipa-training-audit-full.json",
    ):
        shutil.copyfile(legacy / name, destination / name)
    audit = read(legacy / "zipa-training-audit-full.json")
    shutil.copytree(
        Path(audit["directory"]),
        destination / "training-observations",
        dirs_exist_ok=True,
    )
    # A tokenizer is required for new fallback pronunciations, not for cached replay.
    tokenizer = root / config["acoustic"]["tokenizer_path"]
    shutil.copyfile(tokenizer, destination / "tokenizer.model")
    identity = {
        "schema_version": 1,
        "candidate": replay["candidate"],
        "historical_replay_sha256": sha(legacy / "selected-replay-audit.json"),
        "scope": "Self-contained graph rebuild, cached full-recording replay and channel evidence. Fresh acoustic inference uses the retained ZIPA resources in config.json.",
    }
    seal(
        destination,
        "bundle.json",
        identity,
        [
            str(p.relative_to(destination))
            for p in destination.rglob("*")
            if p.is_file() and p.name != "bundle.json"
        ],
    )
    return destination


def replay_baseline(bundle, output):
    verified_files(bundle, read(bundle / "bundle.json"))
    source = bundle / "graph"
    inputs = read(source / "compile_input.json")
    from asr.scn.dataset import digest

    version = digest(
        {
            name: sha(Path(__file__).parents[1] / name)
            for name in (
                "asr/wfst/graph.py",
                "asr/wfst/build.py",
                "pipeline/experiment.py",
                "asr/wfst/cached.py",
                "asr/wfst/compact.py",
                "asr/wfst/cache.py",
                "asr/wfst/storage.py",
                "asr/scn/orthography.py",
                "common/artifacts.py",
                "evaluation/metrics.py",
            )
        }
    )[:16]
    rebuilt = compile_graph(
        output / version / "rebuilt",
        read(source / "lexicon_selected.json"),
        source / "restricted.arpa",
        inputs["tokens"],
        inputs["settings"],
    )
    result = decode_bundle(
        bundle, rebuilt, output / version / "decode", read(bundle / "decoder.json")
    )
    expected = read(bundle / "expected.json")
    old_text = {r["video"]: r["text"] for r in expected["recordings"]}
    changed = [
        r["video"] for r in result["recordings"] if r["text"] != old_text[r["video"]]
    ]
    same = not changed and result["metrics"] == expected["metrics"]
    receipt = {
        "status": "verified" if same else "mismatch",
        "changed_recordings": changed,
        "metrics": result["metrics"],
        "bundle_sha256": sha(bundle / "bundle.json"),
        "rebuild_receipt_sha256": sha(rebuilt / "graph.json"),
        "decode_receipt_sha256": sha(output / version / "decode/result.receipt.json"),
        "policy": "Rebuilt from bundled ARPA, lexicon and settings; full recordings decoded using only bundled posteriors and references.",
    }
    write_json(output / "verification.json", receipt)
    if not same:
        raise ResourceError("Rebuilt baseline failed exact replay")
    return receipt
