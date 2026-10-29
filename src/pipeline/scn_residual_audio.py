"""Frozen-ZIPA segmentation controls and joined-boundary diagnostic observations."""

import sys
from pathlib import Path

from common.artifacts import digest, read, run_native, seal, sha, verified_files
from common.errors import ResourceError
from common.utils import write_json


def boundary_pairs(inventory):
    segments = {(s["video"], s["index"]): s for s in inventory["segments"]}
    pairs = {}
    for error in inventory["errors"]:
        if not error["target"] or not error["near_boundary"]:
            continue
        si, video = error["segment"], error["video"]
        word_count = len(segments[video, si]["text"].split())
        lefts = []
        if error["local_start"] <= 1:
            lefts.append(si - 1)
        if error["local_end"] >= word_count - 2:
            lefts.append(si)
        for left in lefts:
            if (video, left) not in segments or (video, left + 1) not in segments:
                continue
            a, b = segments[video, left], segments[video, left + 1]
            if a["end_sample"] != b["start_sample"]:
                raise ResourceError("Cannot join nonadjacent source audio spans")
            key = f"{video}:{left:03}-{left + 1:03}"
            pair = pairs.setdefault(
                key,
                {
                    "id": key,
                    "video": video,
                    "left": left,
                    "right": left + 1,
                    "start_sample": a["start_sample"],
                    "end_sample": b["end_sample"],
                    "boundary_seconds": a["source_end_seconds"],
                    "forced_cut": a["forced_cut"],
                    "error_ids": [],
                    "diagnostic_only": True,
                    "training_eligible": False,
                },
            )
            pair["error_ids"].append(error["id"])
    return list(pairs.values())


def worker(job):
    import numpy as np
    import soundfile as sf
    import torch

    from asr.wfst.cache import load_graph
    from asr.wfst.compact import decode
    from audio.segmentation import Segment, segment_audio
    from common.resources import load_resources
    from common.settings import SAMPLE_RATE
    from ipa.diagnostics import greedy_ctc
    from ipa.zipa import ZipaAcoustic

    root, bundle, output = (Path(job[k]) for k in ("root", "bundle", "output"))
    config = read(bundle / "config.json")
    manifest = load_resources(root)
    model = ZipaAcoustic(root, config["acoustic"], manifest)
    graph, words = (
        load_graph(bundle / "graph") if job["mode"] == "boundaries" else (None, None)
    )
    decoder = read(bundle / "decoder.json")
    for rec in read(bundle / "recordings.json"):
        video = rec["video"]
        audio = bundle / "audio" / f"{video}.wav"
        samples, rate = sf.read(audio, dtype="float32")
        if rate != SAMPLE_RATE or samples.ndim != 1:
            raise ResourceError(
                "Segmentation control requires the frozen mono 16-kHz waveform"
            )
        directory = output / video
        directory.mkdir(parents=True, exist_ok=True)
        if job["mode"] == "segmentation":
            segments = [
                s.to_dict() for s in segment_audio(samples, job["segmentation"])
            ]
            if (
                segments[0]["start_sample"] != 0
                or segments[-1]["end_sample"] != len(samples)
                or any(
                    a["end_sample"] != b["start_sample"]
                    for a, b in zip(segments, segments[1:])
                )
            ):
                raise ResourceError(
                    "Segmentation does not cover the complete recording"
                )
        else:
            segments = []
            for i, pair in enumerate(p for p in job["pairs"] if p["video"] == video):
                segments.append(
                    {
                        **Segment(
                            i,
                            pair["start_sample"],
                            pair["end_sample"],
                            pair["start_sample"] / rate,
                            pair["end_sample"] / rate,
                            "joined_diagnostic",
                            False,
                        ).to_dict(),
                        "pair": pair,
                    }
                )
        rows, artifacts = [], []
        for segment in segments:
            key = digest(
                {"identity": job["identity"], "video": video, "segment": segment}
            )
            name = f"{segment['index']:05d}"
            receipt_path = directory / f"{name}.receipt.json"
            if receipt_path.exists():
                receipt = read(receipt_path)
                if receipt["key"] != key:
                    raise ResourceError("Segmentation observation inputs changed")
                verified_files(directory, receipt)
                row = read(directory / f"{name}.json")
            else:
                print(job["mode"], video, segment["index"], flush=True)
                observed = model.infer(
                    [
                        torch.from_numpy(
                            samples[
                                segment["start_sample"] : segment["end_sample"]
                            ].copy()
                        )
                    ]
                )
                np.savez_compressed(
                    directory / f"{name}.npz",
                    log_probs=observed.log_probs[0, : int(observed.lengths[0])].numpy(),
                    tokens=np.asarray(observed.tokens),
                    blank_id=observed.blank_id,
                )
                row = {
                    **segment,
                    "path": f"{name}.npz",
                    "greedy_ipa": greedy_ctc(observed)[0],
                }
                if job["mode"] == "boundaries":
                    row.update(decode(graph, observed, words, decoder))
                write_json(directory / f"{name}.json", row)
                seal(
                    directory,
                    receipt_path.name,
                    {"identity": job["identity"], "video": video, "segment": segment},
                    [f"{name}.npz", f"{name}.json"],
                )
            rows.append(row)
            artifacts.extend([f"{name}.npz", f"{name}.json", receipt_path.name])
        manifest_row = {
            "identity": job["identity"],
            "video": video,
            "segments": rows,
            "duration_seconds": len(samples) / rate,
            "artifacts": {p: sha(directory / p) for p in artifacts},
        }
        write_json(directory / "acoustic.json", manifest_row)


def prepare_audio(root, bundle, output, recipe, mode):
    directory = output / (
        "resegmented-posteriors" if mode == "segmentation" else "boundary-probes"
    )
    config = read(bundle / "config.json")
    inventory = read(output / "inventory.json")
    pairs = boundary_pairs(inventory) if mode == "boundaries" else []
    identity = {
        "baseline_bundle": sha(bundle / "bundle.json"),
        "mode": mode,
        "segmentation": recipe["segmentation"],
        "pairs": pairs,
        "acoustic_config": config["acoustic"],
        "acoustic_checkpoint": sha(root / config["acoustic"]["checkpoint_path"]),
        "implementation": {
            name: sha(root / "src" / name)
            for name in (
                "pipeline/scn_residual_audio.py",
                "ipa/zipa.py",
                "ipa/types.py",
                "audio/segmentation.py",
            )
        },
    }
    receipt_path = directory / "observations.receipt.json"
    if receipt_path.exists():
        receipt = read(receipt_path)
        if receipt["key"] != digest(identity):
            raise ResourceError("Frozen-acoustic segmentation study changed")
        verified_files(directory, receipt)
        return directory
    directory.mkdir(parents=True, exist_ok=True)
    write_json(
        directory / "job.json",
        {
            "root": str(root),
            "bundle": str(bundle),
            "output": str(directory),
            "mode": mode,
            "segmentation": recipe["segmentation"],
            "pairs": pairs,
            "identity": identity,
        },
    )
    run_native(
        [
            sys.executable,
            "-m",
            "pipeline.scn_residual_audio",
            str(directory / "job.json"),
        ],
        directory / "worker.log",
    )
    recordings = [
        {**r, "acoustic_dir": r["video"]} for r in read(bundle / "recordings.json")
    ]
    write_json(directory / "recordings.json", recordings)
    names = [
        str(p.relative_to(directory))
        for p in directory.rglob("*")
        if p.is_file() and p not in (receipt_path, directory / "bundle.json")
    ]
    seal(directory, receipt_path.name, identity, names)
    seal(directory, "bundle.json", identity, [*names, receipt_path.name])
    return directory


if __name__ == "__main__":
    worker(read(Path(sys.argv[1])))
