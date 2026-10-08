#!/usr/bin/env python3
"""Validate dependencies and optionally probe ZIPA frame outputs on CPU/MPS.

This is a bounded synthetic smoke test, not a transcription API or decoder.
Native-kernel probes run in isolated subprocesses, with CPU fallback disabled.
"""

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import importlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import traceback

from sicilian_asr.resources import ResourceError, load_manifest, local_path, verify_file, write_json


def external_paths(root: Path, manifest: dict) -> None:
    for name in ("icefall_source", "zipa_source"):
        spec = manifest["resources"][name]
        archive = local_path(root, spec["path"])
        actual = verify_file(archive, spec)
        directory = local_path(root, spec["extract_to"])
        marker = directory / ".sicilian-asr-source.json"
        if not marker.is_file() or json.loads(marker.read_text()) != {
            "revision": spec["revision"], "archive_sha256": actual["sha256"]
        }:
            raise ResourceError(f"Acquire the pinned {name} with scripts/download_resources.py")
        sys.path.insert(0, str(directory))
    sys.path.insert(0, str(root / ".cache/external/zipa/zipformer_crctc"))
    os.environ["WANDB_MODE"] = "disabled"


def dependency_check(root: Path, manifest: dict) -> dict:
    result = {"status": "ok", "packages": {}, "external_sources": {}}
    for name in ("sicilian_asr", "yaml", "numpy", "torch", "torchaudio", "k2", "kenlm", "lhotse", "sentencepiece", "soundfile"):
        try:
            importlib.import_module(name)
            distribution = {"sicilian_asr": "sicilian-asr", "yaml": "pyyaml"}.get(name, name)
            result["packages"][name] = {"status": "ok", "version": version(distribution)}
        except Exception as error:
            result["packages"][name] = {"status": "error", "error": str(error)}
            result["status"] = "error"
    try:
        external_paths(root, manifest)
        for name, resource in (("icefall", "icefall_source"), ("zipa_ctc_inference", "zipa_source")):
            importlib.import_module(name)
            result["external_sources"][name] = manifest["resources"][resource]["revision"]
        import k2
        import torch

        result["k2_torch_version"] = k2.k2_torch_version
        if k2.k2_torch_version.split("+")[0] != torch.__version__.split("+")[0]:
            raise ResourceError("k2 and installed PyTorch versions differ")
        # Exercise a native CPU kernel without building or composing a decoding graph.
        if not torch.isfinite(k2.swoosh_l_forward(torch.zeros(3))).all():
            raise ResourceError("k2 CPU native kernel failed")
        result["mps_built"] = torch.backends.mps.is_built()
        result["mps_available"] = torch.backends.mps.is_available()
    except Exception as error:
        result["status"] = "error"
        result["error"] = str(error)
    return result


def acoustic_worker(root: Path, manifest: dict, mode: str, output: Path) -> dict:
    import resource

    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    external_paths(root, manifest)
    import numpy as np
    import torch
    from lhotse.features.kaldi.extractors import Fbank
    from zipa_ctc_inference import large_params
    from zipformer_crctc.train import get_model

    device = "mps" if mode.endswith("mps") else "cpu"
    if device == "mps" and not torch.backends.mps.is_available():
        return {"status": "unavailable", "reason": "MPS is not visible to this process; check hardware and sandbox permissions."}
    torch.manual_seed(0)
    torch.set_num_threads(4)
    spec = manifest["resources"]["zipa_checkpoint"]
    checkpoint = local_path(root, spec["path"])
    verify_file(checkpoint, spec)
    model = get_model(deepcopy(large_params))
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    del state
    model.eval().requires_grad_(False)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())

    # Generated signal exercises numerical/device compatibility, never linguistic accuracy.
    audio = [0.05 * torch.sin(2 * torch.pi * 220 * torch.arange(n) / 16000) for n in (16000, 24000)]
    features = Fbank().extract_batch(audio, sampling_rate=16000)
    feature_lengths = torch.tensor([feature.shape[0] for feature in features], dtype=torch.int64)
    features = torch.nn.utils.rnn.pad_sequence(features, batch_first=True)
    if mode.startswith("scripted"):
        # Use the existing upstream inference conversion and encoder export wrapper.
        from scaling_converter import convert_scaled_to_non_scaled
        from export import EncoderModel

        convert_scaled_to_non_scaled(model, inplace=True)
        encoder = torch.jit.script(EncoderModel(model.encoder, model.encoder_embed).eval())
        encoder = encoder.to(device)
        ctc_head = model.ctc_output.to(device)
        forward = encoder
    else:
        model.to(device)
        forward = model.forward_encoder
        ctc_head = model.ctc_output
    with torch.inference_mode():
        encoder_out, lengths = forward(features.to(device), feature_lengths.to(device))
        log_probs = ctc_head(encoder_out).cpu()
        lengths = lengths.cpu()
    if log_probs.ndim != 3 or log_probs.shape[0] != 2 or log_probs.shape[-1] != 127:
        raise ResourceError(f"Unexpected frame output shape: {tuple(log_probs.shape)}")
    if not torch.isfinite(log_probs).all():
        raise ResourceError("Nonfinite CTC frame output")
    if not torch.all((lengths > 0) & (lengths <= log_probs.shape[1])) or lengths[0] == lengths[1]:
        raise ResourceError(f"Invalid unequal-batch frame lengths: {lengths}")
    valid = torch.arange(log_probs.shape[1])[None, :] < lengths[:, None]
    normalization_error = log_probs.logsumexp(-1)[valid].abs().max().item()
    if normalization_error > 1e-4:
        raise ResourceError(f"CTC outputs are not normalized log probabilities: {normalization_error}")
    np.savez(output.with_suffix(".npz"), log_probs=log_probs.numpy(), lengths=lengths.numpy())
    return {
        "status": "ok", "device": device, "mode": mode, "parameter_count": parameter_count,
        "shape": list(log_probs.shape), "lengths": lengths.tolist(),
        "normalization_max_abs_error": normalization_error,
        "checkpoint_sha256": spec["sha256"], "cpu_fallback_enabled": False,
    }


def compare_frames(reference: Path, candidate: Path) -> dict:
    import numpy as np

    with np.load(reference) as left, np.load(candidate) as right:
        if left["log_probs"].shape != right["log_probs"].shape or not np.array_equal(left["lengths"], right["lengths"]):
            return {"status": "error", "reason": "Output shape or lengths differ"}
        mask = np.arange(left["log_probs"].shape[1])[None, :] < left["lengths"][:, None]
        a, b = left["log_probs"][mask], right["log_probs"][mask]
        equivalent = bool(np.allclose(a, b, rtol=1e-3, atol=1e-3))
        return {"status": "ok" if equivalent else "error", "max_abs_error": float(np.abs(a - b).max()), "rtol": 1e-3, "atol": 1e-3}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--acoustic", action="store_true", help="Run four isolated synthetic ZIPA CPU/MPS probes")
    parser.add_argument("--require-mps", action="store_true", help="Fail unless the scripted MPS probe matches CPU")
    parser.add_argument("--worker", choices=("eager_cpu", "eager_mps", "scripted_cpu", "scripted_mps"), help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.require_mps:
        args.acoustic = True
    root = args.root.resolve()
    manifest = load_manifest(root / "resources/manifest.yaml")
    directory = root / ".cache/validation"
    directory.mkdir(parents=True, exist_ok=True)
    if args.worker:
        output = directory / f"{args.worker}.json"
        try:
            result = acoustic_worker(root, manifest, args.worker, output)
        except Exception:
            result = {"status": "error", "error": traceback.format_exc()}
        write_json(output, result)
        return 0 if result["status"] == "ok" else 1

    report = {
        "checked_at": datetime.now(timezone.utc).isoformat(), "python": platform.python_version(),
        "platform": platform.platform(), "dependencies": dependency_check(root, manifest),
    }
    errors = report["dependencies"]["status"] != "ok"
    print(json.dumps(report, indent=2), flush=True)
    if args.acoustic and not errors:
        report["acoustic"] = {}
        environment = {**os.environ, "PYTORCH_ENABLE_MPS_FALLBACK": "0", "WANDB_MODE": "disabled"}
        for mode in ("eager_cpu", "eager_mps", "scripted_cpu", "scripted_mps"):
            print(f"Probing ZIPA {mode} (isolated process, no CPU fallback)...", flush=True)
            output = directory / f"{mode}.json"
            output.unlink(missing_ok=True)
            output.with_suffix(".npz").unlink(missing_ok=True)
            try:
                process = subprocess.run(
                    [sys.executable, str(Path(__file__).resolve()), "--root", str(root), "--worker", mode],
                    env=environment, capture_output=True, text=True, timeout=240,
                )
                result = json.loads(output.read_text()) if output.exists() else {
                    "status": "error", "error": process.stderr[-4000:], "returncode": process.returncode,
                }
                if process.returncode != 0 and result["status"] == "ok":
                    result = {"status": "error", "returncode": process.returncode, "error": process.stderr[-4000:]}
            except subprocess.TimeoutExpired:
                result = {"status": "error", "error": "Probe timed out after 240 seconds"}
            report["acoustic"][mode] = result
            print(json.dumps({mode: result}, indent=2), flush=True)
        for name, left, right in (
            ("scripted_cpu_vs_eager_cpu", "eager_cpu", "scripted_cpu"),
            ("scripted_mps_vs_scripted_cpu", "scripted_cpu", "scripted_mps"),
        ):
            if all(report["acoustic"][mode]["status"] == "ok" for mode in (left, right)):
                report["acoustic"][name] = compare_frames(directory / f"{left}.npz", directory / f"{right}.npz")
        report["mps_validated"] = all(
            report["acoustic"].get(name, {}).get("status") == "ok"
            for name in ("scripted_cpu_vs_eager_cpu", "scripted_mps_vs_scripted_cpu")
        )
        errors = report["acoustic"]["eager_cpu"]["status"] != "ok"
        if args.require_mps and not report["mps_validated"]:
            errors = True
        print(f"Scripted MPS validated on smoke input: {report['mps_validated']}", flush=True)
    write_json(directory / "environment.json", report)
    print(f"Report: {directory / 'environment.json'}")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
