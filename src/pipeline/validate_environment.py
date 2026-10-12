"""Validate dependencies and optionally probe ZIPA frame outputs on CPU/MPS.

This is a bounded synthetic smoke test, not a transcription API or decoder.
Native-kernel probes run in isolated subprocesses, with CPU fallback disabled.
"""

import argparse
import importlib
import json
import os
import platform
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

from common.errors import ResourceError
from common.resources import load_resources
from common.utils import write_json
from ipa.validation import acoustic_worker, compare_frames, external_paths


def dependency_check(root: Path, manifest: dict) -> dict:
    result = {"status": "ok", "packages": {}, "external_sources": {}}
    for name in (
        "pipeline",
        "yaml",
        "numpy",
        "torch",
        "torchaudio",
        "k2",
        "kaldilm",
        "kenlm",
        "lhotse",
        "sentencepiece",
        "soundfile",
    ):
        try:
            importlib.import_module(name)
            distribution = {"pipeline": "sicilian-asr", "yaml": "pyyaml"}.get(
                name, name
            )
            result["packages"][name] = {
                "status": "ok",
                "version": version(distribution),
            }
        except Exception as error:
            result["packages"][name] = {"status": "error", "error": str(error)}
            result["status"] = "error"
    try:
        external_paths(root, manifest)
        for name, resource in (
            ("icefall", "icefall_source"),
            ("zipa_ctc_inference", "zipa_source"),
        ):
            importlib.import_module(name)
            result["external_sources"][name] = manifest["resources"][resource][
                "revision"
            ]
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--acoustic",
        action="store_true",
        help="Run four isolated synthetic ZIPA CPU/MPS probes",
    )
    parser.add_argument(
        "--require-mps",
        action="store_true",
        help="Fail unless the scripted MPS probe matches CPU",
    )
    parser.add_argument(
        "--worker",
        choices=("eager_cpu", "eager_mps", "scripted_cpu", "scripted_mps"),
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args()
    if args.require_mps:
        args.acoustic = True
    root = args.root.resolve()
    manifest = load_resources(root)
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
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "dependencies": dependency_check(root, manifest),
    }
    errors = report["dependencies"]["status"] != "ok"
    print(json.dumps(report, indent=2), flush=True)
    if args.acoustic and not errors:
        report["acoustic"] = {}
        environment = {
            **os.environ,
            "PYTORCH_ENABLE_MPS_FALLBACK": "0",
            "WANDB_MODE": "disabled",
        }
        for mode in ("eager_cpu", "eager_mps", "scripted_cpu", "scripted_mps"):
            print(
                f"Probing ZIPA {mode} (isolated process, no CPU fallback)...",
                flush=True,
            )
            output = directory / f"{mode}.json"
            output.unlink(missing_ok=True)
            output.with_suffix(".npz").unlink(missing_ok=True)
            try:
                process = subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "pipeline.validate_environment",
                        "--root",
                        str(root),
                        "--worker",
                        mode,
                    ],
                    env=environment,
                    capture_output=True,
                    text=True,
                    timeout=240,
                )
                result = (
                    json.loads(output.read_text())
                    if output.exists()
                    else {
                        "status": "error",
                        "error": process.stderr[-4000:],
                        "returncode": process.returncode,
                    }
                )
                if process.returncode != 0 and result["status"] == "ok":
                    result = {
                        "status": "error",
                        "returncode": process.returncode,
                        "error": process.stderr[-4000:],
                    }
            except subprocess.TimeoutExpired:
                result = {
                    "status": "error",
                    "error": "Probe timed out after 240 seconds",
                }
            report["acoustic"][mode] = result
            print(json.dumps({mode: result}, indent=2), flush=True)
        for name, left, right in (
            ("scripted_cpu_vs_eager_cpu", "eager_cpu", "scripted_cpu"),
            ("scripted_mps_vs_scripted_cpu", "scripted_cpu", "scripted_mps"),
        ):
            if all(
                report["acoustic"][mode]["status"] == "ok" for mode in (left, right)
            ):
                report["acoustic"][name] = compare_frames(
                    directory / f"{left}.npz", directory / f"{right}.npz"
                )
        report["mps_validated"] = all(
            report["acoustic"].get(name, {}).get("status") == "ok"
            for name in ("scripted_cpu_vs_eager_cpu", "scripted_mps_vs_scripted_cpu")
        )
        errors = report["acoustic"]["eager_cpu"]["status"] != "ok"
        if args.require_mps and not report["mps_validated"]:
            errors = True
        print(
            f"Scripted MPS validated on smoke input: {report['mps_validated']}",
            flush=True,
        )
    write_json(directory / "environment.json", report)
    print(f"Report: {directory / 'environment.json'}")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
