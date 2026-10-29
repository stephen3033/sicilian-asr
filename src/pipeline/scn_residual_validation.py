"""Replay, regression, accounting, packaging, and protected-input validation."""

import re
import subprocess
import sys
import zipfile

from common.artifacts import read, run_native, sha, verified_files
from common.errors import ResourceError
from common.utils import write_json


def command(root, arguments, log):
    if arguments[0] == "transcribe":
        run_native(
            [sys.executable, "-m", "pipeline", *arguments, "--root", str(root)], log
        )
        return
    with log.open("w") as stream:
        subprocess.run(
            [sys.executable, "-m", "pipeline", *arguments],
            cwd=root,
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=True,
        )


def validate(root, bundle, output, recipe):
    import numpy as np

    from pipeline.scn_residual_report import verify_scores

    selected = read(output / "interventions.json")
    if selected["status"] != "complete":
        raise ResourceError("Complete the repair trials before validation")
    calibration = read(output / "calibration.json")
    language_models = read(output / "lm/models.json")
    families = {
        "baseline",
        "converted-original",
        *(m["id"] for m in language_models["models"]),
    }
    expected_grid = {
        (family, scale, bonus)
        for family in families
        for scale in recipe["calibration"]["lm_scales"]
        for bonus in recipe["calibration"]["word_bonuses"]
    }
    actual_grid = {
        (r["family"], r["decoder"]["lm_scale"], r["decoder"].get("word_bonus", 0))
        for r in calibration["results"]
    }
    if (
        calibration["status"] != "complete"
        or actual_grid != expected_grid
        or len(actual_grid) != len(calibration["results"])
    ):
        raise ResourceError("Incomplete or duplicated LM calibration grid")
    normalization_grid = {
        (r["decoder"]["lm_scale"], r["decoder"].get("word_bonus", 0))
        for r in selected["results"]
        if r["family"] == "baseline-normalized"
    }
    if normalization_grid != {
        (scale, bonus)
        for scale in recipe["calibration"]["lm_scales"]
        for bonus in recipe["calibration"]["word_bonuses"]
    }:
        raise ResourceError("Incomplete historical normalization control")
    if any(
        "pronunciation_cost_reversal" in c["mechanistic_flags"]
        for s in read(output / "scores.json")["segments"]
        for c in s["cases"]
    ):
        scales = {
            r["cost_scale"]
            for r in selected["results"]
            if r["family"] == "pronunciation-calibration"
        }
        if scales != set(recipe["pronunciation"].get("cost_scales", (0.0, 0.5, 1.5))):
            raise ResourceError("Incomplete pronunciation-cost ablation")
    original = read(output / "production-inputs.json")
    if any(sha(root / name) != expected for name, expected in original.items()):
        raise ResourceError(
            "A protected production or historical-selection artifact changed"
        )
    verified_files(bundle, read(bundle / "bundle.json"))
    inventory, scores = read(output / "inventory.json"), read(output / "scores.json")
    verify_scores(output, scores, inventory)
    cases = {c["id"]: c for c in inventory["cases"]}
    accounted = {e for c in cases.values() for e in c["error_ids"]}
    target = {e["id"] for e in inventory["errors"] if e["target"]}
    if accounted != target or len(target) != 1184:
        raise ResourceError("Missing or extra diagnostic target error IDs")
    verified_files(
        output / "selected-fresh-replay",
        read(output / "selected-fresh-replay/result.receipt.json"),
    )
    if (
        read(output / "selected-fresh-replay/result.json")["metrics"]
        != selected["selected"]["metrics"]
    ):
        raise ResourceError("Fresh replay metric mismatch")

    # Production regression commands use the normal public interface. The
    # baseline reference directories are retained results, never rewritten.
    for language, source, reference, export in (
        (
            "ita",
            "data/acceptance/layout_reproduction/sample.mp4",
            "data/acceptance/layout_reproduction/reference.txt",
            True,
        ),
        ("scn", ".cache/scn-improvement/smoke.wav", None, False),
    ):
        command(
            root, ["prepare", "--config", language], output / f"{language}-prepare.log"
        )
        directory = output / f"{language}-regression"
        if not (directory / "run_metadata.json").exists():
            args = [
                "transcribe",
                "--config",
                language,
                "--input",
                source,
                "--output",
                str(directory),
            ]
            if reference:
                args += ["--reference", reference]
            if export:
                args += ["--export-probabilities"]
            command(root, args, output / f"{language}-regression.log")
        if read(directory / "run_metadata.json")["status"] != "complete":
            raise ResourceError("Incomplete production regression")
    previous = root / recipe["corpus_study"]
    italian = {
        name: sha(output / "ita-regression" / name)
        == sha(previous / "ita-regression" / name)
        for name in (
            "transcription.txt",
            "metrics.json",
            "ipa_greedy.txt",
            "reference_normalized.txt",
        )
    }
    with (
        np.load(output / "ita-regression/probabilities-00000.npz") as actual,
        np.load(previous / "ita-regression/probabilities-00000.npz") as expected,
    ):
        italian["posterior_arrays"] = actual.files == expected.files and all(
            np.array_equal(actual[k], expected[k]) for k in actual.files
        )
    if not all(italian.values()):
        raise ResourceError("Italian production regression differs")
    sicilian = {
        name: sha(output / "scn-regression" / name)
        == sha(previous / "scn-regression" / name)
        for name in ("transcription.txt", "ipa_greedy.txt")
    }
    if not all(sicilian.values()):
        raise ResourceError("Sicilian production regression differs")
    config = output / "experimental-scn.yaml"
    command(
        root, ["prepare", "--config", str(config)], output / "experimental-prepare.log"
    )
    smoke = output / "experimental-smoke"
    if not (smoke / "run_metadata.json").exists():
        command(
            root,
            [
                "transcribe",
                "--config",
                str(config),
                "--input",
                ".cache/scn-improvement/smoke.wav",
                "--output",
                str(smoke),
                "--export-probabilities",
            ],
            output / "experimental-smoke.log",
        )
    metadata = read(smoke / "run_metadata.json")
    if metadata["status"] != "complete":
        raise ResourceError("Incomplete experimental bundle smoke transcription")
    tokens = read(output / "research-candidate/graph/compile_input.json")["tokens"]
    for path in smoke.glob("probabilities-*.npz"):
        with np.load(path) as saved:
            if (
                saved["tokens"].tolist() != tokens
                or not np.isfinite(saved["log_probs"]).all()
            ):
                raise ResourceError("Invalid experimental acoustic probabilities")
    with (output / "pytest.log").open("w") as stream:
        completed = subprocess.run(
            [sys.executable, "-m", "pytest", "-q"],
            check=False,
            cwd=root,
            stdout=stream,
            stderr=subprocess.STDOUT,
        )
    if completed.returncode:
        raise ResourceError("Retained Python regressions failed; inspect pytest.log")
    passed = re.search(r"(\d+) passed", (output / "pytest.log").read_text())
    build = subprocess.run(
        ["uv", "build", "--wheel", "--offline", "--out-dir", str(output / "dist")],
        check=False,
        cwd=root,
        capture_output=True,
        text=True,
    )
    (output / "wheel.log").write_text(build.stdout + build.stderr)
    if build.returncode:
        raise ResourceError("Wheel build failed")
    wheel = next((output / "dist").glob("*.whl"))
    with zipfile.ZipFile(wheel) as archive:
        if "sicilian_asr_native/lattice_trie_scores.cpp" not in archive.namelist():
            raise ResourceError("Installed diagnostic helper resource is missing")
    report = {
        "status": "verified",
        "target_errors": len(target),
        "counterfactuals": len(cases),
        "factor_parity_segments": len(scores["segments"]),
        "italian": italian,
        "sicilian": sicilian,
        "production_and_historical_selection_unchanged": all(
            sha(root / n) == h for n, h in original.items()
        ),
        "python_tests_passed": int(passed[1]) if passed else None,
        "wheel_sha256": sha(wheel),
        "experimental_bundle": {
            "candidate_sha256": sha(output / "research-candidate/candidate.json"),
            "smoke_metadata_sha256": sha(smoke / "run_metadata.json"),
            "finite_posteriors_and_matching_labels": True,
        },
        "implementation_sha256": sha(__file__),
        "independent_acceptance": "pending_fresh_recordings",
    }
    write_json(output / "validation.json", report)
    return report
