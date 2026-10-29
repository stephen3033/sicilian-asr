"""Frozen-model residual diagnosis and controlled repair validation."""

import json

from common.artifacts import digest, read, seal, sha, verified_files
from common.errors import ResourceError
from common.utils import write_json
from pipeline.experiment import decode_bundle
from pipeline.scn_residual_data import inventory

STAGES = (
    "residual-prepare",
    "residual-score",
    "residual-models",
    "residual-calibrate",
    "residual-audio",
    "residual-repairs",
    "residual-tests",
    "residual-cleanup",
    "residual-validate",
    "residual-report",
)


def prepare(root, bundle, output, recipe):
    protected = output / "production-inputs.json"
    if not protected.exists():
        paths = [
            root / name
            for name in (
                "configs/ita.yaml",
                "configs/scn.yaml",
                "configs/ita_scn.yaml",
                ".cache/scn/selected.json",
            )
        ]
        paths += [
            root / recipe["corpus_study"] / name
            for name in ("best/candidate.json", "experimental-scn.yaml")
        ]
        write_json(
            protected, {str(p.relative_to(root)): sha(p) for p in paths if p.exists()}
        )
    verified_files(bundle, read(bundle / "bundle.json"))
    identity = {
        "bundle": sha(bundle / "bundle.json"),
        "inventory_code": sha(root / "src/pipeline/scn_residual_data.py"),
    }
    receipt = output / "inventory.receipt.json"
    if receipt.exists():
        saved = read(receipt)
        if saved["key"] != digest(identity):
            raise ResourceError("Residual inventory inputs changed")
        verified_files(output, saved)
        return read(output / "inventory.json")
    replay = decode_bundle(
        bundle,
        bundle / "graph",
        output / "baseline-replay",
        read(bundle / "decoder.json"),
    )
    expected = read(bundle / "expected.json")
    if replay["metrics"] != expected["metrics"] or {
        r["video"]: r["text"] for r in replay["recordings"]
    } != {r["video"]: r["text"] for r in expected["recordings"]}:
        raise ResourceError("Residual baseline replay differs")
    data = inventory(
        read(bundle / "recordings.json"),
        replay["recordings"],
        read(bundle / "graph/lexicon_selected.json"),
    )
    if (
        data["target_counts"]
        != {"substitutions": 805, "deletions": 248, "insertions": 131}
        or data["total_errors"] != 1429
    ):
        raise ResourceError(
            "Residual target inventory differs from the requested baseline"
        )
    write_json(output / "inventory.json", data)
    with (output / "errors.jsonl").open("w") as stream:
        for row in data["errors"]:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    seal(output, receipt.name, identity, ["inventory.json", "errors.jsonl"])
    return data


def run(root, bundle, output, recipe, stage):
    output.mkdir(parents=True, exist_ok=True)
    identity = {"recipe": recipe, "baseline": sha(bundle / "bundle.json")}
    if (output / "study.json").exists() and read(output / "study.json")[
        "identity"
    ] != identity:
        raise ResourceError("Residual study changed; use a new output directory")
    write_json(output / "study.json", {"identity": identity, "key": digest(identity)})
    stages = STAGES if stage == "all" else (stage,)
    if any(s not in STAGES for s in stages):
        raise ResourceError("Use a residual stage with a residual recipe")
    for current in stages:
        print("Stage", current, flush=True)
        if current == "residual-prepare":
            prepare(root, bundle, output, recipe)
        elif current == "residual-score":
            from pipeline.scn_residual_scoring import score_all

            data = prepare(root, bundle, output, recipe)
            state = output / "active-scoring.json"
            write_json(state, {"status": "running"})
            try:
                score_all(root, bundle, output, data)
            except BaseException as error:
                write_json(
                    state,
                    {"status": "failed", "error": f"{type(error).__name__}: {error}"},
                )
                raise
            write_json(
                state,
                {"status": "complete", "scores_sha256": sha(output / "scores.json")},
            )
        elif current == "residual-models":
            from pipeline.scn_residual_lm import build_models

            build_models(root, bundle, output, recipe)
        elif current == "residual-tests":
            from pipeline.scn_residual_tests import run_tests

            run_tests(root, bundle, output, recipe)
        elif current == "residual-calibrate":
            from pipeline.scn_residual_tests import calibrate

            calibrate(root, bundle, output, recipe)
        elif current == "residual-audio":
            from pipeline.scn_residual_audio import prepare_audio

            prepare_audio(root, bundle, output, recipe, "segmentation")
            prepare_audio(root, bundle, output, recipe, "boundaries")
        elif current == "residual-repairs":
            from pipeline.scn_residual_tests import run_repairs

            run_repairs(root, bundle, output, recipe)
        elif current == "residual-report":
            from pipeline.scn_residual_report import report

            report(root, bundle, output, recipe)
        elif current == "residual-validate":
            from pipeline.scn_residual_validation import validate

            validate(root, bundle, output, recipe)
        elif current == "residual-cleanup":
            from pipeline.scn_residual_cleanup import cleanup

            cleanup(root, bundle, output, recipe)
