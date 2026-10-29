"""Retire superseded diagnostic attempts after validating their replacement."""

import shutil

from common.artifacts import digest, read, sha, verified_files
from common.errors import ResourceError
from common.utils import write_json


def cleanup(root, bundle, output, recipe):
    from pipeline.scn_residual_report import verify_scores

    if read(output / "interventions.json")["status"] != "complete":
        raise ResourceError("Complete intervention selection before retirement")
    verified_files(bundle, read(bundle / "bundle.json"))
    verified_files(
        output / "research-candidate",
        read(output / "research-candidate/candidate.json"),
    )
    scores = read(output / "scores.json")
    verify_scores(output, scores, read(output / "inventory.json"))
    implementation = {
        name: sha(root / "src" / name)
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
    retained = output / "scores" / digest(implementation)[:12]
    for row in scores["segments"]:
        directory = retained / row["segment"].replace(":", "-")
        receipt = read(directory / "result.receipt.json")
        verified_files(directory, receipt)
        if digest(read(directory / "result.json")) != digest(row):
            raise ResourceError(
                "Replacement diagnostics do not match the complete score report"
            )
    manifest = output / "cleanup.json"
    if manifest.exists():
        saved = read(manifest)
        if saved["retained_diagnostics"] != str(retained):
            raise ResourceError("Diagnostic retirement selection changed")
        return saved
    inventory, history = [], []
    for directory in sorted((output / "scores").iterdir()):
        if directory == retained:
            continue
        if (
            directory.is_symlink()
            or not directory.is_dir()
            or directory.resolve().parent != (output / "scores").resolve()
        ):
            raise ResourceError("Unexpected diagnostic retirement target")
        files = []
        for path in sorted(directory.rglob("*")):
            if path.is_symlink():
                raise ResourceError("Refusing a linked diagnostic retirement target")
            if path.is_file():
                files.append(
                    {
                        "path": str(path.relative_to(output)),
                        "sha256": sha(path),
                        "bytes": path.stat().st_size,
                    }
                )
        for path in directory.rglob("result.receipt.json"):
            receipt = read(path)
            history.append(
                {
                    "path": str(path.relative_to(output)),
                    "sha256": sha(path),
                    "identity": receipt["identity"],
                    "artifact_state": "retired; historical evidence only",
                }
            )
        for path in directory.rglob("worker.log"):
            if "Traceback" in path.read_text():
                history.append(
                    {
                        "path": str(path.relative_to(output)),
                        "sha256": sha(path),
                        "failure": path.read_text().split("Traceback", 1)[1][-1200:],
                        "resolution": "Superseded by verified compiled-grammar scoring and exact native word-trie matching.",
                    }
                )
        inventory.append(
            {
                "directory": str(directory.relative_to(output)),
                "reason": "Superseded diagnostic implementation; all 85 replacement segments verified",
                "files": files,
            }
        )
    write_json(
        output / "deletion-inventory.json",
        {
            "status": "verified",
            "retained": str(retained),
            "directories": inventory,
            "retained_dependencies": [
                str(bundle),
                str(output / "research-candidate"),
                str(output / "graphs"),
                str(output / "lm"),
                str(output / "resegmented-posteriors"),
                str(output / "boundary-probes"),
            ],
            "policy": "Retire only diagnosis-development attempts. Keep source data, all controlled comparison inputs/results, production graphs, and research candidate bundles.",
        },
    )
    write_json(
        output / "diagnostic-attempt-history.json",
        {
            "attempts": history,
            "note": "Initial KenLM-only parity and slow lattice-product attempts are superseded. Their results are historical evidence, not the current classification.",
        },
    )
    removed = 0
    for item in inventory:
        for file in item["files"]:
            path = output / file["path"]
            if path.stat().st_size != file["bytes"] or sha(path) != file["sha256"]:
                raise ResourceError("Diagnostic deletion inventory changed")
        directory = output / item["directory"]
        shutil.rmtree(directory)
        removed += sum(f["bytes"] for f in item["files"])
    result = {
        "status": "complete",
        "retained_diagnostics": str(retained),
        "removed_file_bytes": removed,
        "removed_directories": [r["directory"] for r in inventory],
        "inventory_sha256": sha(output / "deletion-inventory.json"),
        "history_sha256": sha(output / "diagnostic-attempt-history.json"),
        "prior_repository_cleanup": str(
            root / recipe["corpus_study"] / "cleanup/cleanup.json"
        ),
        "controlled_candidate_graphs_retained": True,
    }
    write_json(manifest, result)
    return result
