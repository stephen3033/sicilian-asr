"""Explicit dependency-checked retirement inventories and compact experiment history."""

import shutil
from pathlib import Path

from common.artifacts import digest
from common.errors import ResourceError
from common.utils import local_path, write_json
from pipeline.experiment import read, sha, verified_files


def historical_ledger(root, output, legacy):
    experiments = []
    for path in sorted(legacy.glob("*.json")):
        if path.name in {
            "channel.json",
            "training.json",
            "results.json",
            "baseline.json",
        }:
            continue
        data = read(path)
        if not isinstance(data, dict):
            continue
        results = data.get("results", [])
        if not isinstance(results, list):
            results = []
        study_path = (
            Path(data["directory"]) / "study.json" if data.get("directory") else None
        )
        input_identity = None
        if study_path and study_path.is_file():
            study = read(study_path)
            identity = study.get("identity", {})
            input_identity = {
                "key": study.get("key"),
                "receipt_sha256": sha(study_path),
                "dependencies": {
                    k: v
                    for k, v in identity.items()
                    if k.endswith("sha256")
                    or k
                    in {
                        "normalization",
                        "source_study",
                        "parent_manifest",
                        "source_graph",
                        "source_lm",
                    }
                },
                "source_sha256": identity.get("sources", {}),
                "recordings": [
                    {
                        "id": r.get("video", r.get("id")),
                        "split": r.get("split"),
                        "reference_sha256": digest(r.get("reference", "")),
                        "acoustic_dir": r.get("acoustic_dir"),
                    }
                    for r in identity.get("recordings", [])
                ],
            }
        compact = []
        for row in results:
            if not isinstance(row, dict):
                continue
            compact.append(
                {
                    k: row[k]
                    for k in (
                        "id",
                        "kind",
                        "status",
                        "graph",
                        "metrics",
                        "lm_scale",
                        "word_bonus",
                        "settings",
                    )
                    if k in row
                }
            )
            if "metrics" in compact[-1]:
                compact[-1]["metrics"] = {
                    k: {kk: vv for kk, vv in v.items() if kk != "alignment"}
                    for k, v in compact[-1]["metrics"].items()
                    if k in {"wer", "cer"}
                }
        experiments.append(
            {
                "manifest": str(path.relative_to(root)),
                "sha256": sha(path),
                "status": data.get("status", "historical_evidence"),
                "results": compact,
                "failures": data.get("failures", []),
                "policy": data.get("policy", data.get("scope")),
                "directory": data.get("directory"),
                "settings": data.get("settings"),
                "input_identity": input_identity,
                "artifact_state": "historical_receipt; consult deletion inventory for retired paths",
            }
        )
    write_json(
        output / "history.json",
        {
            "schema_version": 1,
            "experiments": experiments,
            "policy": "Past measurements are historical evidence, not a claim that retired artifacts remain replayable.",
        },
    )
    # Preserve the concise hypotheses and conclusions already documented, without
    # duplicated executable snapshots or multi-megabyte word alignments.
    text = (root / "docs/scn/improvement.md").read_text()
    if not (output / "prior-experiment-notes.md").exists():
        (output / "prior-experiment-notes.md").write_text(text)
    return experiments


def inventory(root, bundle, output, legacy):
    verified_files(bundle, read(bundle / "bundle.json"))
    verification = read(output.parent / "baseline-replay/verification.json")
    if verification["status"] != "verified" or verification["bundle_sha256"] != sha(
        bundle / "bundle.json"
    ):
        raise ResourceError(
            "Cleanup requires the independently verified baseline bundle"
        )
    output.mkdir(parents=True, exist_ok=True)
    historical_ledger(root, output, legacy)
    # All runtime manifests use .cache/models for ZIPA/Italian. The old isolated
    # Omnilingual cache is enumerated file-by-file, never deleted as a blind root.
    protected = [
        bundle,
        root / ".venv",
        root / "data",
        root / "resources",
        root / ".cache/models",
        root / ".cache/scn/acquired",
        root / ".cache/scn/tools",
        root / ".cache/scn/youtube",
        root / ".cache/scn/posteriors",
        legacy / "audio",
        legacy / "public",
    ]
    candidates = []
    model_root = root / ".cache/scn-improvement/models"
    candidates.extend(
        (p, "unused_direct_text_model_cache")
        for p in sorted(model_root.iterdir())
        if p.name != "model-worker.lock"
    )
    for name in ("venv", "venv311", "speech-swift"):
        candidates.append(
            (root / ".cache/scn-improvement" / name, "isolated_direct_text_environment")
        )
    for name in (
        "adaptation",
        "gradient-smoke",
        "mps-bucket-repair",
        "alignment-jobs",
        "preflights",
        "repaired-preflights",
        "runs",
        "decoding",
        "coverage-decoding",
        "language-models",
    ):
        candidates.append((legacy / name, "retired_direct_text_experiment"))
    # Historical WFST parent graphs are no longer live inputs. Raw speech and
    # training observation directories are retained explicitly.
    candidates.extend(
        (p, "retired_wfst_experiment")
        for p in sorted(legacy.iterdir())
        if p.is_dir() and (p.name.startswith("wfst") or p.name == "selected-replay")
    )
    entries = []
    for path, reason in candidates:
        if not path.exists():
            continue
        resolved = path.resolve()
        if not resolved.is_relative_to(root.resolve()) or any(
            resolved == p.resolve()
            or resolved.is_relative_to(p.resolve())
            or p.resolve().is_relative_to(resolved)
            for p in protected
        ):
            raise ResourceError(f"Deletion overlaps retained dependencies: {path}")
        files = []
        paths = [path] if path.is_file() else sorted(path.rglob("*"))
        for item in paths:
            if item.is_symlink():
                files.append(
                    {
                        "path": str(item.relative_to(root)),
                        "symlink": str(item.readlink()),
                        "bytes": item.lstat().st_size,
                    }
                )
            elif item.is_file():
                files.append(
                    {
                        "path": str(item.relative_to(root)),
                        "bytes": item.stat().st_size,
                        "sha256": sha(item),
                    }
                )
        entries.append(
            {
                "path": str(path.relative_to(root)),
                "reason": reason,
                "bytes": sum(f["bytes"] for f in files),
                "files": files,
            }
        )
    result = {
        "status": "inventoried",
        "bundle_sha256": sha(bundle / "bundle.json"),
        "baseline_verification_sha256": sha(
            output.parent / "baseline-replay/verification.json"
        ),
        "protected_roots": [str(p.relative_to(root)) for p in protected],
        "entries": entries,
        "bytes": sum(e["bytes"] for e in entries),
        "dependency_policy": "Production resources, raw data, ZIPA, Italian, KenLM tools and exported baseline are retained. Only explicitly enumerated retired experimental roots are eligible.",
    }
    write_json(output / "deletion-inventory.json", result)
    return result


def apply_inventory(root, bundle, output):
    verified_files(bundle, read(bundle / "bundle.json"))
    data = read(output / "deletion-inventory.json")
    if data["bundle_sha256"] != sha(bundle / "bundle.json"):
        raise ResourceError("Retained bundle changed since inventory")
    removed = []
    # Verify all files before the first mutation. Symlink targets are never read
    # or recursively removed; lexical paths remain confined to the recorded root.
    for entry in data["entries"]:
        target = local_path(root, entry["path"])
        if not target.exists():
            continue
        live = {
            str(p.relative_to(root))
            for p in ([target] if target.is_file() else target.rglob("*"))
            if p.is_file() or p.is_symlink()
        }
        if live != {item["path"] for item in entry["files"]}:
            raise ResourceError(
                f"Files were added or removed after deletion inventory: {target}"
            )
        for item in entry["files"]:
            p = root / item["path"]
            if "symlink" in item:
                if not p.is_symlink() or str(p.readlink()) != item["symlink"]:
                    raise ResourceError(f"Deletion inventory changed: {p}")
            elif not p.is_file() or sha(p) != item["sha256"]:
                raise ResourceError(f"Deletion inventory changed: {p}")
    for entry in data["entries"]:
        target = local_path(root, entry["path"])
        if target.is_dir():
            shutil.rmtree(target)
        elif target.exists():
            target.unlink()
        removed.append(
            {"path": entry["path"], "reason": entry["reason"], "bytes": entry["bytes"]}
        )
        write_json(
            output / "cleanup.json",
            {
                "status": "running",
                "removed": removed,
                "bytes_removed": sum(r["bytes"] for r in removed),
            },
        )
    result = {
        "status": "complete",
        "removed": removed,
        "bytes_removed": sum(r["bytes"] for r in removed),
        "inventory_sha256": sha(output / "deletion-inventory.json"),
        "bundle_sha256": sha(bundle / "bundle.json"),
    }
    write_json(output / "cleanup.json", result)
    return result
