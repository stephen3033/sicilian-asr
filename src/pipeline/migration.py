"""Move verified local resources from the former layout without downloading data."""

import argparse
import hashlib
import json
import shutil
import tarfile
from pathlib import Path

from common.errors import ResourceError
from common.resources import load_resources, verify_file
from common.utils import local_path, write_json


def old_resource_path(relative: str) -> str:
    for language in ("ita", "scn"):
        relative = relative.replace(
            f"src/asr/{language}/resources/", f"resources/{language}/"
        )
    return relative


def migrate(root: Path, *, apply: bool = False, manifest: dict | None = None) -> dict:
    """Preflight all bytes and receipts before any move; repeated runs are safe."""
    root = root.resolve()
    manifest = load_resources(root) if manifest is None else manifest
    moves = {}
    updates = {}
    verified = {}

    def candidate(relative, spec):
        destination = local_path(root, relative)
        old = local_path(root, old_resource_path(relative))
        existing = destination if destination.exists() else old
        if not existing.exists():
            return None
        actual = verify_file(existing, spec)
        if old != destination and old.exists():
            previous = verify_file(old, spec)
            if previous != actual:
                raise ResourceError(f"Conflicting migration destinations: {relative}")
            moves[old] = destination
        verified[relative] = actual
        return actual

    for spec in manifest["resources"].values():
        if spec["status"] != "available" or not spec.get("path"):
            continue
        relative = spec["path"]
        # Archives are verified below when their member receipts need migration.
        if old_resource_path(relative) != relative:
            candidate(relative, spec)
        if not spec.get("members"):
            continue
        receipt_path = local_path(root, relative + ".members.json")
        if not receipt_path.exists():
            continue
        receipt = json.loads(receipt_path.read_text())
        archive = local_path(root, relative)
        actual = verify_file(archive, spec)
        if receipt["archive_sha256"] != actual["sha256"]:
            raise ResourceError(f"Stale archive receipt: {archive}")
        artifacts = {}
        with tarfile.open(archive, "r:gz") as stream:
            for member, target in spec["members"].items():
                expected = receipt["artifacts"].get(target)
                if expected is None:
                    expected = receipt["artifacts"].get(old_resource_path(target))
                if expected is None:
                    raise ResourceError(f"Unrecorded archive member: {member}")
                derived = candidate(target, expected)
                if derived is None:
                    raise ResourceError(f"Missing derived artifact: {target}")
                info = stream.getmember(member)
                if not info.isfile():
                    raise ResourceError(f"Not a regular archive member: {member}")
                with stream.extractfile(info) as data:
                    checksum = hashlib.file_digest(data, "sha256").hexdigest()
                if checksum != derived["sha256"]:
                    raise ResourceError(
                        f"Derived artifact differs from archive: {target}"
                    )
                artifacts[target] = derived
        updated = {**receipt, "artifacts": artifacts}
        if updated != receipt:
            updates[receipt_path] = updated

    status_path = root / ".cache/resource-status.json"
    if status_path.exists():
        receipts = json.loads(status_path.read_text())
        updated = json.loads(status_path.read_text())
        for name, receipt in updated.items():
            spec = manifest["resources"].get(name)
            if spec and spec.get("path") in verified:
                if receipt.get("sha256") != verified[spec["path"]]["sha256"]:
                    raise ResourceError(f"Local receipt checksum mismatch: {name}")
                receipt["path"] = spec["path"]
        if updated != receipts:
            updates[status_path] = updated

    report = {
        "applied": apply,
        "moves": {
            str(a.relative_to(root)): str(b.relative_to(root)) for a, b in moves.items()
        },
        "receipts": [str(p.relative_to(root)) for p in updates],
        "verified": verified,
    }
    if not apply:
        return report

    backup = root / ".cache/layout-migration/original-receipts"
    for path in updates:
        saved = backup / path.relative_to(root)
        if not saved.exists():
            saved.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, saved)
    for source, destination in moves.items():
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            # Both copies passed the same integrity checks above.
            source.unlink()
        else:
            source.rename(destination)
    for path, value in updates.items():
        write_json(path, value)
    # Keep the first migration receipt; a subsequent no-op must not replace it.
    report_path = root / ".cache/layout-migration/migration.json"
    if not report_path.exists() or moves or updates:
        write_json(report_path, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--apply", action="store_true", help="Move verified files and update receipts"
    )
    mode.add_argument(
        "--dry-run", action="store_true", help="Verify and show moves (the default)"
    )
    args = parser.parse_args()
    report = migrate(args.root, apply=args.apply)
    print(json.dumps({k: v for k, v in report.items() if k != "verified"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
