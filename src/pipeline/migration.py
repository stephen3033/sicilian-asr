"""Move verified resources out of source directories without downloading data."""

import argparse
import hashlib
import json
import shutil
import tarfile
from pathlib import Path

from common.errors import ResourceError
from common.resources import load_resources, verify_file
from common.settings import RESOURCE_RELOCATIONS
from common.utils import local_path, write_json


def legacy_paths(relative: str) -> list[str]:
    """Return resource locations used before configuration and code were separated."""
    previous = next(
        (old for old, new in RESOURCE_RELOCATIONS if new == relative), relative
    )
    paths = [previous]
    for language in ("ita", "scn"):
        prefix = f"resources/{language}/"
        if previous.startswith(prefix):
            paths.insert(0, f"src/asr/{language}/resources/" + previous[len(prefix) :])
    if relative == ".cache/scn/acquired/orthography-2024.pdf":
        paths += ["src/asr/scn/resources/orthography/Cademia-Siciliana-DOS24-v.2.2.pdf"]
    return [p for p in paths if p != relative]


def migrate(root: Path, *, apply: bool = False, manifest: dict | None = None) -> dict:
    """Preflight every move and receipt; reject conflicts before changing anything."""
    root = root.resolve()
    manifest = load_resources(root) if manifest is None else manifest
    moves, updates, verified = {}, {}, {}

    def candidate(relative, spec):
        destination = local_path(root, relative)
        previous = [local_path(root, p) for p in legacy_paths(relative)]
        present = [p for p in [destination, *previous] if p.exists()]
        if not present:
            return None
        actual = verify_file(present[0], spec)
        for path in present[1:]:
            if verify_file(path, spec) != actual:
                raise ResourceError(f"Conflicting migration destinations: {relative}")
        moves.update({p: destination for p in previous if p.exists()})
        verified[relative] = actual
        return present[0], actual

    for spec in manifest["resources"].values():
        if spec["status"] != "available" or not spec.get("path"):
            continue
        relative = spec["path"]
        found = (
            candidate(relative, spec)
            if legacy_paths(relative) or spec.get("members")
            else None
        )
        if not spec.get("members") or found is None:
            continue
        archive, actual = found
        receipt_path = local_path(root, relative + ".members.json")
        if not receipt_path.is_file():
            continue
        receipt = json.loads(receipt_path.read_text())
        if receipt["archive_sha256"] != actual["sha256"]:
            raise ResourceError(f"Stale archive receipt: {archive}")
        artifacts = {}
        with tarfile.open(archive, "r:gz") as stream:
            for member, target in spec["members"].items():
                expected = next(
                    (
                        receipt["artifacts"][p]
                        for p in [target, *legacy_paths(target)]
                        if p in receipt["artifacts"]
                    ),
                    None,
                )
                if expected is None:
                    raise ResourceError(f"Unrecorded archive member: {member}")
                found = candidate(target, expected)
                if found is None:
                    raise ResourceError(f"Missing derived artifact: {target}")
                _, derived = found
                if not stream.getmember(member).isfile():
                    raise ResourceError(f"Not a regular archive member: {member}")
                with stream.extractfile(member) as data:
                    checksum = hashlib.file_digest(data, "sha256").hexdigest()
                if checksum != derived["sha256"]:
                    raise ResourceError(
                        f"Derived artifact differs from archive: {target}"
                    )
                artifacts[target] = derived
        updated = {**receipt, "artifacts": artifacts}
        if updated != receipt:
            updates[receipt_path] = updated

    unknown = [
        str(p.relative_to(root))
        for p in (root / "src").rglob("*")
        if p.is_file()
        and p.suffix not in {".py", ".pyc"}
        and p.name != ".DS_Store"
        and p not in moves
    ]
    if unknown:
        raise ResourceError(
            f"Unmapped non-code files in src; preserve and relocate explicitly: {unknown}"
        )
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
    backup = root / ".cache/code-only-migration/original-receipts"
    for path in updates:
        saved = backup / path.relative_to(root)
        if not saved.exists():
            saved.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, saved)
    for source, destination in moves.items():
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            source.unlink()  # Both copies passed the same integrity checks.
        else:
            source.rename(destination)
    for path, value in updates.items():
        write_json(path, value)
    report_path = root / ".cache/code-only-migration/migration.json"
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
