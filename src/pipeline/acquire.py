"""Acquire manifest-listed resources into ignored local paths."""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from common.resources import (
    download_resource,
    extract_members,
    extract_source,
    load_resources,
)
from common.utils import write_json


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "resources",
        nargs="*",
        help="Manifest resource IDs; default: all available except checkpoint",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Include the 1.2 GB checkpoint and report all placeholders",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List URLs and destinations without network access",
    )
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    if args.all and args.resources:
        parser.error("Use either --all or specific resource IDs")
    root = args.root.resolve()
    specs = load_resources(root)["resources"]
    names = args.resources or [
        name for name, spec in specs.items() if args.all or spec["kind"] != "checkpoint"
    ]
    if unknown := set(names) - specs.keys():
        parser.error(f"Unknown resource IDs: {sorted(unknown)}")
    receipt_path = root / ".cache/resource-status.json"
    receipts = json.loads(receipt_path.read_text()) if receipt_path.exists() else {}
    failures = 0
    for name in names:
        spec = specs[name]
        if spec["status"] != "available":
            print(f"WARNING {name}: {spec['status']}: {spec['reason']}", flush=True)
            if spec.get("access_instructions"):
                print(f"  {spec['access_instructions']}", flush=True)
            if name in args.resources:
                failures += 1
            continue
        if args.dry_run:
            print(f"{name}: {spec['url']} -> {spec['path']}")
            continue
        if spec.get("acquisition_handler") == "scn_reviewed":
            from asr.scn.acquisition import acquire

            failures += bool(acquire(root, [name]))
            continue
        print(f"Checking/downloading {name} -> {spec['path']}", flush=True)
        try:
            actual = download_resource(root, spec, receipts.get(name))
            if spec.get("extract_to"):
                extract_source(root, spec, actual["sha256"])
            if spec.get("members"):
                extract_members(root, spec, actual)
            receipts[name] = {
                **actual,
                "revision": spec["revision"],
                "source_url": spec["url"],
                "path": spec["path"],
                "verified_at": datetime.now(timezone.utc).isoformat(),
            }
            write_json(receipt_path, receipts)
            print(
                f"  Verified {actual['size_bytes']:,} bytes; SHA-256 {actual['sha256']}",
                flush=True,
            )
        except Exception as error:
            failures += 1
            print(f"ERROR {name}: {error}", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
