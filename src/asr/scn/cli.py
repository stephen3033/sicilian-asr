"""Acquire, extract, analyze and diagnostically probe reviewed Sicilian resources."""

import argparse
from pathlib import Path

from asr.scn.acquisition import acquire, sync_manifest
from common.errors import ResourceError


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    sub = parser.add_subparsers(dest="command", required=True)
    download = sub.add_parser("acquire")
    download.add_argument("sources", nargs="*")
    download.add_argument("--dry-run", action="store_true")
    download.add_argument("--offline", action="store_true")
    download.add_argument(
        "--record",
        action="store_true",
        help="Record checksums on first permission-reviewed acquisition",
    )
    for name in (
        "extract",
        "analyze",
        "probe-lm",
        "inventory-ipa",
        "sync-manifest",
        "audit",
        "audit-report",
        "audit-validate",
    ):
        sub.add_parser(name)
    metadata = sub.add_parser("audit-access")
    metadata.add_argument("--refresh", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    try:
        if args.command == "acquire":
            return bool(
                acquire(
                    root,
                    args.sources,
                    dry_run=args.dry_run,
                    offline=args.offline,
                    record=args.record,
                )
            )
        if args.command == "sync-manifest":
            sync_manifest(root)
            return 0
        if args.command == "audit-access":
            from asr.scn.discovery import probe

            probe(root, refresh=args.refresh)
            return 0
        if args.command in {"audit", "audit-report", "audit-validate"}:
            from asr.scn.audit import audit, report, validate

            {"audit": audit, "audit-report": report, "audit-validate": validate}[
                args.command
            ](root)
            return 0
        from asr.scn.corpus import analyze, extract, inventory_ipa

        if args.command == "extract":
            extract(root)
        elif args.command == "analyze":
            analyze(root)
        elif args.command == "inventory-ipa":
            inventory_ipa(root)
        else:
            from asr.scn.lm import probe

            probe(root)
        return 0
    except (ResourceError, OSError, ValueError) as error:
        print(f"ERROR: {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
