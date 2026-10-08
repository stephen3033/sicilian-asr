#!/usr/bin/env python3
"""Check local resources offline; --strict fails when selected language resources are missing."""

import argparse
from pathlib import Path

from sicilian_asr.resources import ResourceError, write_json
from sicilian_asr.validation import validate_resources


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--config", choices=("ita", "scn", "ita_scn"), default="ita")
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()
    try:
        report = validate_resources(args.root.resolve(), args.config)
    except (ResourceError, OSError, KeyError, TypeError) as error:
        print(f"ERROR configuration/manifest: {error}")
        return 1
    for name, result in report["resources"].items():
        level = "OK" if result["status"] == "ok" else "ERROR" if result["status"] == "error" else "WARNING"
        print(f"{level} {name}: {result['status']} {result.get('reason', result.get('error', ''))}")
        if result.get("statistics"):
            print(f"  {result['statistics']}")
    for error in report["errors"]:
        print(f"ERROR {error}")
    print(f"Resources ready for {args.config}: {report['resource_ready']}; missing: {report['required_missing']}")
    output = args.root.resolve() / ".cache/validation/resources.json"
    write_json(output, report)
    print(f"Report: {output}")
    return 1 if report["errors"] or (args.strict and not report["resource_ready"]) else 0


if __name__ == "__main__":
    raise SystemExit(main())
