#!/usr/bin/env python3
"""Inspect upstream metadata only; never authorize corpus acquisition implicitly."""

import argparse
import json
from pathlib import Path
import time
from urllib.error import HTTPError
from urllib.request import Request, urlopen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repositories", nargs="+")
    parser.add_argument("--output", type=Path, default=Path(".cache/scn/discovery"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    failures = 0
    for repo in args.repositories:
        path = args.output / (repo.replace("/", "--") + ".json")
        if path.exists():
            print(f"Cached {repo}", flush=True)
            continue
        def get(suffix):
            time.sleep(1)
            with urlopen(Request(f"https://api.github.com/repos/{repo}/{suffix}".rstrip("/"),
                                 headers={"User-Agent": "sicilian-asr-resource-research/0.1"}), timeout=30) as response:
                return json.load(response)
        try:
            info = get("")
            commit = get("commits/" + info["default_branch"])
            # Root trees first: large all-language repositories can truncate recursive trees.
            root = get("git/trees/" + commit["sha"])
            locale = next((x for x in root["tree"] if x["path"] == "scn" and x["type"] == "tree"), None)
            tree = get("git/trees/" + (locale["sha"] if locale else commit["sha"]) + "?recursive=1")
            result = {"repository": repo, "revision": commit["sha"],
                      "date": commit["commit"]["committer"]["date"], "license": info.get("license"),
                      "prefix": "scn/" if locale else "", "root": root, "tree": tree}
            path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
            print(repo, commit["sha"], len(tree["tree"]), "truncated", tree["truncated"], flush=True)
        except (HTTPError, OSError) as error:
            failures += 1
            print(repo, type(error).__name__, str(error), flush=True)
            # Failures remain retryable; never save them as successful discovery snapshots.
    return bool(failures)


if __name__ == "__main__":
    raise SystemExit(main())
