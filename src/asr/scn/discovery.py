"""Bounded, resumable metadata checks. Never grants collection permission."""

import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from urllib.parse import urlsplit
from urllib.request import Request

from asr.scn.acquisition import PoliteHTTP, load_catalog
from common.settings import SCN_EVIDENCE
from common.utils import write_json


def probe(root, *, refresh=False):
    path = root / SCN_EVIDENCE / "sources/access-audit.json"
    previous = json.loads(path.read_text()) if path.exists() else {"sources": {}}
    sources = load_catalog(root)["sources"]
    hosts = {}
    for name, spec in sources.items():
        if (
            not refresh
            and previous["sources"].get(name, {}).get("url") == spec["source_url"]
        ):
            continue
        hosts.setdefault(urlsplit(spec["source_url"]).netloc, []).append(
            (name, spec["source_url"])
        )

    def check_host(items):
        http, result, seen = PoliteHTTP(), {}, {}
        for name, url in items:
            if url in seen:
                result[name] = dict(seen[url])
                continue
            row = {
                "url": url,
                "checked_on": datetime.now(timezone.utc).date().isoformat(),
                "method": "HEAD after robots; one landing URL, no pagination or corpus copying",
            }
            try:
                with http(Request(url, method="HEAD"), timeout=20) as response:
                    row.update(
                        outcome="reachable",
                        http_status=response.status,
                        final_url=response.geturl(),
                        content_type=response.headers.get("Content-Type"),
                    )
            except Exception as error:
                row.update(outcome="access_attempt_failed", explanation=str(error))
            row["robots"] = dict(http.evidence)
            seen[url] = row
            result[name] = row
            print(name, row["outcome"], flush=True)
        return result

    with ThreadPoolExecutor(max_workers=6) as pool:
        for result in pool.map(check_host, hosts.values()):
            previous["sources"].update(result)
            write_json(path, previous)
    previous.update(
        schema_version=1,
        scope="One source landing HEAD per distinct URL; robots plus redirects; no exhaustive web claim",
    )
    write_json(path, previous)
    return previous
