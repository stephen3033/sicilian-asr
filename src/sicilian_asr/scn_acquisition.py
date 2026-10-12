"""Permission-reviewed SCN acquisitions, isolated from decoder preparation."""

from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import json
from pathlib import Path
import re
import time
from urllib.error import HTTPError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener, urlopen
from urllib.robotparser import RobotFileParser

from sicilian_asr.resources import ResourceError, download_resource, local_path, verify_file, write_json

USER_AGENT = "sicilian-asr-resource-research/0.1"
CATALOG = "resources/scn/sources/catalog.json"


def load_catalog(root: Path) -> dict:
    catalog = json.loads((root / CATALOG).read_text())
    if catalog.get("schema_version") not in {1, 2} or not isinstance(catalog.get("sources"), dict):
        raise ResourceError("Expected SCN source catalog version 1 or 2")
    for name, spec in catalog["sources"].items():
        if not name.startswith("scn_"):
            raise ResourceError("SCN acquisition cannot manage Italian resource IDs")
        if spec.get("path"):
            path = local_path(root, spec["path"])
            if not path.is_relative_to(root / ".cache/scn/acquired"):
                raise ResourceError("SCN originals must be in .cache/scn/acquired")
        if spec.get("orthography") not in {"dos24", "cademia_pre2024", "other_sicilian", "uncertain_mixed", "reference"}:
            raise ResourceError(f"Missing orthographic classification: {name}")
    return catalog


def eligible(spec: dict) -> bool:
    legacy = spec.get("permissions", {}).get("acquisition") == "allowed"
    audit = spec.get("audit", {}).get("bulk_collection", {})
    # Conditional collection requires an explicit, reviewed satisfaction record.
    permitted = not audit or audit.get("value") == "permitted" or (
        audit.get("value") == "conditional" and audit.get("conditions_satisfied") is True)
    return legacy and permitted and bool(spec.get("url"))


def sync_manifest(root: Path):
    """Register reviewed entries while preserving all unrelated manifest bytes."""
    import yaml
    from sicilian_asr.resources import load_manifest
    catalog = load_catalog(root)
    path = root / "resources/manifest.yaml"
    original = path.read_text()
    before = load_manifest(path)["resources"]
    fields = ("name", "kind", "language", "status", "source_url", "url", "revision", "version", "path",
              "format", "license", "license_url", "citation", "checksum_note", "provenance", "sha256", "md5",
              "size_bytes", "acquired_at", "final_url", "reason", "acquisition_status", "audit")
    result = original
    for name, spec in catalog["sources"].items():
        entry = {k:spec[k] for k in fields if k in spec}
        if entry["status"] != "available":
            entry["path"] = None
        entry.update(acquisition_handler="scn_reviewed", catalog="resources/scn/sources/catalog.json")
        block = "".join("  " + line if line.strip() else line for line in yaml.safe_dump({name:entry}, sort_keys=False, allow_unicode=True).splitlines(keepends=True))
        pattern = re.compile(r"^  " + re.escape(name) + r":\n.*?(?=^  [A-Za-z0-9_]+:\n|\Z)", re.M | re.S)
        if pattern.search(result):
            result = pattern.sub(lambda _:block, result)
        else:
            result = result.rstrip() + "\n" + block
    after = yaml.safe_load(result)["resources"]
    if any(after[k] != v for k,v in before.items() if k not in catalog["sources"]):
        raise ResourceError("SCN manifest synchronization changed an unrelated resource")
    path.write_text(result)
    load_manifest(path)
    print(f"Registered {len(catalog['sources'])} SCN inventory entries")


class PoliteHTTP:
    """Check robots (including AI crawler restrictions), redirect policy, and pacing."""

    def __init__(self, delay=1.0):
        self.delay = delay
        self.last = {}
        self.robots = {}
        self.evidence = {}
        self.final_url = None
        owner = self

        class Redirect(HTTPRedirectHandler):
            def redirect_request(self, req, fp, code, msg, headers, newurl):
                owner.check(newurl)
                return super().redirect_request(req, fp, code, msg, headers, newurl)

        self.opener = build_opener(Redirect())

    def pace(self, host, delay=None):
        delay = max(self.delay, delay or 0)
        time.sleep(max(0, delay - (time.monotonic() - self.last.get(host, 0))))
        self.last[host] = time.monotonic()

    def check(self, url):
        parts = urlsplit(url)
        if parts.scheme != "https" or not parts.netloc or parts.username:
            raise ResourceError("SCN acquisition requires an unauthenticated HTTPS URL")
        origin = f"https://{parts.netloc}"
        if origin not in self.robots:
            robots_url = origin + "/robots.txt"
            self.pace(parts.netloc)
            try:
                # Robot policy itself is public discovery metadata, not corpus content.
                with urlopen(Request(robots_url, headers={"User-Agent": USER_AGENT}), timeout=30) as response:
                    content = response.read().decode("utf-8", errors="replace")
                    status = response.status
            except HTTPError as error:
                if error.code not in (404, 410):
                    raise ResourceError(f"Cannot establish robots policy: {robots_url}: HTTP {error.code}") from error
                content, status = "", error.code
            robot = RobotFileParser()
            robot.parse(content.splitlines())
            self.robots[origin] = robot
            self.evidence[origin] = {"url": robots_url, "status": status, "text": content,
                                     "checked_at": datetime.now(timezone.utc).isoformat()}
        robot = self.robots[origin]
        if not all(robot.can_fetch(agent, url) for agent in (USER_AGENT, "GPTBot")):
            raise ResourceError(f"Robots policy excludes automated acquisition: {url}")
        self.pace(parts.netloc, robot.crawl_delay(USER_AGENT))

    def __call__(self, request, timeout=30):
        request.add_header("User-Agent", USER_AGENT)
        for attempt in range(3):
            self.check(request.full_url)
            try:
                response = self.opener.open(request, timeout=timeout)
                self.final_url = response.geturl()
                return response
            except HTTPError as error:
                if error.code not in (429, 503) or attempt == 2:
                    raise
                retry = error.headers.get("Retry-After", "5")
                try:
                    seconds = float(retry)
                except ValueError:
                    seconds = (parsedate_to_datetime(retry) - datetime.now(timezone.utc)).total_seconds()
                if seconds > 60:
                    raise ResourceError(f"Server requests retry after {seconds:.0f}s; retry acquisition later") from error
                time.sleep(max(1, seconds))


def acquire(root: Path, names=None, *, dry_run=False, offline=False, record=False):
    catalog = load_catalog(root)
    specs = catalog["sources"]
    selected = names or list(specs)
    if set(selected) - specs.keys():
        raise ResourceError(f"Unknown source IDs: {sorted(set(selected) - specs.keys())}")
    receipts_path = root / ".cache/scn/acquisition.json"
    receipts = json.loads(receipts_path.read_text()) if receipts_path.exists() else {}
    http = PoliteHTTP()
    failures = []
    for name in selected:
        spec = specs[name]
        if not eligible(spec):
            print(f"METADATA {name}: {spec.get('reason', 'collection permission unresolved')}", flush=True)
            if names:
                failures.append(name)
            continue
        if dry_run:
            print(f"{name}: {spec['url']} -> {spec['path']}", flush=True)
            continue
        try:
            if not spec.get("sha256") and not record:
                raise ResourceError("Missing pinned SHA-256; first acquisition requires --record")
            path = local_path(root, spec["path"])
            if offline:
                actual = verify_file(path, spec, receipts.get(name))
                print(f"VERIFIED {name}: {actual['size_bytes']} bytes", flush=True)
                continue
            existed = path.exists()
            actual = download_resource(root, {**spec, "status": "available"}, receipts.get(name), opener=http)
            receipt = {**actual, "revision": spec["revision"], "url": spec["url"],
                       "final_url": receipts.get(name, {}).get("final_url", spec["url"]) if existed else http.final_url,
                       "path": spec["path"], "acquired_at": receipts.get(name, {}).get("acquired_at", datetime.now(timezone.utc).isoformat())}
            receipts[name] = receipt
            if record:
                spec.update({key: actual[key] for key in ("sha256", "md5", "size_bytes")})
                spec.update(status="available", acquisition_status="acquired", download_availability="acquired", acquired_at=receipt["acquired_at"],
                            final_url=receipt["final_url"])
                if "audit" in spec:
                    spec["audit"]["bulk_collection"]["outcome"] = "acquired"
                    spec["audit"]["dates"]["acquired_at"] = receipt["acquired_at"]
                write_json(root / CATALOG, catalog)
            write_json(receipts_path, receipts)
            print(f"ACQUIRED {name}: {actual['size_bytes']} bytes", flush=True)
        except (ResourceError, OSError, ValueError) as error:
            failures.append(name)
            print(f"ERROR {name}: {error}", flush=True)
            if record:
                spec.update(acquisition_status="failed", acquisition_error=str(error))
                write_json(root / CATALOG, catalog)
    if http.evidence:
        evidence_path = root / ".cache/scn/robots-evidence.json"
        evidence = json.loads(evidence_path.read_text()) if evidence_path.exists() else {}
        evidence.update(http.evidence)
        write_json(evidence_path, evidence)
    return failures
