"""Reproducible acquisition of upstream artifacts, without ASR processing."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tarfile
import tempfile
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

import yaml

from common.errors import ResourceError
from common.utils import fingerprints, local_path, write_json


def load_manifest(path: Path) -> dict:
    manifest = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
        raise ResourceError("Expected resource manifest schema_version: 1")
    resources = manifest.get("resources")
    if not isinstance(resources, dict) or not resources:
        raise ResourceError("Manifest resources must be a nonempty mapping")
    required = {
        "kind",
        "language",
        "status",
        "source_url",
        "revision",
        "version",
        "path",
        "format",
        "license",
        "license_url",
        "citation",
        "checksum_note",
    }
    for name, spec in resources.items():
        if not isinstance(spec, dict) or required - spec.keys():
            raise ResourceError(f"Incomplete manifest entry: {name}")
        if spec["language"] not in {"ita", "scn", "mul"}:
            raise ResourceError(f"Non-ISO language code: {name}")
        if spec["status"] not in {"available", "unavailable", "planned"}:
            raise ResourceError(f"Unknown resource status: {name}")
        for field in ("version", "format", "license", "citation", "checksum_note"):
            if not isinstance(spec[field], str) or not spec[field].strip():
                raise ResourceError(f"Missing {field}: {name}")
        if spec["status"] == "available":
            for field in ("source_url", "url", "license_url"):
                if not str(spec.get(field, "")).startswith("https://"):
                    raise ResourceError(f"Missing HTTPS {field}: {name}")
            provenance = spec.get("provenance", "git")
            if provenance == "git":
                if not re.fullmatch(r"[0-9a-f]{40}", str(spec["revision"])):
                    raise ResourceError(
                        f"Available resource needs a pinned revision: {name}"
                    )
            elif provenance == "academic_archive":
                if not spec.get("revision") or not (
                    spec.get("md5") or spec.get("sha256")
                ):
                    raise ResourceError(
                        f"Academic archive needs a release and checksum: {name}"
                    )
            elif provenance == "http_snapshot":
                if not all(
                    spec.get(k)
                    for k in ("revision", "sha256", "acquired_at", "final_url")
                ):
                    raise ResourceError(
                        f"HTTP snapshot needs identity, checksum and acquisition receipt: {name}"
                    )
                if not str(spec["final_url"]).startswith("https://"):
                    raise ResourceError(f"HTTP snapshot has insecure final URL: {name}")
            else:
                raise ResourceError(f"Unknown provenance type: {name}")
            if not spec["path"]:
                raise ResourceError(f"Available resource needs a path: {name}")
        elif not spec.get("reason"):
            raise ResourceError(f"Missing unavailable/planned explanation: {name}")
        for field, length in (("sha256", 64), ("git_blob_sha1", 40), ("md5", 32)):
            if spec.get(field) and not re.fullmatch(
                rf"[0-9a-f]{{{length}}}", spec[field]
            ):
                raise ResourceError(f"Invalid {field}: {name}")
    return manifest


def verify_file(path: Path, spec: dict, receipt: dict | None = None) -> dict:
    if not path.is_file() or path.stat().st_size == 0:
        raise ResourceError(f"Missing or empty artifact: {path}")
    actual = fingerprints(path)
    for field in ("size_bytes", "sha256", "git_blob_sha1", "md5"):
        expected = spec.get(field)
        if expected is not None and actual[field] != expected:
            raise ResourceError(
                f"{field} mismatch: {path.name}; expected {expected}, got {actual[field]}"
            )
    if receipt and receipt.get("revision") == spec["revision"]:
        if receipt.get("sha256") and actual["sha256"] != receipt["sha256"]:
            raise ResourceError(f"Local receipt checksum mismatch: {path.name}")
    return actual


def download_resource(
    root: Path, spec: dict, receipt: dict | None = None, *, opener=urlopen
) -> dict:
    """Download atomically or verify existing bytes. Never fabricate a missing resource."""
    if spec["status"] != "available":
        raise ResourceError(spec["reason"])
    path = local_path(root, spec["path"])
    if path.exists():
        return verify_file(path, spec, receipt)
    path.parent.mkdir(parents=True, exist_ok=True)
    request = Request(
        spec["url"],
        headers={"User-Agent": "sicilian-asr/0.1 research-resource-acquisition"},
    )
    for attempt in range(3):
        temporary = None
        try:
            with opener(request, timeout=30) as response:
                if not response.geturl().startswith("https://"):
                    raise ResourceError("Refusing an insecure download redirect")
                if response.status != 200:
                    raise ResourceError(f"Unexpected HTTP status: {response.status}")
                expected_length = response.headers.get("Content-Length")
                with tempfile.NamedTemporaryFile(
                    dir=path.parent, suffix=".part", delete=False
                ) as stream:
                    temporary = Path(stream.name)
                    for chunk in iter(lambda: response.read(1024 * 1024), b""):
                        stream.write(chunk)
                if expected_length and temporary.stat().st_size != int(expected_length):
                    raise ResourceError(f"Incomplete HTTP body: {path.name}")
            actual = verify_file(temporary, spec, receipt)
            os.replace(temporary, path)
            return actual
        except (OSError, URLError):
            if attempt == 2:
                raise
            time.sleep(attempt + 1)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    raise AssertionError("Unreachable")


def extract_source(root: Path, spec: dict, sha256: str) -> Path:
    destination = local_path(root, spec["extract_to"])
    stamp = {"revision": spec["revision"], "archive_sha256": sha256}
    marker = destination / ".sicilian-asr-source.json"
    if destination.exists():
        if not marker.is_file() or json.loads(marker.read_text()) != stamp:
            raise ResourceError(
                f"Source cache has a different or unknown revision: {destination}"
            )
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination.parent) as temporary:
        with tarfile.open(local_path(root, spec["path"]), "r:gz") as archive:
            archive.extractall(temporary, filter="data")
        entries = list(Path(temporary).iterdir())
        if len(entries) != 1 or not entries[0].is_dir():
            raise ResourceError("Expected one top-level source archive directory")
        write_json(entries[0] / marker.name, stamp)
        os.replace(entries[0], destination)
    return destination


def extract_members(root: Path, spec: dict, actual: dict) -> dict:
    """Extract only named regular files and verify derived bytes on every reuse."""
    import shutil

    receipt_path = local_path(root, spec["path"] + ".members.json")
    expected = json.loads(receipt_path.read_text()) if receipt_path.exists() else None
    if expected and expected["archive_sha256"] != actual["sha256"]:
        raise ResourceError("Derived artifacts have a different source archive")
    results = {}
    with tarfile.open(local_path(root, spec["path"]), "r:gz") as archive:
        for member, relative in spec["members"].items():
            path = local_path(root, relative)
            if not path.exists():
                info = archive.getmember(member)
                if not info.isfile():
                    raise ResourceError(f"Not a regular archive member: {member}")
                path.parent.mkdir(parents=True, exist_ok=True)
                with (
                    archive.extractfile(info) as src,
                    tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as dst,
                ):
                    temporary = Path(dst.name)
                    try:
                        shutil.copyfileobj(src, dst)
                    except BaseException:
                        temporary.unlink(missing_ok=True)
                        raise
                os.replace(temporary, path)
            results[relative] = fingerprints(path)
            if expected and results[relative] != expected["artifacts"].get(relative):
                raise ResourceError(f"Derived artifact checksum mismatch: {path}")
            if not expected:
                # Existing files cannot establish their own provenance: compare archive bytes.
                with archive.extractfile(member) as stream:
                    digest = hashlib.file_digest(stream, "sha256").hexdigest()
                if digest != results[relative]["sha256"]:
                    raise ResourceError(
                        f"Derived artifact differs from archive: {path}"
                    )
    receipt = {"archive_sha256": actual["sha256"], "artifacts": results}
    write_json(receipt_path, receipt)
    return receipt


def load_resources(root: Path, paths=None) -> dict:
    """Combine independently validated manifests without duplicate resource IDs."""
    from common.settings import MANIFEST_PATHS

    resources = {}
    for relative in MANIFEST_PATHS if paths is None else paths:
        manifest = load_manifest(local_path(root, relative))
        duplicates = resources.keys() & manifest["resources"].keys()
        if duplicates:
            raise ResourceError(f"Duplicate resource IDs: {sorted(duplicates)}")
        resources.update(manifest["resources"])
    return {"schema_version": 1, "resources": resources}
