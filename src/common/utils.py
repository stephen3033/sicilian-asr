from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path

from common.errors import ResourceError


def local_path(root: Path, relative: str) -> Path:
    """Resolve repository-relative paths, rejecting traversal and symlink escapes."""
    root = root.resolve()
    path = (root / relative).resolve()
    if Path(relative).is_absolute() or not path.is_relative_to(root):
        raise ResourceError(f"Path escapes repository: {relative}")
    return path


def fingerprints(path: Path) -> dict:
    size = path.stat().st_size
    sha256 = hashlib.sha256()
    md5 = hashlib.md5()
    git_blob = hashlib.sha1(f"blob {size}\0".encode())
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            sha256.update(chunk)
            md5.update(chunk)
            git_blob.update(chunk)
    return {
        "size_bytes": size,
        "sha256": sha256.hexdigest(),
        "md5": md5.hexdigest(),
        "git_blob_sha1": git_blob.hexdigest(),
    }


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", dir=path.parent, suffix=".part", delete=False
    ) as stream:
        temporary = Path(stream.name)
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def command(args: list[str]) -> str:
    try:
        result = subprocess.run(args, capture_output=True, text=True, check=True)
    except (OSError, subprocess.CalledProcessError) as error:
        raise ResourceError(
            f"{args[0]} failed: {getattr(error, 'stderr', str(error))}"
        ) from error
    return result.stdout
