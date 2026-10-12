import json
from pathlib import Path

from common.errors import ResourceError
from common.utils import fingerprints, local_path


def verify_cache(directory: Path, key: str) -> dict:
    path = directory / "prepared.json"
    if not path.is_file():
        raise ResourceError(f"Missing prepared graph: {directory}; run prepare")
    prepared = json.loads(path.read_text())
    if prepared.get("status") != "complete" or prepared.get("key") != key:
        raise ResourceError("Stale or incomplete graph cache; run prepare")
    for name, expected in prepared["artifacts"].items():
        artifact = local_path(directory, name)
        if not artifact.is_file() or fingerprints(artifact)["sha256"] != expected:
            raise ResourceError(f"Prepared artifact checksum mismatch: {artifact}")
    return prepared
