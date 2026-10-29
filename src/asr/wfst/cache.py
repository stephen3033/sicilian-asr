import json
from pathlib import Path

from asr.wfst.storage import load_tensor, tensor_path
from common.errors import ResourceError
from common.utils import fingerprints, local_path


def load_graph(directory: Path):
    """Load a previously verified graph once as a reusable single-graph vector."""
    import k2

    graph = k2.Fsa.from_dict(load_tensor(tensor_path(directory)))
    words = {
        int(identifier): word
        for word, identifier in (
            line.rsplit(" ", 1)
            for line in (directory / "words.txt").read_text().splitlines()
        )
    }
    return k2.create_fsa_vec([graph]), words


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
