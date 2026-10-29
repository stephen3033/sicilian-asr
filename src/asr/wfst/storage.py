"""Lossless optional compression for experimental WFST tensor artifacts."""

import gzip
from pathlib import Path

from common.errors import ResourceError


def tensor_path(directory: Path, name="TLG") -> Path:
    plain = directory / f"{name}.pt"
    compressed = directory / f"{name}.pt.gz"
    if plain.exists() and compressed.exists():
        raise ResourceError(f"Ambiguous graph artifacts: {plain}, {compressed}")
    if not plain.exists() and not compressed.exists():
        raise FileNotFoundError(f"Missing graph tensor: {name}")
    return plain if plain.exists() else compressed


def save_tensor(value, directory: Path, name, compressed=False) -> Path:
    import torch

    path = directory / (f"{name}.pt.gz" if compressed else f"{name}.pt")
    if compressed:
        with (
            path.open("wb") as raw,
            gzip.GzipFile(
                filename="", mode="wb", fileobj=raw, compresslevel=1, mtime=0
            ) as stream,
        ):
            torch.save(value, stream)
    else:
        torch.save(value, path)
    return path


def load_tensor(path: Path):
    import torch

    if path.suffix == ".gz":
        with gzip.open(path, "rb") as stream:
            return torch.load(stream, map_location="cpu", weights_only=False)
    return torch.load(path, map_location="cpu", weights_only=False)
