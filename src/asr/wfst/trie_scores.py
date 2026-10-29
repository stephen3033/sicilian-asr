"""Bounded exact word-trie queries without materializing lattice products."""

import ctypes
import hashlib
import subprocess
import tempfile
from functools import lru_cache
from pathlib import Path

import k2
import numpy as np

from common.errors import ResourceError


@lru_cache(1)
def library():
    source = (
        Path(__file__).resolve().parents[3] / "resources/native/lattice_trie_scores.cpp"
    )
    if not source.exists():
        from importlib.resources import files

        source = Path(
            str(files("sicilian_asr_native").joinpath("lattice_trie_scores.cpp"))
        )
    key = hashlib.sha256(source.read_bytes()).hexdigest()[:16]
    directory = Path(tempfile.gettempdir()) / "sicilian-asr-trie" / key
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / "trie.so"
    if not target.exists():
        subprocess.run(
            [
                "c++",
                "-O3",
                "-std=c++17",
                "-shared",
                "-fPIC",
                str(source),
                "-o",
                str(target),
            ],
            check=True,
            capture_output=True,
        )
    native = ctypes.CDLL(str(target)).lattice_trie_scores
    i32 = np.ctypeslib.ndpointer(dtype=np.int32, flags="C_CONTIGUOUS")
    f32 = np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS")
    f64 = np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")
    native.argtypes = [
        ctypes.c_int64,
        ctypes.c_int64,
        i32,
        f32,
        i32,
        i32,
        ctypes.c_int64,
        ctypes.c_int64,
        i32,
        ctypes.c_int64,
        i32,
        f64,
    ]
    native.restype = ctypes.c_int
    return native


def query(lattice, sequences, word_ids):
    fsa = lattice[0]
    if not fsa.properties & k2.fsa_properties.TOPSORTED_AND_ACYCLIC:
        fsa = k2.top_sort(fsa)
    if fsa.num_arcs == 0:
        return [{"present": False, "score": None} for _ in sequences]
    arcs = np.ascontiguousarray(fsa.arcs.values().numpy(), dtype=np.int32)
    weights = np.ascontiguousarray(fsa.scores.detach().numpy(), dtype=np.float32)
    if isinstance(fsa.aux_labels, k2.RaggedTensor):
        splits = np.ascontiguousarray(
            fsa.aux_labels.shape.row_splits(1).numpy(), dtype=np.int32
        )
        words = np.ascontiguousarray(fsa.aux_labels.values.numpy(), dtype=np.int32)
    else:
        splits = np.arange(len(arcs) + 1, dtype=np.int32)
        words = np.ascontiguousarray(fsa.aux_labels.numpy(), dtype=np.int32)
    prefixes, trie, terminals = {(): 0}, [], []
    for sequence in sequences:
        prefix = ()
        for word in sequence:
            following = (*prefix, word_ids[word])
            if following not in prefixes:
                prefixes[following] = len(prefixes)
                trie.append((prefixes[prefix], prefixes[following], word_ids[word]))
            prefix = following
        terminals.append(prefixes[prefix])
    result = np.full(len(sequences), -np.inf, dtype=np.float64)
    status = library()(
        fsa.shape[0],
        len(arcs),
        arcs,
        weights,
        splits,
        words,
        len(prefixes),
        len(trie),
        np.asarray(trie, dtype=np.int32),
        len(sequences),
        np.asarray(terminals, dtype=np.int32),
        result,
    )
    if status:
        raise ResourceError(f"Exact lattice trie query failed validation: {status}")
    return [
        {"present": bool(np.isfinite(s)), "score": float(s) if np.isfinite(s) else None}
        for s in result
    ]
