"""Inspect and restrict published ARPA weights; never estimate a language model."""

import math
import re
import shutil
import tempfile
from collections import Counter
from pathlib import Path

from common.errors import ResourceError


def entries(path: Path):
    order = 0
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            stripped = line.strip()
            if match := re.fullmatch(r"\\(\d+)-grams:", stripped):
                order = int(match[1])
            elif stripped.startswith("\\"):
                order = 0
            elif order and stripped:
                row = stripped.split()
                if len(row) not in (order + 1, order + 2):
                    raise ResourceError(f"Malformed {order}-gram ARPA row")
                if not math.isfinite(float(row[0])) or (
                    len(row) == order + 2 and not math.isfinite(float(row[-1]))
                ):
                    raise ResourceError("Nonfinite ARPA weight")
                yield order, tuple(row[1 : order + 1]), float(row[0]), line


def read_unigrams(arpa: Path) -> dict[str, float]:
    unigrams = {}
    for order, words, score, _ in entries(arpa):
        if order != 1:
            break
        if words[0] in unigrams:
            raise ResourceError("Duplicate LM unigram")
        unigrams[words[0]] = score
    if not {"<s>", "</s>", "<unk>"} <= unigrams.keys():
        raise ResourceError("LM must supply <s>, </s>, and <unk>")
    return unigrams


def restrict_arpa(
    source: Path, destination: Path, vocabulary: list[str], max_order: int = 3
) -> dict:
    """Keep all n-grams whose words are allowed, including unmodified backoff weights."""
    allowed = set(vocabulary) | {"<s>", "</s>", "<unk>"}
    if max_order not in (2, 3):
        raise ResourceError("Supported ARPA orders are two and three")
    orders = tuple(range(1, max_order + 1))
    counts, original = Counter(), Counter()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination.parent) as temporary:
        paths = {n: Path(temporary) / str(n) for n in orders}
        streams = {n: p.open("w", encoding="utf-8") for n, p in paths.items()}
        try:
            for n, words, _, line in entries(source):
                if n > max_order:
                    raise ResourceError(f"Expected order-{max_order} ARPA")
                original[n] += 1
                if all(w in allowed for w in words):
                    streams[n].write(line)
                    counts[n] += 1
        finally:
            for stream in streams.values():
                stream.close()
        if set(original) != set(orders):
            raise ResourceError(f"Expected all {max_order} ARPA orders")
        with destination.open("w", encoding="utf-8") as output:
            output.write("\\data\\\n")
            for n in orders:
                output.write(f"ngram {n}={counts[n]}\n")
            for n in orders:
                output.write(f"\n\\{n}-grams:\n")
                with paths[n].open() as stream:
                    shutil.copyfileobj(stream, output)
            output.write("\n\\end\\\n")
    return {
        "original_ngrams": dict(original),
        "retained_ngrams": dict(counts),
        "weights": "original log10 probabilities and backoff weights; no renormalization or retraining",
    }
