"""Inspect and restrict published ARPA weights; never estimate a language model."""

from collections import Counter
from pathlib import Path
import math
import re
import shutil
import tempfile

from sicilian_asr.resources import ResourceError
from sicilian_asr.text import normalize

VOCAB_RULES = "shared-exact-normalized-mls-unigram-desc-lexical-tie-v1"


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
                if len(row) not in (order+1, order+2):
                    raise ResourceError(f"Malformed {order}-gram ARPA row")
                if not math.isfinite(float(row[0])) or (len(row) == order+2 and not math.isfinite(float(row[-1]))):
                    raise ResourceError("Nonfinite ARPA weight")
                yield order, tuple(row[1:order+1]), float(row[0]), line


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


def select_vocabulary(arpa: Path, lexicon: dict, limit: int | None) -> tuple[list[str], dict]:
    if limit is not None and (isinstance(limit, bool) or limit <= 0):
        raise ResourceError("Vocabulary size must be positive or null for full shared vocabulary")
    unigrams = read_unigrams(arpa)
    shared = sorted((w for w in lexicon if w in unigrams and normalize(w) == w), key=lambda w: (-unigrams[w], w))
    selected = shared if limit is None else shared[:limit]
    if not selected:
        raise ResourceError("LM and supported lexicon have no shared words")
    return sorted(selected), {"rules": VOCAB_RULES, "lm_unigrams": len(unigrams), "shared_words": len(shared),
                              "selected_words": len(selected), "limit": limit, "unknown_symbol": "<unk>",
                              "noncanonical_lm_unigrams": sum(normalize(w) != w for w in unigrams if w not in {"<s>", "</s>", "<unk>"})}


def restrict_arpa(source: Path, destination: Path, vocabulary: list[str]) -> dict:
    """Keep all n-grams whose words are allowed, including unmodified backoff weights."""
    allowed = set(vocabulary) | {"<s>", "</s>", "<unk>"}
    counts, original = Counter(), Counter()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination.parent) as temporary:
        paths = {n: Path(temporary)/str(n) for n in (1, 2, 3)}
        streams = {n: p.open("w", encoding="utf-8") for n, p in paths.items()}
        try:
            for n, words, _, line in entries(source):
                if n > 3:
                    raise ResourceError("Expected pretrained trigram ARPA")
                original[n] += 1
                if all(w in allowed for w in words):
                    streams[n].write(line)
                    counts[n] += 1
        finally:
            for stream in streams.values():
                stream.close()
        if set(original) != {1, 2, 3}:
            raise ResourceError("Expected all three ARPA orders")
        with destination.open("w", encoding="utf-8") as output:
            output.write("\\data\\\n")
            for n in (1, 2, 3):
                output.write(f"ngram {n}={counts[n]}\n")
            for n in (1, 2, 3):
                output.write(f"\n\\{n}-grams:\n")
                with paths[n].open() as stream:
                    shutil.copyfileobj(stream, output)
            output.write("\n\\end\\\n")
    return {"original_ngrams": dict(original), "retained_ngrams": dict(counts), "weights": "original log10 probabilities and backoff weights; no renormalization or retraining"}
