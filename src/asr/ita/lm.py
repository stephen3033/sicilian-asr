from pathlib import Path

from asr.wfst.arpa import read_unigrams
from common.errors import ResourceError
from common.text import normalize

VOCAB_RULES = "shared-exact-normalized-mls-unigram-desc-lexical-tie-v1"


def select_vocabulary(
    arpa: Path, lexicon: dict, limit: int | None
) -> tuple[list[str], dict]:
    if limit is not None and (isinstance(limit, bool) or limit <= 0):
        raise ResourceError(
            "Vocabulary size must be positive or null for full shared vocabulary"
        )
    unigrams = read_unigrams(arpa)
    shared = sorted(
        (w for w in lexicon if w in unigrams and normalize(w) == w),
        key=lambda w: (-unigrams[w], w),
    )
    selected = shared if limit is None else shared[:limit]
    if not selected:
        raise ResourceError("LM and supported lexicon have no shared words")
    return sorted(selected), {
        "rules": VOCAB_RULES,
        "lm_unigrams": len(unigrams),
        "shared_words": len(shared),
        "selected_words": len(selected),
        "limit": limit,
        "unknown_symbol": "<unk>",
        "noncanonical_lm_unigrams": sum(
            normalize(w) != w for w in unigrams if w not in {"<s>", "</s>", "<unk>"}
        ),
    }
