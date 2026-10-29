"""DOS24 text handling. Spelling changes require explicit, sourced decisions."""

import re
import unicodedata

from common.errors import ResourceError
from common.text import APOSTROPHES

RULES = "scn-dos24-nfc-single-token-quotes-preserve-edge-elisions-v2"
WORD = re.compile(r"<unk>|'?[^\W\d_]+(?:'[^\W\d_]+)*'?|\d+", re.UNICODE)


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFC", text).lower().translate(APOSTROPHES)
    # Never pair elisions across words, e.g. 'n Sicilia su'.
    text = re.sub(r"(?<!\w)'([^\W\d_]+)'(?!\w)", r"\1", text)
    # Apostrophes at lexical edges are significant in Sicilian ('n, 'na, po').
    return " ".join(WORD.findall(text))


def corrected(text: str, replacements: dict) -> tuple[str, list[dict]]:
    words = normalize(text).split()
    changes = []
    result = []
    for word in words:
        item = replacements.get(word)
        if item is not None:
            if not item.get("evidence") or not item.get("reason"):
                raise ResourceError(f"Unsourced Sicilian spelling correction: {word}")
            replacement = item["replacement"]
            changes.append({"original": word, **item})
            result.extend(normalize(replacement).split())
        else:
            result.append(word)
    return " ".join(result), changes
