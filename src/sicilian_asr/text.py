"""Shared, versioned orthographic normalization; no number verbalization."""

import re
import unicodedata

TEXT_RULES = "nfc-lower-apostrophes-punctuation-v1"
APOSTROPHES = str.maketrans({c: "'" for c in "’‘ʼ＇`"})


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFC", text).lower().translate(APOSTROPHES)
    # Preserve the LM unknown marker as a visible, scored token.
    parts = re.split(r"(<unk>)", text)
    result = []
    for part in parts:
        if part == "<unk>":
            result.append(part)
            continue
        chars = []
        for i, char in enumerate(part):
            internal = char == "'" and i > 0 and i + 1 < len(part) and part[i-1].isalnum() and part[i+1].isalnum()
            chars.append(char if char.isalnum() or unicodedata.category(char).startswith("M") or internal else " ")
        result.append("".join(chars))
    return " ".join("".join(result).split())
