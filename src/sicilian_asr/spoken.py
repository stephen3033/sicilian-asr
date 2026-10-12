"""Conservative Italian spoken-form scoring; independent of decoder vocabulary."""

import re

from sicilian_asr.text import normalize

SPOKEN_RULES = "ita-standalone-cardinals-0-999999-v1"
UNITS = ("zero", "uno", "due", "tre", "quattro", "cinque", "sei", "sette", "otto", "nove",
         "dieci", "undici", "dodici", "tredici", "quattordici", "quindici", "sedici",
         "diciassette", "diciotto", "diciannove")
TENS = ("", "", "venti", "trenta", "quaranta", "cinquanta", "sessanta", "settanta", "ottanta", "novanta")


def integer_italian(number: int) -> str:
    if not 0 <= number <= 999999:
        raise ValueError("Italian cardinal must be between 0 and 999999")

    def plain(n):
        if n < 20:
            return UNITS[n]
        if n < 100:
            tens, units = divmod(n, 10)
            stem = TENS[tens][:-1] if units in (1, 8) else TENS[tens]
            return stem + (UNITS[units] if units else "")
        if n < 1000:
            hundreds, rest = divmod(n, 100)
            stem = (UNITS[hundreds] if hundreds > 1 else "") + "cento"
            if rest == 8 or 80 <= rest < 90:
                stem = stem[:-1]
            return stem + (plain(rest) if rest else "")
        thousands, rest = divmod(n, 1000)
        return ("mille" if thousands == 1 else plain(thousands) + "mila") + (plain(rest) if rest else "")

    result = plain(number)
    return result[:-1] + "é" if number > 3 and result.endswith("tre") else result


def spoken_form(text: str) -> dict:
    transformations, unsupported = [], []

    def verbalize(match):
        value = match[0]
        before, after = text[:match.start()], text[match.end():]
        # Do not reinterpret decimals, dates, signs, ranges, identifiers or
        # leading-zero sequences as cardinals. Sentence punctuation is allowed.
        mixed = (bool(before) and (before[-1].isalnum() or before[-1] in "_/'’:+-−")) or (
                 bool(after) and (after[0].isalnum() or after[0] in "_/'’:%+-−")) or (
                 bool(re.search(r"\d[.,]$", before))) or bool(re.match(r"[.,]\d", after))
        if mixed or (len(value) > 1 and value[0] == "0") or len(value) > 6:
            unsupported.append({"source": value, "start": match.start(), "end": match.end()})
            return value
        replacement = integer_italian(int(value))
        transformations.append({"source": value, "replacement": replacement,
                                "start": match.start(), "end": match.end()})
        return replacement

    converted = re.sub(r"[0-9]+", verbalize, text)
    return {"normalization": SPOKEN_RULES, "text": normalize(converted),
            "transformations": transformations, "unsupported_numbers": unsupported}
