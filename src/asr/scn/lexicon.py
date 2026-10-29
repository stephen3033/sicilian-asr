"""Sicilian IPA evidence and a bounded, explicitly provisional DOS24 G2P."""

import re
import unicodedata
from collections import defaultdict
from pathlib import Path

from asr.scn.orthography import normalize

RULES = "scn-dos24-broad-seed-rules-v1"
VOWELS = "aeiouàèìòùâêîôû"
BASE = str.maketrans("àèìòùâêîôû", "aeiouaeiou")
SILENT_H = {"hai", "haju", "havi", "hannu", "hê", "hâ", "hî", "hô", "hû"}


def project(ipa, tokens, tokenizer):
    """Only named projection changes; never silently drop unsupported phones."""
    value = unicodedata.normalize("NFC", ipa.replace("ɡ", "g"))
    value = value.replace("ˈ", "").replace("ˌ", "")
    for affricate in ("tʃ", "dʒ", "ts", "dz", "ʈɽ", "ɖɽ"):
        for tie in ("͡", "͜"):
            value = value.replace(affricate[0] + tie + affricate[1], affricate)
    ids = tokenizer.encode(value, out_type=int)
    if ids and ids[0] == 3:
        ids = ids[1:]
    if not ids or any(i < 4 for i in ids) or "".join(tokens[i] for i in ids) != value:
        return None
    return ids


def g2p(word: str, *, repair_soft_geminates: bool = False) -> list[str]:
    """Broad default plus bounded z/length alternatives; not a gold transcription."""
    if not re.fullmatch(r"[a-zàèìòùâêîôûḍ']+", word) or any(c in word for c in "kwxy"):
        return []
    original = word.replace("'", "")
    if not original:
        return []
    positions = [i for i, c in enumerate(original) if c in VOWELS]
    explicit = [i for i, c in enumerate(original) if c in "àèìòù"]
    stress = (
        explicit[-1]
        if explicit
        else (
            positions[-2] if len(positions) > 1 else (positions[0] if positions else -1)
        )
    )
    base = original.translate(BASE)
    output, i = [], 0
    while i < len(base):
        rest = base[i:]
        # Diagnostic opt-in: look beyond the doubled consonant before choosing
        # velar versus affricate. The frozen production recipe remains unchanged.
        if (
            repair_soft_geminates
            and rest[:2] in {"cc", "gg"}
            and rest[2:3] in {"e", "i"}
        ):
            output.append("tʃː" if rest[0] == "c" else "dʒː")
            i += 2
            if base[i : i + 1] == "i" and base[i + 1 : i + 2] in {"a", "o", "u"}:
                i += 1
            continue
        matched = False
        for spelling, phone in (
            ("ḍḍ", "ɖː"),
            ("str", "sʈɽ"),
            ("tr", "ʈɽ"),
            ("dr", "ɖɽ"),
            ("gghi", "ɟː"),
            ("cchi", "cː"),
            ("ghi", "ɟ"),
            ("chi", "c"),
            ("sci", "ʃː"),
            ("sce", "ʃː"),
            ("gn", "ɲː"),
            ("ch", "k"),
            ("gh", "g"),
            ("qu", "kw"),
        ):
            if not rest.startswith(spelling):
                continue
            if spelling in {"gghi", "cchi", "ghi", "chi"} and (
                len(rest) <= len(spelling) or rest[len(spelling)] not in "aeou"
            ):
                continue
            if spelling == "sce":
                output.append(phone)
                i += 2
            elif spelling == "sci" and (len(rest) == 3 or rest[3] not in "aeou"):
                output.append(phone)
                i += 2
            else:
                output.append(phone)
                i += len(spelling)
            matched = True
            break
        if matched:
            continue
        c = base[i]
        following = base[i + 1 : i + 2]
        if c in "aeiou":
            phone = {
                "e": "ɛ" if i == stress else "i",
                "o": "ɔ" if i == stress else "u",
            }.get(c, c)
            # Unstressed i before a vowel is a default glide; explicit stress overrides.
            if c == "i" and following in "aeou" and following and i != stress:
                phone = "j"
        elif c in "cg":
            phone = (
                ("tʃ" if c == "c" else "dʒ")
                if following in {"e", "i"}
                else ("k" if c == "c" else "g")
            )
            if following == "i" and base[i + 2 : i + 3] in {"a", "o", "u"}:
                i += 1
        elif c == "z":
            phone = "ts"
        elif c == "h":
            phone = "" if original in SILENT_H else "x"
        elif c == "j":
            phone = "j"
        elif c in "bdf lmnprstv".replace(" ", ""):
            phone = c
        else:
            return []
        if c not in "aeiouh" and following == c:
            phone += "ː"
            i += 1
        output.append(phone)
        i += 1
    broad = "".join(output)
    broad = re.sub(r"n(?=[kg])", "ŋ", broad)
    variants = [broad]
    if "ts" in broad:
        variants.append(broad.replace("ts", "dz"))
    for value in list(variants):
        doubled = re.sub(
            r"(tʃ|dʒ|ts|dz|[pbtdkgcɟɖʈfvszʃʒmnŋɲlʎrɽ])ː",
            lambda m: m[1][0] + m[1],
            value,
        )
        if doubled != value:
            variants.append(doubled)
    return list(dict.fromkeys(variants))[:4]


def seed_entries(paths: list[Path]) -> dict:
    result = defaultdict(list)
    for path in paths:
        for n, line in enumerate(path.read_text().splitlines(), 1):
            row = line.split("\t")
            if len(row) != 2:
                continue
            word = normalize(row[0])
            if word and " " not in word:
                result[word].append(
                    {
                        "ipa": "".join(row[1].split()),
                        "source": path.name,
                        "row": n,
                        "kind": "observed_unvalidated",
                    }
                )
    return dict(result)


def build(words, seeds, tokens, tokenizer, exceptions=None):
    lexicon, records, excluded = {}, [], []
    for word in sorted(set(words)):
        candidates = list((exceptions or {}).get(word, [])) + seeds.get(word, [])
        candidates += [
            {"ipa": ipa, "kind": "rule_generated", "source": RULES} for ipa in g2p(word)
        ]
        accepted, seen = [], set()
        for candidate in candidates:
            ids = project(candidate["ipa"], tokens, tokenizer)
            if ids and tuple(ids) not in seen:
                accepted.append(ids)
                seen.add(tuple(ids))
                records.append(
                    {
                        "word": word,
                        **candidate,
                        "token_ids": ids,
                        "projection": "g-equivalence-stress-removal-known-affricate-ties",
                    }
                )
                if len(accepted) == 4:
                    break
        if accepted:
            lexicon[word] = accepted
        else:
            excluded.append(word)
    return lexicon, {
        "rules": RULES,
        "pronunciations": records,
        "excluded": excluded,
        "warning": "Generated and crowdsourced pronunciations are provisional, not independently validated gold.",
    }
