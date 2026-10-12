"""Auditable PhonItalia → ZIPA conversion, preserving pronunciation alternatives."""

import csv
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

from common.errors import ResourceError
from common.text import normalize

LEXICON_RULES = "phonitalia-concatenate-g-tokenizer-no-stress-exclude-tz-bv-v1"
ADAPTATION_RULES = "ita-length-doubled-closure-nasal-l-c-elisions-v1"


def adapt(
    lexicon: dict,
    tokens: list[str],
    lm_words: set[str],
    *,
    pronunciation_variants: bool,
    contractions: bool,
) -> tuple[dict, dict]:
    """Add model-compatible alternatives without discarding source distinctions.

    Rules apply to complete pronunciations, yielding at most four variants per
    source pronunciation rather than a combinatorial choice at every phone.
    Contractions are derived only from independently attested LM spellings.
    """
    table = {t: i for i, t in enumerate(tokens)}
    result = {w: {tuple(p) for p in ps} for w, ps in lexicon.items()}
    added, unsupported = [], []

    def add(word, pronunciation, source_word, source, rules):
        ids = tuple(table[t] for t in pronunciation)
        if ids not in result.setdefault(word, set()):
            result[word].add(ids)
            added.append(
                {
                    "word": word,
                    "pronunciation": pronunciation,
                    "source_word": source_word,
                    "source_pronunciation": source,
                    "rules": rules,
                }
            )

    if contractions:
        for word in sorted(lm_words):
            if "'" not in word or word in result:
                continue
            prefix, suffix = word.split("'", 1)
            eligible = lexicon.get(suffix, []) if prefix in {"l", "c"} else []
            for ids in eligible:
                source = "".join(tokens[i] for i in ids)
                if source and source[0] in "aeɛioɔu":
                    add(
                        word,
                        ("l" if prefix == "l" else "tʃ") + source,
                        suffix,
                        source,
                        ["lm-attested-elision"],
                    )
            if word not in result:
                unsupported.append(word)

    if pronunciation_variants:
        for word, pronunciations in list(result.items()):
            for ids in sorted(pronunciations):
                original = "".join(tokens[i] for i in ids)
                # Match affricates first: length doubles their closure only.
                doubled = re.sub(
                    r"(tʃ|dʒ|ts|dz|[pbtdkgfvszʃʒmnŋɲlʎr])ː",
                    lambda m: m[1][0] + m[1],
                    original,
                )
                nasal = lambda s: re.sub(r"ŋ(?=[kg])", "n", s)
                for pronunciation, rules in [
                    (doubled, ["length-to-double"]),
                    (nasal(original), ["prevelar-nasal"]),
                    (nasal(doubled), ["length-to-double", "prevelar-nasal"]),
                ]:
                    if pronunciation != original:
                        add(word, pronunciation, word, original, rules)
    return {w: [list(p) for p in sorted(ps)] for w, ps in sorted(result.items())}, {
        "rules": ADAPTATION_RULES,
        "pronunciation_variants": pronunciation_variants,
        "contractions": contractions,
        "source_words": len(lexicon),
        "words": len(result),
        "added_words": sorted(set(result) - set(lexicon)),
        "added_pronunciations": added,
        "unsupported_lm_contractions": unsupported,
    }


def convert(
    path: Path, tokens: list[str], tokenizer
) -> tuple[dict[str, list[list[int]]], dict]:
    pairs = set()
    spellings = defaultdict(set)
    exclusions = []
    source_phones = Counter()
    rows = 0
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        row = line.split("\t")
        if len(row) != 2 or not row[0] or not row[1].split():
            raise ResourceError(f"Malformed IPA dictionary row {number}")
        rows += 1
        original, phones = row[0], row[1].split()
        word = normalize(original)
        spellings[word].add(original)
        source_phones.update(phones)
        reason = None
        if not word or " " in word or word == "<unk>":
            reason = "not one normalized lexical word"
        elif bad := sorted(set(phones) & {"tz", "bv"}):
            reason = "ambiguous source phones: " + ", ".join(bad)
        joined = unicodedata.normalize("NFC", "".join(phones)).replace("ɡ", "g")
        ids = tokenizer.encode(joined, out_type=int)
        # Training adds one SentencePiece dummy prefix to the entire utterance.
        if ids and ids[0] == tokens.index("▁"):
            ids = ids[1:]
        if (
            not ids
            or any(i < 4 for i in ids)
            or "".join(tokens[i] for i in ids) != joined
        ):
            reason = reason or "not losslessly representable by pinned inventory"
        if reason:
            exclusions.append(
                {
                    "row": number,
                    "source_word": original,
                    "word": word,
                    "phones": phones,
                    "reason": reason,
                }
            )
        else:
            pairs.add((word, tuple(ids)))
    lexicon = defaultdict(list)
    for word, ids in sorted(pairs):
        lexicon[word].append(list(ids))
    return dict(lexicon), {
        "rules": LEXICON_RULES,
        "source_rows": rows,
        "source_words": len({s for v in spellings.values() for s in v}),
        "normalized_source_words": len(spellings),
        "retained_words": len(lexicon),
        "retained_pronunciations": len(pairs),
        "duplicate_normalized_pairs_removed": rows - len(exclusions) - len(pairs),
        "words_with_alternatives": sum(len(v) > 1 for v in lexicon.values()),
        "word_coverage": len(lexicon) / len(spellings),
        "excluded_rows": len(exclusions),
        "excluded_words": sorted({x["word"] for x in exclusions}),
        "exclusions": exclusions,
        "normalization_collisions": {
            w: sorted(s) for w, s in sorted(spellings.items()) if len(s) > 1
        },
        "source_phone_counts": dict(sorted(source_phones.items())),
        "stress": "Source IPA dictionary has no explicit stress; ZIPA inventory has no stress labels.",
    }


def lexicon_statistics(path: Path) -> dict:
    words, pairs = set(), set()
    count = 0
    with path.open(encoding="utf-8", newline="") as stream:
        for row in csv.reader(stream, delimiter="\t"):
            if len(row) != 2 or not row[0] or not row[1].split():
                raise ResourceError(f"Malformed IPA dictionary row {count + 1}")
            count += 1
            words.add(row[0])
            pairs.add(tuple(row))
    return {
        "rows": count,
        "vocabulary": len(words),
        "distinct_word_pronunciation_pairs": len(pairs),
    }
