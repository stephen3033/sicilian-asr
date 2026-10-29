"""Evidence-bearing DOS24 conversion. This is separate from scoring normalization.

Lexical resources provide analyses, not canonical spellings. Every productive
change has an explicit rule; missing stress/identity/function stays unresolved.
Offsets in edits and analyses always address the original Python string.
"""

import re
import unicodedata
from collections import defaultdict
from dataclasses import asdict, dataclass
from functools import lru_cache

from common.text import APOSTROPHES

VERSION = "dos24-v2.2-converter-1"
REFERENCE = (
    "https://cademiasiciliana.org/wp-content/uploads/Cademia-Siciliana-DOS24-v.2.2.pdf"
)
TOKEN = re.compile(
    r"[‘’'ʼ]?(?:[^\W\d_]|[\u0300-\u036f])+(?:[‘’'ʼ](?:[^\W\d_]|[\u0300-\u036f])+)*[‘’'ʼ]?|\d+(?:[.,]\d+)*",
    re.UNICODE,
)
FOLD = str.maketrans("àèìòùáéíóú", "aeiouaeiou")
GRAVE = str.maketrans("áéíóú", "àèìòù")


@dataclass(frozen=True)
class Rule:
    id: str
    family: str
    pages: tuple
    condition: str
    output: str
    exceptions: tuple
    positive: tuple
    negative: tuple


RULES = [
    Rule(
        "T01",
        "typography",
        (7, 8),
        "Unicode-equivalent characters and lexical apostrophes",
        "NFC; straight apostrophe",
        (),
        ("l’acqua → l'acqua",),
        ("do not erase 'n",),
    ),
    Rule(
        "T02",
        "explicit_stress",
        (7, 9),
        "Native analyzed word with an explicitly marked acute accent",
        "Grave accent at the same position",
        ("names", "quotations", "foreign words"),
        ("pirchí → pirchì",),
        ("José stays José",),
    ),
    Rule(
        "L01",
        "lexical",
        (13, 14, 15, 17),
        "A published lexical correspondence or a uniquely supported retroflex lexical family",
        "The cited consonant correspondence, retaining inflection",
        ("dental dd", "ambiguous lexical identity"),
        ("cavaddu → cavaḍḍu",),
        ("friddu stays friddu",),
    ),
    Rule(
        "M01",
        "derivation",
        (16, 17),
        "Native noun with the nominalizing -zioni suffix (including inflection)",
        "-zzioni",
        ("consonant preceding z", "foreign names"),
        ("culabburazioni → culabburazzioni",),
        ("pranzu stays pranzu",),
    ),
    Rule(
        "M02",
        "inflection",
        (10, 11),
        "Analyzed native form; final e/o is unaccented and lexical alternatives agree on i/u",
        "Final e→i or o→u",
        ("stressed final vowels", "loan/name without native analysis"),
        ("canto → cantu, when analyzed as cantu",),
        ("parrò stays parrò",),
    ),
    Rule(
        "S01",
        "stress_hiatus",
        (7, 8, 9, 11),
        "Unique stress pattern among analyses of the same orthographic form",
        "Explicit grave accent where the lexical evidence supplies it",
        ("aju/àju ai/ài avi/àvi annu/ànnu", "competing stress patterns"),
        ("magaria → magarìa",),
        ("do not infer talìu from an arbitrary -iu ending",),
    ),
    Rule(
        "B01",
        "elision",
        (8, 12),
        "Analyzed permanently reduced in- lexical family",
        "Remove initial apostrophe",
        ("'n", "occasional elision"),
        ("'nzirtari → nzirtari",),
        ("'n vacanza stays 'n vacanza",),
    ),
    Rule(
        "B02",
        "contraction",
        (8, 16),
        "Contraction or elision already explicitly represented",
        "Preserve circumflex and lexical boundaries",
        ("unknown contraction requires source expansion",),
        ("âm'a → âm'a",),
        ("do not turn a lu into ô by frequency",),
    ),
    Rule(
        "G01",
        "grammatical",
        (7, 17, 18),
        "Explicit grammatical analysis disambiguates copula/conjunction or locative/personal/partitive clitic",
        "e/è, ci/cci, ni/nni according to function",
        ("no acoustic disambiguation", "no frequency-based rewriting"),
        ("ci [locative] → cci",),
        ("ci [dative] stays ci",),
    ),
    Rule(
        "P01",
        "preservation",
        (7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18),
        "Native lexical or productive morphological analysis and no remaining contradictory spelling/stress/function evidence",
        "Preserve supported form and legitimate variants",
        ("absence from captions is not an exclusion",),
        ("gatta and jatta are retained",),
        ("resource canonical labels are not spelling replacement rules",),
    ),
]

# These are lexical facts from the specified pages, never imported contextual
# caption replacement dictionaries. Inflection is licensed separately below.
PUBLISHED = {
    "cavaddu": "cavaḍḍu",
    "cavaddru": "cavaḍḍu",
    "cavaddhu": "cavaḍḍu",
    "chiu": "cchiù",
    "chiù": "cchiù",
    "cchiu": "cchiù",
    "calmu": "carmu",
    "cammu": "carmu",
    "tottu": "tortu",
    "vìriri": "vìdiri",
    "bbonu": "bonu",
    "çira": "cira",
    "çertu": "certu",
    "ciavi": "chiavi",
    "ciovu": "chiovu",
    "candari": "cantari",
    "mangiari": "manciari",
    "cangiari": "canciari",
}
PUBLISHED_FORMS = set(PUBLISHED.values()) | {
    "ḍḍa",
    "ḍḍà",
    "ḍḍu",
    "ḍḍi",
    "ḍḍocu",
    "friddu",
    "addisijari",
    "aju",
    "àju",
    "ai",
    "ài",
    "avi",
    "àvi",
    "annu",
    "ànnu",
    "avemu",
    "avimu",
    "aviti",
    "magarìa",
    "talìu",
    "taliari",
    "passijari",
    "pajisi",
    "nzirtari",
    "nnuccenti",
    "mmidia",
    "gatta",
    "jatta",
    "pagari",
    "pajari",
    "pavari",
    "quannu",
    "quandu",
    "alienu",
    "novu",
    "veru",
    "parrò",
    "talè",
    "iu",
    "ju",
    "jo",
    "cani",
    "càntaru",
    "àlica",
    "pranzu",
    "âm'a",
    "âmu",
    "hê",
    "ê",
    "'n",
    "in",
    "cullabburazzioni",
    "trippedi",
    "agghiri",
    "vota",
    "voti",
    "friddu",
    "ballu",
    "allura",
}
OPTIONAL_AVIRI = {"aju", "ai", "avi", "annu"}


def key(word):
    return unicodedata.normalize("NFC", word).lower().translate(APOSTROPHES)


def family_key(word):
    return key(word).translate(FOLD).replace("ḍḍ", "dd").replace("zzioni", "zioni")


@lru_cache(maxsize=32768)
def stress_spellings(form):
    """Interpret lexical stress under DOS24 instead of copying lexical accents.

    Adjacent unstressed i/u can be a glide or a vowel. Enumerate both analyses
    and require agreement; explicit stressed i/u entails hiatus.
    """
    marked = [i for i, c in enumerate(form) if c in "àèìòù"]
    if len(marked) != 1:
        return {form}
    stress = marked[0]
    plain = form.translate(FOLD)
    vowels = [i for i, c in enumerate(plain) if c in "aeiou"]
    if len(vowels) <= 1:
        return {plain}
    partitions = [[(vowels[0],)]]
    for position in vowels[1:]:
        next_partitions = []
        for groups in partitions:
            previous = groups[-1][-1]
            next_partitions.append(groups + [(position,)])
            stressed_weak = any(i == stress and plain[i] in "iu" for i in (previous, position))
            if position == previous + 1 and not stressed_weak and any(plain[i] in "iu" and i != stress for i in (previous, position)):
                next_partitions.append(groups[:-1] + [groups[-1] + (position,)])
        partitions = next_partitions
    candidates = set()
    for groups in partitions:
        stressed_group = next(i for i, group in enumerate(groups) if stress in group)
        final_hiatus = len(groups) >= 2 and groups[-1][-1] == len(form)-1 and groups[-2][-1]+1 == groups[-1][0]
        needs_mark = stressed_group != len(groups)-2 or final_hiatus
        candidates.add(form if needs_mark else plain)
    return candidates


class Converter:
    def __init__(self, analyses=None):
        # Keep the complete lexical provenance in index.json. Refer to its rows
        # instead of repeating thousands of UD occurrences at every token.
        self.analyses = {
            form: [{**{k: v for k, v in row.items() if k in {"lemma", "pos", "features", "source", "evidence", "source_sha256"}},
                    "index_reference": [form, i]} for i, row in enumerate(rows)]
            for form, rows in (analyses or {}).items()
        }
        generated = defaultdict(list)
        for form, rows in self.analyses.items():
            for row in rows:
                if key(row["lemma"]) != form:
                    continue
                endings = ("i",) if row["pos"] == "NOUN" and form.endswith(("u", "a")) else ("a", "i") if row["pos"] == "ADJ" and form.endswith("u") else ()
                for ending in endings:
                    stem = form[:-1]
                    if ending == "i" and stem.endswith(("c", "g")):
                        stem += "h"
                    generated[stem + ending].append({**row, "derivation": "regular-nominal-inflection", "features": {"Ending": ending}})
                if row["pos"] == "VERB" and form.endswith("ari"):
                    for ending in ("amu", "ati", "ava", "avi", "avanu", "annu", "atu", "ata"):
                        generated[form[:-3] + ending].append({**row, "derivation": "regular-ari-paradigm", "features": {"Ending": ending}})
        for form, rows in generated.items():
            self.analyses.setdefault(form, rows)
        self.families = defaultdict(set)
        for word in set(self.analyses) | PUBLISHED_FORMS:
            self.families[family_key(word)].add(word)

    def analyze(self, word):
        rows = list(self.analyses.get(word, []))
        if word in PUBLISHED_FORMS or word in PUBLISHED:
            rows.append(
                {
                    "lemma": PUBLISHED.get(word, word),
                    "pos": "X",
                    "features": {},
                    "source": "DOS24-v2.2",
                    "evidence": "published-example",
                }
            )
        # Productive adjective gender/number and infinitive-based finite forms.
        # This supplies morphological evidence; it never copies a lemma spelling.
        if not rows and word.endswith(("a", "i")):
            lemma = word[:-1] + "u"
            for row in self.analyses.get(lemma, []):
                if row["pos"] == "ADJ":
                    rows.append(
                        {
                            **row,
                            "derivation": "adjective-u-a-i",
                            "features": {"Ending": word[-1]},
                        }
                    )
        if not rows and "'" in word[1:]:
            prefix, suffix = word.split("'", 1)
            if prefix in {"l", "n", "d"} and suffix:
                rows = [{**r, "elision": prefix + "'"} for r in self.analyze(suffix)]
        if not rows:
            for ending in ("amu", "ati", "annu", "ava", "avi", "avanu", "annu"):
                if word.endswith(ending):
                    for row in self.analyses.get(word[: -len(ending)] + "ari", []):
                        if row["pos"] == "VERB":
                            rows.append(
                                {
                                    **row,
                                    "derivation": "regular-ari-paradigm",
                                    "features": {"Ending": ending},
                                }
                            )
        return rows

    def word(self, original, grammatical=None):
        value = key(original)
        rules = ["T01"] if value != original.lower() else []
        analyses = self.analyze(value)
        if value in {"du", "da"}:
            contraction = {"di_lu": "dû", "di_la": "dâ"}
            if grammatical in contraction:
                return contraction[grammatical], rules + ["B02"], analyses, None
            return (
                value,
                rules,
                analyses,
                {
                    "reason": "contraction_or_lexical_form",
                    "missing": "preposition/article contraction versus independent lexical identity",
                    "candidates": [value, "dû" if value == "du" else "dâ"],
                },
            )
        if value == "dda" and grammatical in {"demonstrative", "locative_adverb"}:
            return (
                ("ḍḍa" if grammatical == "demonstrative" else "ḍḍà"),
                rules + ["L01", "S01", "G01"],
                analyses,
                None,
            )
        if "'" in value[1:]:
            prefix, suffix = value.split("'", 1)
            if prefix in {"l", "n", "d"} and suffix:
                target, applied, evidence, issue = self.word(suffix)
                return prefix + "'" + target, rules + ["B02"] + applied, evidence, issue
        if value in {"e", "è", "ci", "cci", "ni", "nni"}:
            choices = {
                "conjunction": "e",
                "copula": "è",
                "dative": "ci",
                "locative": "cci",
                "personal": "ni",
                "partitive": "nni",
            }
            valid_functions = {
                "e": {"conjunction", "copula"},
                "è": {"conjunction", "copula"},
                "ci": {"dative", "locative"},
                "cci": {"dative", "locative"},
                "ni": {"personal", "partitive"},
                "nni": {"personal", "partitive"},
            }
            if grammatical in valid_functions[value]:
                return choices[grammatical], rules + ["G01"], analyses, None
            # Explicit marked spellings distinguish the intended function; an
            # unmarked source e/ci/ni can be an older spelling of either member.
            if value in {"è", "cci", "nni"}:
                return value, rules + ["G01"], analyses, None
            return (
                value,
                rules,
                analyses,
                {
                    "reason": "grammatical_function",
                    "missing": "copula vs conjunction, locative vs dative, or partitive vs personal analysis",
                    "candidates": {
                        "e": ["e", "è"],
                        "ci": ["ci", "cci"],
                        "ni": ["ni", "nni"],
                    }[value],
                },
            )
        if value in PUBLISHED:
            value = PUBLISHED[value]
            rules.append("L01")
            analyses += self.analyze(value)
        if value.startswith("'") and (
            value[1:] in self.analyses or value[1:] in PUBLISHED_FORMS
        ):
            if value[1:].startswith(("nzir", "nnucc", "mmidi")):
                value = value[1:]
                rules.append("B01")
                analyses += self.analyze(value)
        # Match only explicit orthographic correspondences, not arbitrary lemma
        # variants (which can be synonyms or divergent dialectal morphology).
        forms = self.families.get(family_key(value), set())
        if not analyses and forms:
            analyses = [r for form in sorted(forms) for r in self.analyze(form)]
        if len(value) > 2 and value[-1:] in {"e", "o"}:
            target = value[:-1] + {"e": "i", "o": "u"}[value[-1]]
            target_analyses = self.analyze(target)
            shared_lemma = {key(r["lemma"]) for r in analyses} & {key(r["lemma"]) for r in target_analyses}
            if target_analyses and shared_lemma:
                value, analyses = target, self.analyze(target)
                rules.append("M02")
                forms = self.families.get(family_key(value), set())
        if not analyses:
            return (
                value,
                rules,
                [],
                {
                    "reason": "word_identity",
                    "missing": "native lemma/POS or a supported productive morphological analysis",
                    "candidates": [],
                },
            )
        if any(c in value for c in "áéíóú"):
            value = value.translate(GRAVE)
            rules.append("T02")
        if re.search(r"(?<![zbcdfghjklmnpqrstvwxy])zioni$", value) and any(
            r["pos"] == "NOUN" for r in analyses
        ):
            value = value[:-5] + "zzioni"
            rules.append("M01")
        if "dd" in value:
            retro = {f for f in forms if "ḍḍ" in f}
            if len(retro) == 1 and value not in {"friddu", "addisijari"}:
                value = next(iter(retro))
                rules.append("L01")
            elif value not in {"friddu", "addisijari"} and not all(key(r["lemma"]) in {"friddu", "addisijari"} for r in analyses):
                return (
                    value,
                    rules,
                    analyses,
                    {
                        "reason": "dental_or_retroflex",
                        "missing": "lexical consonant identity/etymology",
                        "candidates": [value, value.replace("dd", "ḍḍ")],
                    },
                )
        stressed = set().union(*(stress_spellings(f.translate(GRAVE)) for f in forms if any(c in f for c in "àèìòùáéíóú")))
        if value in PUBLISHED_FORMS:
            stressed = {value}
        if value.translate(FOLD) not in OPTIONAL_AVIRI and not any(
            c in value for c in "àèìòùâêîôû"
        ):
            if len(stressed) == 1:
                target = next(iter(stressed))
                if target != value:
                    value = target
                    rules.append("S01")
            elif len(stressed) > 1:
                return (
                    value,
                    rules,
                    analyses,
                    {
                        "reason": "stress_ambiguity",
                        "missing": "lexical identity or stress position",
                        "candidates": sorted(stressed),
                    },
                )
            elif (
                value.endswith(("ia", "iu"))
                and value not in PUBLISHED_FORMS
                and not any(r.get("source") == "channel-training" for r in analyses)
            ):
                return (
                    value,
                    rules,
                    analyses,
                    {
                        "reason": "hiatus",
                        "missing": "stress and hiatus/diphthong analysis",
                        "candidates": [value],
                    },
                )
        if re.search(r"(?<!ḍ)ḍ(?!ḍ)|[kwxyç]", value):
            return (
                value,
                rules,
                analyses,
                {
                    "reason": "unsupported_native_spelling",
                    "missing": "borrowing/lexical analysis or extraction repair",
                    "candidates": [value],
                },
            )
        if len(value) > 2 and value.endswith(("e", "o")):
            return value, rules, analyses, {"reason": "atonic_vowel_or_borrowing", "missing": "native final-vowel/stress analysis or foreign-language identity", "candidates": [value]}
        if any(c in value for c in "âêîôû") or "'" in value:
            rules.append("B02")
        return value, rules or ["P01"], analyses, None

    def convert(self, text, *, grammatical=None):
        grammatical = grammatical or {}
        edits, unresolved, analyses, pieces = [], [], [], []
        cursor = 0
        quotes = [
            (m.start(), m.end())
            for m in re.finditer(r'"[^"\n]*"|«[^»]*»|“[^”]*”', text)
        ]
        for token in TOKEN.finditer(text):
            start, end = token.span()
            original = token.group()
            pieces.append(text[cursor:start])
            folded = key(original)
            if original[0].isdigit():
                converted, rules, evidence, issue, kind = (
                    original,
                    [],
                    [],
                    None,
                    "number",
                )
            elif original.isupper() and len(original) > 1 and not self.analyze(folded):
                converted, rules, evidence, issue, kind = original, [], [], None, "abbreviation"
            elif any(a <= start < b for a, b in quotes):
                converted, rules, evidence, issue, kind = (
                    original,
                    [],
                    [],
                    {
                        "reason": "quotation",
                        "missing": "language and quotation provenance",
                        "candidates": [],
                    },
                    "quotation",
                )
            elif (
                start > 0 and original[0].isupper() and not self.analyze(folded)
            ) or any(r["pos"] == "PROPN" for r in self.analyze(folded)):
                converted, rules, evidence, issue, kind = (
                    original,
                    [],
                    [],
                    None,
                    "proper_name",
                )
            else:
                converted, rules, evidence, issue = self.word(
                    original, grammatical.get(start)
                )
                kind = "native_word"
                if original.isupper():
                    converted = converted.upper()
                elif original[0].isupper():
                    converted = converted[0].upper() + converted[1:]
            pieces.append(converted)
            analyses.append(
                {
                    "span": [start, end],
                    "original": original,
                    "converted": converted,
                    "kind": kind,
                    "rules": rules,
                    "analyses": evidence,
                }
            )
            if converted != original:
                edits.append(
                    {
                        "start": start,
                        "end": end,
                        "original": original,
                        "replacement": converted,
                        "rules": rules,
                        "evidence": sorted(
                            {r.get("evidence", r.get("source", "")) for r in evidence}
                        ),
                    }
                )
            if issue:
                unresolved.append({"span": [start, end], "original": original, **issue})
            cursor = end
        pieces.append(text[cursor:])
        if re.search(r"[\ufffd\x00\ue000-\uf8ff]|\w-\s*\n\s*\w", text):
            unresolved.append(
                {
                    "reason": "extraction_damage",
                    "missing": "undamaged source text",
                    "span": [0, len(text)],
                }
            )
        return {
            "original_text": text,
            "converted_text": "".join(pieces),
            "edits": edits,
            "analyses": analyses,
            "unresolved": unresolved,
            "lm_eligible": not unresolved and bool(analyses),
            "converter_version": VERSION,
            "certification": "automated_support_only",
        }


def rule_manifest():
    return {
        "version": VERSION,
        "reference": REFERENCE,
        "rules": [asdict(r) for r in RULES],
    }
