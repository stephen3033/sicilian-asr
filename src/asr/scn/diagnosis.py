"""Text and pronunciation controls for an isolated Sicilian diagnostic study."""

import hashlib
import html
import re
import unicodedata
from collections import Counter, defaultdict

from asr.scn.lexicon import g2p, project
from asr.scn.orthography import WORD, normalize
from common.errors import ResourceError
from common.text import APOSTROPHES
from evaluation.metrics import edits

NORMALIZATION = "scn-diagnostic-single-token-quotes-preserve-elisions-v1"
VOWEL_FOLD = str.maketrans("àèìòùâêîôûáéíóú", "aeiouaeiouaeiou")


def normalize_elisions(text):
    """Do not infer quotation pairs across whitespace-bearing lexical spans.

    Single-token quotations retain the old handling. Multiword ASCII quotation
    and elision are ambiguous; preserve edge marks rather than erase speech.
    This is a diagnostic policy, not a replacement for production DOS24 rules.
    """
    text = unicodedata.normalize("NFC", text).lower().translate(APOSTROPHES)
    text = re.sub(r"(?<!\w)'([^\W\d_]+)'(?!\w)", r"\1", text)
    return " ".join(WORD.findall(text))


def reviewed_normalization(record, ledger):
    from asr.scn.dataset import apply_review

    raw = apply_review(record, ledger["decisions"][record["id"]])
    expected = ledger["decisions"][record["id"]].get("text_sha256")
    if expected and hashlib.sha256(record["text"].encode()).hexdigest() != expected:
        raise ResourceError(f"Stale diagnostic source review: {record['id']}")
    words = []
    for word in normalize_elisions(raw).split():
        replacement = ledger.get("replacements", {}).get(word)
        if replacement:
            if not replacement.get("evidence") or not replacement.get("reason"):
                raise ResourceError(f"Unsourced diagnostic spelling correction: {word}")
            words.extend(normalize_elisions(replacement["replacement"]).split())
        else:
            words.append(word)
    return " ".join(words)


def views(reference, hypothesis):
    reference, hypothesis = normalize(reference), normalize(hypothesis)
    transforms = {
        "strict": lambda x: x,
        "vowel_accents": lambda x: x.translate(VOWEL_FOLD),
        "apostrophes": lambda x: x.replace("'", ""),
        "vowel_accents_and_apostrophes": lambda x: x.translate(VOWEL_FOLD).replace(
            "'", ""
        ),
    }
    result = {}
    for label, transform in transforms.items():
        ref, hyp = transform(reference), transform(hypothesis)
        result[label] = {
            "wer": edits(ref.split(), hyp.split())[0],
            "cer": edits(list(ref.replace(" ", "")), list(hyp.replace(" ", "")))[0],
        }
    result["no_word_boundaries"] = {
        "cer": edits(
            list(reference.replace(" ", "")), list(hypothesis.replace(" ", ""))
        )[0]
    }
    return result


def coverage(words, frequency, lexicon, lm_vocabulary, graph_vocabulary):
    """Trace each reference type through independently supplied coverage sets."""
    accent = defaultdict(list)
    apostrophes = defaultdict(list)
    for word in frequency:
        accent[word.translate(VOWEL_FOLD)].append(word)
        apostrophes[word.replace("'", "")].append(word)
    result = []
    for word, count in sorted(Counter(words).items()):
        if word not in frequency:
            cause = "absent_from_training"
        elif word not in lexicon:
            cause = "pronunciation_rejected"
        elif word not in lm_vocabulary:
            cause = "absent_from_lm"
        elif word not in graph_vocabulary:
            cause = "absent_from_graph"
        else:
            cause = "covered"
        result.append(
            {
                "word": word,
                "tokens": count,
                "cause": cause,
                "in_training": word in frequency,
                "in_lexicon": word in lexicon,
                "in_lm": word in lm_vocabulary,
                "in_graph": word in graph_vocabulary,
                "number": word.isdigit(),
                "unsupported_rule_spelling": not word.isdigit() and not g2p(word),
                "vowel_accent_matches": sorted(accent[word.translate(VOWEL_FOLD)])
                if word not in frequency
                else [],
                "apostrophe_matches": sorted(apostrophes[word.replace("'", "")])
                if word not in frequency
                else [],
            }
        )
    return result


def pronunciation_control(vocabulary, seeds, tokens, tokenizer, *, repair=False):
    """Keep vocabulary fixed; retain seed fallback only if rules cannot project."""
    lexicon, evidence = {}, []
    for word in sorted(vocabulary):
        generated = [
            {"ipa": ipa, "kind": "repaired_rule" if repair else "rule"}
            for ipa in g2p(word, repair_soft_geminates=repair)
        ]
        projected = [(x, project(x["ipa"], tokens, tokenizer)) for x in generated]
        fallback = not any(p for _, p in projected)
        # In the repair arm observed variants retain their original priority;
        # the rules-only arm removes seed preferences while preserving support.
        candidates = (
            list(seeds.get(word, [])) + generated if repair or fallback else generated
        )
        pronunciations = []
        for candidate in candidates:
            ids = project(candidate["ipa"], tokens, tokenizer)
            if ids and ids not in pronunciations:
                pronunciations.append(ids)
                evidence.append(
                    {
                        "word": word,
                        **candidate,
                        "token_ids": ids,
                        "seed_fallback": fallback,
                    }
                )
                if len(pronunciations) == 4:
                    break
        if not pronunciations:
            raise ResourceError(
                f"Pronunciation control lost vocabulary support: {word}"
            )
        lexicon[word] = pronunciations
    return lexicon, evidence


def aggregate(rows):
    result = {"recordings": len(rows)}
    for metric in ("wer", "cer"):
        counts = Counter()
        for row in rows:
            counts.update(
                {k: v for k, v in row["metrics"][metric].items() if type(v) is int}
            )
        result[metric] = {
            **counts,
            "rate": counts["errors"] / counts["reference_units"]
            if counts["reference_units"]
            else None,
        }
    return result


def choose_development(trials):
    """Final-video metrics must never participate in selecting a diagnostic LM."""
    trials = [t for t in trials if t["development"]["wer"]["rate"] is not None]
    if not trials:
        raise ResourceError("No valid development trials")
    return min(
        trials,
        key=lambda t: (
            t["development"]["wer"]["rate"],
            t["order"],
            abs(t["lm_scale"] - 1),
            t["lm_scale"],
        ),
    )


def aggregate_split(reports, split, *, corrected=False):
    """An incomplete split must never acquire a valid-looking partial WER."""
    selected = [r for r in reports if r["split"] == split]
    valid = [r for r in selected if r.get("status") != "invalid"]
    metric = "corrected_reference_metrics" if corrected else "metrics"
    result = aggregate([{"metrics": r[metric]} for r in valid])
    result["failed_recordings"] = [
        r["video"] for r in selected if r.get("status") == "invalid"
    ]
    result["complete"] = not result["failed_recordings"]
    if not result["complete"]:
        for name in ("wer", "cer"):
            result[name]["rate"] = None
    return result


def final_vowel_control(lexicon, tokens, tokenizer):
    """Exploratory ZIPA/lexicon compatibility, not a verified linguistic repair.

    The inspected greedy IPA often ends written unstressed -u/-i words in o/e.
    Preserve every old pronunciation and word, adding final-vowel alternatives
    only for polysyllabic words without a written final stress mark.
    """
    result, evidence = {}, []
    for word, original in lexicon.items():
        result[word] = [list(p) for p in original]
        bare = word.rstrip("'")
        if len(re.findall(r"[aeiouàèìòùâêîôû]", bare)) < 2 or bare[-1:] not in {
            "u",
            "i",
        }:
            continue
        for pronunciation in original:
            ipa = "".join(tokens[i] for i in pronunciation)
            if ipa[-1:] not in {"u", "i"}:
                continue
            alternative = ipa[:-1] + {"u": "o", "i": "e"}[ipa[-1]]
            ids = project(alternative, tokens, tokenizer)
            if ids and ids not in result[word]:
                result[word].append(ids)
                evidence.append(
                    {
                        "word": word,
                        "original_ipa": ipa,
                        "alternative_ipa": alternative,
                        "token_ids": ids,
                    }
                )
    return result, evidence


def table(headers, rows):
    return (
        "<table><thead><tr>"
        + "".join(f"<th>{html.escape(str(h))}</th>" for h in headers)
        + "</tr></thead><tbody>"
        + "".join(
            "<tr>" + "".join(f"<td>{html.escape(str(v))}</td>" for v in row) + "</tr>"
            for row in rows
        )
        + "</tbody></table>"
    )
