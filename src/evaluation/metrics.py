"""Cue-aware reference parsing and reproducible edit counts."""

from collections import Counter

from common.errors import ResourceError
from common.text import TEXT_RULES, normalize


def edits(reference: list[str], hypothesis: list[str]) -> tuple[dict, list]:
    import kaldialign

    epsilon = "\0"
    if epsilon in reference or epsilon in hypothesis:
        raise ResourceError("NUL is not a supported reference symbol")
    alignment = kaldialign.align(reference, hypothesis, epsilon)
    counts = Counter(substitutions=0, deletions=0, insertions=0, correct=0)
    rows = []
    for ref, hyp in alignment:
        operation = (
            "insertions"
            if ref == epsilon
            else "deletions"
            if hyp == epsilon
            else "correct"
            if ref == hyp
            else "substitutions"
        )
        counts[operation] += 1
        rows.append(
            {
                "operation": operation,
                "reference": None if ref == epsilon else ref,
                "hypothesis": None if hyp == epsilon else hyp,
            }
        )
    errors = counts["substitutions"] + counts["deletions"] + counts["insertions"]
    return {
        **counts,
        "errors": errors,
        "reference_units": len(reference),
        "hypothesis_units": len(hypothesis),
        "rate": errors / len(reference) if reference else None,
        "undefined_reason": None if reference else "empty reference",
    }, rows


def oov(words: list[str], vocabulary: set[str]) -> dict:
    missing = Counter(w for w in words if w not in vocabulary)
    return {
        "tokens": sum(missing.values()),
        "types": len(missing),
        "reference_tokens": len(words),
        "words": dict(sorted(missing.items())),
    }


def evaluate(
    hypothesis: str,
    reference: dict,
    full_lexicon: set[str],
    selected: set[str],
    *,
    strict_normalizer=None,
    strict_rules: str | None = None,
    spoken_normalizer=None,
    spoken_rules: str | None = None,
) -> tuple[dict, str]:
    normalizer = strict_normalizer or normalize
    hyp = normalizer(hypothesis)
    ref = (
        normalizer(reference["text"]) if strict_normalizer else reference["normalized"]
    )
    wer, alignment = edits(ref.split(), hyp.split())
    cer, _ = edits(list(ref.replace(" ", "")), list(hyp.replace(" ", "")))
    metrics = {
        "normalization": strict_rules or TEXT_RULES,
        "wer": wer,
        "cer": cer,
        "oov_full_supported_lexicon": oov(ref.split(), full_lexicon),
        "oov_decoding_vocabulary": oov(ref.split(), selected),
        "per": None,
        "per_reason": "No IPA reference",
        "word_alignment": alignment,
    }
    if spoken_normalizer is not None:
        spoken_ref = spoken_normalizer(reference.get("text", reference["normalized"]))
        spoken_hyp = spoken_normalizer(hypothesis)
        spoken_wer, spoken_alignment = edits(
            spoken_ref["text"].split(), spoken_hyp["text"].split()
        )
        spoken_cer, _ = edits(
            list(spoken_ref["text"].replace(" ", "")),
            list(spoken_hyp["text"].replace(" ", "")),
        )
        metrics["spoken_form"] = {
            "normalization": spoken_rules,
            "reference": spoken_ref,
            "hypothesis": spoken_hyp,
            "wer": spoken_wer,
            "cer": spoken_cer,
            "word_alignment": spoken_alignment,
            "oov_full_supported_lexicon": oov(spoken_ref["text"].split(), full_lexicon),
            "oov_decoding_vocabulary": oov(spoken_ref["text"].split(), selected),
        }
    metrics["oov_causes"] = [
        {
            "word": w,
            "count": count,
            "cause": "absent_from_supported_lexicon"
            if w not in full_lexicon
            else "outside_decoding_vocabulary",
            "spoken_form": spoken_normalizer(w)["text"]
            if spoken_normalizer
            else normalizer(w),
        }
        for w, count in metrics["oov_decoding_vocabulary"]["words"].items()
    ]
    comparison = (
        f"REFERENCE: {ref}\nHYPOTHESIS: {hyp}\n\noperation\treference\thypothesis\n"
    )
    comparison += "".join(
        f"{a['operation']}\t{a['reference'] or '∅'}\t{a['hypothesis'] or '∅'}\n"
        for a in alignment
    )
    if spoken_normalizer is not None:
        comparison += f"\nSPOKEN-FORM REFERENCE: {spoken_ref['text']}\nSPOKEN-FORM HYPOTHESIS: {spoken_hyp['text']}\n"
    return metrics, comparison
