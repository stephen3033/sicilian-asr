"""Strict DOS24 spelling evaluation without Italian number verbalization."""

from asr.scn.orthography import RULES, normalize
from evaluation.metrics import evaluate as evaluate_text


def evaluate(hypothesis, reference, full_lexicon, selected):
    return evaluate_text(
        hypothesis,
        reference,
        full_lexicon,
        selected,
        strict_normalizer=normalize,
        strict_rules=RULES,
    )
