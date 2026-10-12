"""Italian evaluation adds conservative spoken-number metrics to strict scores."""

from asr.ita.spoken import SPOKEN_RULES, spoken_form
from evaluation.metrics import evaluate as evaluate_text


def evaluate(hypothesis, reference, full_lexicon, selected):
    return evaluate_text(
        hypothesis,
        reference,
        full_lexicon,
        selected,
        spoken_normalizer=spoken_form,
        spoken_rules=SPOKEN_RULES,
    )
