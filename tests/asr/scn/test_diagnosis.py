from collections import Counter

import pytest

from asr.scn.diagnosis import (
    aggregate_split,
    choose_development,
    coverage,
    final_vowel_control,
    normalize_elisions,
    pronunciation_control,
    views,
)
from asr.scn.lexicon import g2p
from common.errors import ResourceError


def test_diagnostic_quote_handling_preserves_two_independent_elisions():
    assert (
        normalize_elisions("‘N Sicilia, i du’ su’ cca.") == "'n sicilia i du' su' cca"
    )
    assert normalize_elisions("'lingua' e 'n Sicilia po'") == "lingua e 'n sicilia po'"
    text = "'n sicilia po'"
    assert normalize_elisions(normalize_elisions(text)) == text


def test_relaxed_view_does_not_merge_retroflex_and_dental_spelling():
    metrics = views("beḍḍu è", "beddu e")
    assert metrics["vowel_accents"]["wer"]["errors"] == 1
    assert metrics["strict"]["wer"]["errors"] == 2


@pytest.mark.parametrize(
    "word, expected",
    [
        ("cci", "tʃːi"),
        ("accittata", "atʃːitːata"),
        ("maggiu", "madʒːu"),
        ("cacciari", "katʃːari"),
    ],
)
def test_soft_geminate_repair_is_opt_in(word, expected):
    assert g2p(word, repair_soft_geminates=True)[0] == expected
    assert g2p(word) != g2p(word, repair_soft_geminates=True)
    assert g2p("cchiù", repair_soft_geminates=True) == g2p("cchiù")
    assert g2p("chi", repair_soft_geminates=True) == ["ki"]


def test_coverage_traces_independent_stages():
    frequency = Counter({"a": 5, "b": 3, "c": 2, "d": 1})
    rows = coverage(
        ["a", "b", "c", "d", "e", "e"], frequency, {"a", "c", "d"}, {"a", "d"}, {"a"}
    )
    assert {r["word"]: r["cause"] for r in rows} == {
        "a": "covered",
        "b": "pronunciation_rejected",
        "c": "absent_from_lm",
        "d": "absent_from_graph",
        "e": "absent_from_training",
    }
    assert rows[-1]["tokens"] == 2


def test_pronunciation_control_keeps_words_that_require_seed_fallback():
    tokens = ["<blk>", "<s>", "<unk>", "▁", "a", "t"]

    class Tokenizer:
        def encode(self, text, out_type):
            return [3] + [tokens.index(c) if c in tokens else 2 for c in text]

    lexicon, evidence = pronunciation_control(
        {"a", "texas"}, {"texas": [{"ipa": "t", "kind": "seed"}]}, tokens, Tokenizer()
    )
    assert lexicon == {"a": [[4]], "texas": [[5]]}
    assert next(r for r in evidence if r["word"] == "texas")["seed_fallback"]


def test_lm_selection_never_uses_final_scores():
    best_dev = {
        "development": {"wer": {"rate": 0.4}},
        "test": {"wer": {"rate": 0.9}},
        "order": 3,
        "lm_scale": 1,
    }
    best_test = {
        "development": {"wer": {"rate": 0.5}},
        "test": {"wer": {"rate": 0.1}},
        "order": 2,
        "lm_scale": 1,
    }
    assert choose_development([best_dev, best_test]) is best_dev


def test_invalid_decoder_trial_cannot_win_selection_or_report_partial_wer():
    invalid = {"development": {"wer": {"rate": None}}, "order": 1, "lm_scale": 2}
    valid = {"development": {"wer": {"rate": 0.5}}, "order": 3, "lm_scale": 1}
    assert choose_development([invalid, valid]) is valid
    with pytest.raises(ResourceError, match="No valid development"):
        choose_development([invalid])
    result = aggregate_split(
        [
            {
                "video": "a",
                "split": "test",
                "status": "complete",
                "metrics": {
                    "wer": {"errors": 1, "reference_units": 2},
                    "cer": {"errors": 1, "reference_units": 4},
                },
            },
            {"video": "b", "split": "test", "status": "invalid"},
        ],
        "test",
    )
    assert result["wer"]["rate"] is None
    assert result["failed_recordings"] == ["b"]
    assert not result["complete"]


def test_final_vowel_probe_preserves_words_and_original_pronunciations():
    tokens = ["<blk>", "<s>", "<unk>", "▁", "a", "i", "u", "o", "e"]

    class Tokenizer:
        def encode(self, text, out_type):
            return [3] + [tokens.index(c) for c in text]

    original = {"aiu": [[4, 5, 6]], "tu": [[6]], "più": [[6]]}
    lexicon, evidence = final_vowel_control(original, tokens, Tokenizer())
    assert set(lexicon) == set(original)
    assert lexicon["aiu"] == [[4, 5, 6], [4, 5, 7]]
    assert lexicon["tu"] == original["tu"]
    assert lexicon["più"] == original["più"]
    assert original["aiu"] == [[4, 5, 6]]
    assert evidence[0]["alternative_ipa"] == "aio"
