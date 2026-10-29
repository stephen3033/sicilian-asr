import math

import pytest

from pipeline.scn_word_labels import (
    add_observed_word_paths,
    interior_word_labels,
    recurring_word_labels,
)


def pairs(*values):
    return [{"reference": a, "hypothesis": b} for a, b in values]


def test_interior_insertion_and_deletion_preserve_word_assignment():
    words, seeds = ["one", "two", "three"], [["a"], ["b", "c"], ["d"]]
    observed = interior_word_labels(
        words,
        seeds,
        pairs(
            ("a", "a"),
            ("b", "x"),
            (None, "y"),
            ("c", "z"),
            ("d", "d"),
        ),
    )
    assert observed == [{"word": "two", "labels": ["x", "y", "z"]}]
    assert interior_word_labels(
        words,
        seeds,
        pairs(
            ("a", "a"),
            ("b", None),
            ("c", "z"),
            ("d", "d"),
        ),
    ) == [{"word": "two", "labels": ["z"]}]


def test_boundary_ambiguity_empty_outputs_and_cue_edges_are_excluded():
    words, seeds = ["one", "two", "three"], [["a"], ["b"], ["c"]]
    assert (
        interior_word_labels(
            words,
            seeds,
            pairs(
                ("a", "a"),
                (None, "x"),
                ("b", "b"),
                ("c", "c"),
            ),
        )
        == []
    )
    assert (
        interior_word_labels(
            words,
            seeds,
            pairs(
                ("a", "a"),
                ("b", None),
                ("c", "c"),
            ),
        )
        == []
    )
    assert interior_word_labels(["one"], [["a"]], pairs(("a", "a"))) == []


def test_alignment_identity_is_required():
    with pytest.raises(ValueError, match="does not match"):
        interior_word_labels(["one"], [["a"]], pairs(("b", "b")))
    with pytest.raises(ValueError, match="omits"):
        interior_word_labels(["one"], [["a"]], [])


def test_recurrence_requires_independent_recordings_and_fraction():
    rows = [
        {"word": "word", "labels": ["x"], "video": video} for video in ["a", "b", "c"]
    ]
    accepted = recurring_word_labels(rows)
    assert accepted[0]["count"] == 3
    assert accepted[0]["recordings"] == ["a", "b", "c"]
    assert recurring_word_labels([rows[0]] * 10) == []
    rows += [{"word": "word", "labels": ["y"], "video": "a"}] * 13
    assert recurring_word_labels(rows) == []


def test_observed_paths_have_empirical_priors_without_expanding_vocabulary():
    tokens = ["<blank>", "<s>", "<unk>", "▁", "a", "b", "c"]
    original = {"word": [[4]], "other": [[5]]}
    accepted = [
        {"word": "word", "labels": ["b"], "count": 10},
        {"word": "word", "labels": ["c"], "count": 5},
        {"word": "absent", "labels": ["a"], "count": 20},
    ]
    lexicon, settings, _ = add_observed_word_paths(
        original, accepted, tokens, "empirical", 0.5
    )
    assert set(lexicon) == set(original)
    assert lexicon["word"] == [[4], [5], [6]]
    assert original["word"] == [[4]]
    assert settings["word"][0] == {"cost": 0.5 * math.log(21), "exact": False}
    assert settings["word"][1] == {"cost": 0, "exact": True}
    assert settings["word"][2]["cost"] == pytest.approx(0.5 * math.log(10.5 / 5.5))
    _, control, _ = add_observed_word_paths(original, accepted, tokens, "append", 0.5)
    assert control["word"] == [
        {"cost": 0.0, "exact": False},
        {"cost": 0.5, "exact": True},
        {"cost": 0.5, "exact": True},
    ]
