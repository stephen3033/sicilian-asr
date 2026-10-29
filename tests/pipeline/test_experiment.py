import json

import pytest

from common.errors import ResourceError
from pipeline.experiment import aggregate, improves, score, seal, verified_files
from pipeline.scn_lm_experiment import mixture
from pipeline.scn_pronunciation import coordinate, estimate


def test_strict_scores_and_selection_preserve_dos24_distinctions():
    assert score("e ci", "è cci")["wer"]["errors"] == 2
    base = {"wer": {"rate": 0.45}, "cer": {"rate": 0.17}}
    assert not improves({"wer": {"rate": 0.44}, "cer": {"rate": 0.18}}, base)
    assert improves({"wer": {"rate": 0.44}, "cer": {"rate": 0.17}}, base)
    assert improves({"wer": {"rate": 0.45}, "cer": {"rate": 0.16}}, base)
    assert not improves(base, base)
    rows = [
        {"metrics": score("e", "è")},
        {"metrics": score("unu dui tri", "unu dui tri")},
    ]
    assert aggregate(rows)["wer"]["rate"] == 0.25


def test_receipt_rejects_changed_bytes_and_traversal(tmp_path):
    (tmp_path / "model.txt").write_text("original")
    saved = seal(tmp_path, "receipt.json", {"version": 1}, ["model.txt"])
    verified_files(tmp_path, saved)
    (tmp_path / "model.txt").write_text("changed")
    with pytest.raises(ResourceError, match="checksum"):
        verified_files(tmp_path, saved)
    with pytest.raises(ResourceError, match="escapes"):
        verified_files(tmp_path, {"artifacts": {"../outside": "digest"}})


@pytest.mark.parametrize("share", [0.05, 0.2, 0.5])
def test_matched_mixtures_keep_boundaries_and_effective_counts(share):
    channel = ["unu dui", "tri quattru cincu"]
    raw = ["chiu cavaddu", "cchiu friddu"]
    converted = ["cchiù cavaḍḍu", "cchiù friddu"]
    weighted_raw, a = mixture(channel, raw, share)
    weighted_converted, b = mixture(channel, converted, share)
    assert a == b
    assert a["effective_wikipedia_share"] <= share
    assert len(weighted_raw) == len(weighted_converted)
    assert set(weighted_raw) == set(channel + raw)
    assert a["wikipedia_unique_sentences"] == 2
    assert a["channel_effective_tokens"] >= a["channel_source_tokens"]


def test_effective_mixture_share_counts_fixed_vocabulary_prior():
    texts, weight = mixture(["a b"], ["c d e f"], 0.5, prior_tokens=100)
    assert weight["wikipedia_copies"] > 1
    assert weight["effective_wikipedia_share"] <= 0.5
    total = sum(len(t.split()) for t in texts) + 100
    assert (
        weight["effective_wikipedia_share"]
        == weight["wikipedia_effective_tokens"] / total
    )


def test_pronunciation_support_counts_recordings_and_rebuilds_each_fold():
    rows = [
        {
            "video": v,
            "word": "ci",
            "seed": "i",
            "observed": "e",
            "offset": 1,
            "length": 2,
        }
        for v in ("a", "b", "c")
        for _ in range(10)
    ]
    thresholds = {"minimum_count": 20, "minimum_fraction": 0.1, "minimum_recordings": 3}
    accepted = estimate(rows, "context", thresholds)[0]
    assert coordinate(rows[0], "context") in accepted
    assert not estimate([r for r in rows if r["video"] != "c"], "context", thresholds)[
        0
    ]
    assert coordinate({**rows[0], "word": "cci"}, "context") not in accepted


def test_recipe_defines_fourteen_controls_and_frozen_thresholds():
    from tests.support import ROOT

    recipe = json.loads((ROOT / "configs/scn-improve.json").read_text())
    assert (
        len(recipe["lm"]["orders"]) * (1 + 2 * len(recipe["lm"]["wikipedia_shares"]))
        == 14
    )
    assert recipe["lm"]["lm_scales"] == [0.6, 0.8, 1.0]
    assert recipe["vocabulary"]["minimum_articles"] == 5
    assert recipe["pronunciation"]["minimum_recordings"] == 3


def test_review_frequency_ranks_candidates_without_resolving_them():
    from collections import Counter

    from pipeline.scn_review import ranked_candidates

    result = ranked_candidates(["e", "è"], Counter({"è": 10, "e": 2}))
    assert [r["text"] for r in result] == ["è", "e"]
    assert all(set(r) == {"text", "channel_training_count"} for r in result)


def test_partial_experiment_cannot_feed_candidate_selection(tmp_path):
    from pipeline.scn_experiment import completed

    path = tmp_path / "lm-results.json"
    path.write_text(json.dumps({"status": "running", "results": [{"id": "first"}]}))
    with pytest.raises(ResourceError, match="Prerequisite stage is incomplete"):
        completed(path)
