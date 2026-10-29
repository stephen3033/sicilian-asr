import pytest

from pipeline.scn_residual_report import classify
from pipeline.scn_residual_tests import scale_pronunciation_costs, semantic_settings


@pytest.mark.parametrize(
    "flags,category",
    [
        (["lm_score_reversal"], "lm_preference"),
        (["pronunciation_cost_reversal"], "pronunciation_cost"),
        (
            ["lm_score_reversal", "pronunciation_cost_reversal"],
            "mixed_lm_pronunciation",
        ),
        (
            ["acoustic_evidence_conflict"],
            "acoustic_or_pronunciation_support_unresolved",
        ),
    ],
)
def test_causal_categories_require_mechanistic_evidence(flags, category):
    error = {
        "caption_interval": {"start_seconds": 1, "end_seconds": 2},
        "segment_interval": [0, 20],
        "cross_boundary_deletion": False,
        "alignment_ambiguous": False,
    }
    case = {
        "id": "case",
        "error_ids": ["error"],
        "kinds": ["individual"],
        "mechanistic_flags": flags,
        "delta": {"lm_score": -2},
    }
    assert classify(error, [case], set())[0] == category
    # Uncertain frame ownership prevents an acoustic/word-specific claim.
    error["caption_interval"] = {"start_seconds": 21, "end_seconds": 22}
    assert classify(error, [case], set())[0] == "alignment_or_segmentation_unresolved"
    error["caption_interval"] = None
    error["alignment_ambiguous"] = True
    assert classify(error, [case], set())[0] == "ambiguous_edit_alignment"


def test_reuse_canonicalizes_numeric_settings_without_changing_labels_or_costs():
    assert semantic_settings(
        {"word_bonus": 0, "max_active_states": 10000}
    ) == semantic_settings(
        {"word_bonus": 0.0, "max_active_states": 10000.0, "compress_artifacts": True}
    )
    settings = {
        "phone_seed_costs": {"4": 2},
        "phone_optional_costs": {"8": 1},
        "phone_alternatives": {"4": [{"token": 5, "cost": 3}]},
        "pronunciation_settings": {"word": [{"exact": True, "cost": 4}]},
    }
    calibrated = scale_pronunciation_costs(settings, 0.5)
    assert calibrated["phone_alternatives"]["4"] == [{"token": 5, "cost": 1.5}]
    assert calibrated["pronunciation_settings"]["word"] == [{"exact": True, "cost": 2}]
    assert (
        settings["phone_seed_costs"]["4"] == 2
        and calibrated["phone_seed_costs"]["4"] == 1
    )
