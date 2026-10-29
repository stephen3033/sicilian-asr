from pipeline.experiment import score
from pipeline.scn_residual_data import candidate_effects, inventory


def fixture(reference="a b c d e", hypothesis="a x c e z"):
    recordings = [
        {
            "video": "v",
            "reference": reference,
            "cues": [{"text": reference, "start_seconds": 0, "end_seconds": 4}],
        }
    ]
    decoded = [
        {
            "video": "v",
            "text": hypothesis,
            "metrics": score(hypothesis, reference),
            "segments": [
                {
                    "index": 0,
                    "text": hypothesis,
                    "source_start_seconds": 0,
                    "source_end_seconds": 4,
                    "forced_cut": False,
                }
            ],
        }
    ]
    return recordings, decoded


def test_inventory_keeps_stable_target_ids_and_counterfactuals_out_of_training():
    recordings, decoded = fixture()
    lexicon = {w: [[4]] for w in ["a", "b", "c", "d", "e", "x", "z"]}
    data = inventory(recordings, decoded, lexicon)
    assert data["target_counts"] == {
        "substitutions": 1,
        "deletions": 1,
        "insertions": 1,
    }
    ids = {r["id"] for r in data["errors"]}
    assert ids == {i for c in data["cases"] for i in c["error_ids"]}
    assert all(
        c["reference_constrained"] and not c["training_eligible"] for c in data["cases"]
    )
    fixed = [{"video": "v", "text": recordings[0]["reference"]}]
    effects = candidate_effects(
        data["errors"], decoded, fixed, {"v": recordings[0]["reference"]}
    )
    assert set(effects["repaired_target_ids"]) == ids
    assert effects["new_reference_position_errors_and_insertions"] == 0


def test_oov_reference_is_counted_but_never_inserted_into_counterfactuals():
    recordings, decoded = fixture("a absent c", "a x c")
    lexicon = {w: [[4]] for w in ["a", "x", "c"]}
    data = inventory(recordings, decoded, lexicon)
    assert data["total_errors"] == 1 and data["target_counts"] == {}
    assert not data["cases"]
