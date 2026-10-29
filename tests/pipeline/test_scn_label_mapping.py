import math

import pytest

from pipeline.scn_label_mapping import empirical_label_costs, positional_label_costs

TOKENS = ["<blank>", "<pad>", "<unk>", "▁", "u", "o", "e", "ŋ", "n"]


def observations(video, seed, counts):
    return {
        "video": video,
        "alignment": [
            {"reference": seed, "hypothesis": label}
            for label, count in counts.items()
            for _ in range(count)
        ],
    }


def test_empirical_preference_can_reverse_seed_and_retains_smoothed_fallback():
    records = [observations(v, "u", {"o": 12, "u": 6, None: 2}) for v in range(3)]
    records += [observations(v, "ŋ", {"n": 8}) for v in range(3)]
    alternatives, seeds, evidence = empirical_label_costs(records, TOKENS, 1.5)
    assert alternatives == {
        "4": [{"token": 5, "cost": 0}],
        "7": [{"token": 8, "cost": 0}],
    }
    assert seeds["4"] == pytest.approx(1.5 * math.log(36.5 / 18.5))
    assert seeds["7"] == pytest.approx(1.5 * math.log(24.5 / 0.5))
    assert evidence[0]["total"] == 60
    assert evidence[0]["options"][1]["recordings"] == 3


def test_substitutions_require_count_fraction_and_recording_support():
    # Repeated occurrences from one recording do not establish recurrence.
    records = [observations(0, "u", {"o": 100})]
    assert empirical_label_costs(records, TOKENS) == ({}, {}, [])
    records = [observations(v, "u", {"o": 7, "u": 100}) for v in range(3)]
    assert empirical_label_costs(records, TOKENS) == ({}, {}, [])
    records = [observations(v, "u", {"o": 6}) for v in range(3)]
    assert empirical_label_costs(records, TOKENS) == ({}, {}, [])


@pytest.mark.parametrize("strength", [0, -1, float("inf"), float("nan")])
def test_invalid_strength_rejected(strength):
    with pytest.raises(ValueError):
        empirical_label_costs([], TOKENS, strength)


def test_position_costs_use_aggregated_counts_and_can_reverse_preference():
    rows = []
    for offset, counts in [(0, {"u": 65, "o": 40}), (1, {"u": 46, "o": 114})]:
        for label, count in counts.items():
            rows.append(
                {
                    "seed_label_offset": offset,
                    "seed_label_length": 2,
                    "seed": "u",
                    "observed": label,
                    "count": count,
                    "recordings": ["a", "b", "c"],
                }
            )
    mapping, evidence = positional_label_costs(rows, TOKENS)
    assert mapping["final"]["4"][1]["cost"] == 0
    assert mapping["final"]["4"][0]["cost"] == pytest.approx(math.log(114.5 / 46.5))
    assert mapping["nonfinal"]["4"][0]["cost"] == 0
    assert mapping["nonfinal"]["4"][1]["cost"] == pytest.approx(math.log(65.5 / 40.5))
    assert [r["total"] for r in evidence] == [160, 105]
    # Support is the union of recordings, not their sum over duplicate rows.
    for row in rows:
        row["recordings"] = ["a"]
    assert positional_label_costs(rows * 3, TOKENS) == ({}, [])
