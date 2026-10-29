import math

import kenlm
import pytest

from pipeline.scn_residual_lm import BackoffModel, joined_cues
from tests.support import ROOT


def test_support_floor_adds_no_fake_contexts_and_normalizes_backoff(tmp_path):
    model = BackoffModel(ROOT / "tests/fixtures/tiny.arpa")
    contexts = {ws for ws in model.probabilities if len(ws) > 1}
    audit = model.supported({"uno", "una", "due", "unseen"}, 0.001)
    assert {ws for ws in model.probabilities if len(ws) > 1} == contexts
    assert model.probability("unseen") > 0
    assert audit["after"]["maximum_conditional_mass_error"] < 1e-7
    path = tmp_path / "model.arpa"
    model.write(path)
    reloaded = BackoffModel(path)
    assert reloaded.audit()["maximum_conditional_mass_error"] < 1e-7
    native = kenlm.Model(str(path))
    for text in ("uno due", "unseen una", "due uno unseen", ""):
        history, total = ("<s>",), 0
        for word in [*text.split(), "</s>"]:
            total += math.log10(model.probability(word, history))
            history = (*history, word)[-2:]
        assert native.score(text) == pytest.approx(total, abs=2e-6)


def test_cue_boundaries_never_bridge_exclusions_or_recordings():
    rows = [
        {
            "recording_id": video,
            "id": f"{video}:{i}",
            "text": str(i),
            "start_seconds": start,
            "end_seconds": start + 1,
        }
        for video, i, start in (
            ("a", 0, 0),
            ("a", 1, 1),
            ("a", 3, 2),
            ("a", 4, 8),
            ("b", 0, 9),
        )
    ]
    joined = joined_cues(rows)
    assert [r["cue_ids"] for r in joined] == [["a:0", "a:1"], ["a:3"], ["a:4"], ["b:0"]]
    assert sum(len(r["text"].split()) for r in joined) == len(rows)
