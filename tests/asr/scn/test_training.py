import hashlib
import json

import pytest

from asr.scn.dataset import apply_review, digest, passage_hashes, reviewed_text
from asr.scn.evaluation import evaluate
from asr.scn.lexicon import g2p, project
from asr.scn.orthography import normalize
from asr.scn.training import score, verify_bundle
from common.errors import ResourceError
from common.text import normalize as italian_normalize
from common.utils import fingerprints


def test_sicilian_apostrophes_accents_and_retroflex_are_distinct():
    assert (
        normalize("‘n Sicilia, âmu a parrari di 'lingua'.")
        == "'n sicilia âmu a parrari di lingua"
    )
    assert normalize("beḍḍu è bellu") == "beḍḍu è bellu"
    assert italian_normalize("‘n Sicilia") == "n sicilia"
    metrics, _ = evaluate(
        "n sicilia", {"text": "'n sicilia"}, {"'n", "sicilia"}, {"'n", "sicilia"}
    )
    assert metrics["wer"]["substitutions"] == 1
    assert "spoken_form" not in metrics


def test_reviews_bind_exact_source_text():
    row = {"id": "x", "text": "Cca sugnu."}
    review = {
        "decisions": {
            "x": {
                "decision": "include",
                "reviewer": "test",
                "reason": "checked",
                "text_sha256": hashlib.sha256(row["text"].encode()).hexdigest(),
            }
        }
    }
    assert reviewed_text(row, review)[0] == "cca sugnu"
    assert reviewed_text(row, {}) is None
    row["text"] = "Changed"
    with pytest.raises(ResourceError, match="Stale"):
        reviewed_text(row, review)


def test_review_word_correction_preserves_embedded_stress():
    record = {"id": "example", "text": "lu sò pinseri filusòficu"}
    decision = {"edits": [{"old": "sò", "new": "so", "mode": "word"}]}
    assert apply_review(record, decision) == "lu so pinseri filusòficu"
    with pytest.raises(ResourceError, match="Stale"):
        apply_review({"id": "changed", "text": "filusòficu"}, decision)


def test_sicilian_rule_families_are_bounded():
    assert "ɖː" in g2p("beḍḍu")[0]
    assert "dː" in g2p("friddu")[0]
    assert g2p("hê") == ["ɛ"]
    assert g2p("harbia")[0].startswith("x")
    assert g2p("cchiù")[0] == "cːu"
    assert g2p("chi") == ["ki"]
    assert g2p("ciuri")[0].startswith("tʃ")
    assert 1 <= len(g2p("zzazzà")) <= 4
    assert g2p("123") == []


def test_project_does_not_erase_unknown_diacritics():
    class Tokenizer:
        def encode(self, text, out_type):
            return [3] + [tokens.index(c) if c in tokens else 2 for c in text]

    tokens = ["<blk>", "<s>", "<unk>", "▁", "g", "a"]
    assert project("ˈɡa", tokens, Tokenizer()) == [4, 5]
    assert project("gã", tokens, Tokenizer()) is None


def test_corrupt_model_receipt_is_rejected(tmp_path):
    (tmp_path / "model.json").write_text(
        json.dumps({"status": "complete", "key": "wrong", "identity": {}})
    )
    with pytest.raises(ResourceError, match="invalid"):
        verify_bundle(tmp_path, "model.json")


def test_changed_model_artifact_is_rejected(tmp_path):
    model = tmp_path / "model.arpa"
    model.write_text("original")
    receipt = {
        "status": "complete",
        "key": digest({}),
        "identity": {},
        "artifacts": {"model.arpa": fingerprints(model)["sha256"]},
    }
    (tmp_path / "model.json").write_text(json.dumps(receipt))
    verify_bundle(tmp_path, "model.json")
    model.write_text("changed")
    with pytest.raises(ResourceError, match="Changed"):
        verify_bundle(tmp_path, "model.json")


def test_development_score_cannot_overwrite_training_token_count():
    from tests.support import ROOT

    result = {
        "tokens": 10000,
        **score(ROOT / "tests/fixtures/tiny.arpa", ["uno absent"], {"uno"}),
    }
    assert result["tokens"] == 10000
    assert result["development_tokens"] == 2
    assert result["oov_rate"] == 0.5


def test_passage_screen_finds_overlap_without_exposing_text():
    heldout = "one two three four five six seven eight nine ten eleven twelve".split()
    assert passage_hashes(heldout) & passage_hashes(["prefix", *heldout, "suffix"])
    assert not passage_hashes(heldout[:-1])
    assert all(len(value) == 64 for value in passage_hashes(heldout))
