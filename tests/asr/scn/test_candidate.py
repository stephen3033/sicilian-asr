"""Portable candidates must agree with production acoustic labels and costs."""

import pytest

from asr.scn import prepare
from common.artifacts import seal
from common.errors import ResourceError
from common.utils import write_json


@pytest.mark.parametrize("mismatch", ["tokens", "costs"])
def test_candidate_preparation_rejects_incompatible_inputs(
    tmp_path, monkeypatch, mismatch
):
    directory = tmp_path / "candidate"
    write_json(
        directory / "graph/compile_input.json",
        {
            "tokens": ["wrong"] if mismatch == "tokens" else ["<blk>", "a"],
            "settings": {"lm_scale": 0.8, "word_bonus": 0},
        },
    )
    seal(directory, "candidate.json", {}, ["graph/compile_input.json"])
    monkeypatch.setattr(prepare, "inventory", lambda *_: (["<blk>", "a"], None))
    config = {
        "resources": {"scn": {"candidate_bundle_path": "candidate"}},
        "acoustic": {
            "vocabulary_path": "tokens.txt",
            "tokenizer_path": "tokenizer.model",
        },
        "decoder": {"lm_scale": 1.0 if mismatch == "costs" else 0.8, "word_bonus": 0},
    }
    with pytest.raises(
        ResourceError,
        match="inventories disagree" if mismatch == "tokens" else "costs disagree",
    ):
        prepare.identity(tmp_path, config, {})
