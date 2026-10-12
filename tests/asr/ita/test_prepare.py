import json


def test_graph_identity_separates_adaptation_settings(tmp_path, monkeypatch):
    from copy import deepcopy

    from asr.ita import prepare as preparation

    sha = "a" * 64
    manifest = {"resources": {"lm": {"path": "lm.bin", "revision": "fixed"}}}
    config = {
        "resources": {"ita": {"lm_resource": "lm", "arpa_path": "model.arpa"}},
        "decoder": {"vocabulary_size": 20, "ctc_topology": "compact", "lm_scale": 1.0},
    }
    (tmp_path / "lm.bin.members.json").write_text(
        json.dumps({"archive_sha256": sha, "artifacts": {"model.arpa": {}}})
    )
    monkeypatch.setattr(preparation, "required_resources", lambda _: ["lm"])
    monkeypatch.setattr(preparation, "verify_file", lambda *args: {"sha256": sha})
    keys = set()
    for pronunciation, contractions in [
        (False, False),
        (True, False),
        (False, True),
        (True, True),
    ]:
        changed = deepcopy(config)
        changed["decoder"].update(
            pronunciation_variants=pronunciation, contractions=contractions
        )
        key, payload = preparation.identity(tmp_path, changed, manifest)
        assert payload["settings"]["pronunciation_variants"] == pronunciation
        assert "asr/ita/lexicon.py" in payload["compiler_sources"]
        keys.add(key)
    assert len(keys) == 4
