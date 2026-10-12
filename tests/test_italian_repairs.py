from pathlib import Path
import json

import pytest

from sicilian_asr.evaluation import evaluate
from sicilian_asr.lexicon.prepare import adapt
from sicilian_asr.spoken import integer_italian, spoken_form


def test_pronunciation_alternatives_preserve_length_and_source():
    tokens = list(dict.fromkeys("ɔdʒːianŋktrgof"))
    encode = lambda s: [tokens.index(c) for c in s]
    lex = {"oggi": [encode("ɔdʒːi")], "anni": [encode("anːi")],
           "incontro": [encode("iŋkontro")], "ano": [encode("ano")], "lungo": [encode("aŋgo")]}
    result, audit = adapt(lex, tokens, set(), pronunciation_variants=True, contractions=False)
    decode = lambda w: {"".join(tokens[i] for i in p) for p in result[w]}
    assert decode("oggi") == {"ɔdʒːi", "ɔddʒi"}
    assert decode("anni") == {"anːi", "anni"}
    assert decode("incontro") == {"iŋkontro", "inkontro"}
    assert decode("ano") == {"ano"}
    assert decode("lungo") == {"aŋgo", "ango"}
    assert lex["anni"] == [encode("anːi")]
    assert len(audit["added_pronunciations"]) == 4


def test_contractions_require_lm_suffix_and_initial_vowel():
    tokens = list(dict.fromkeys("lɔtʃɛkas"))
    encode = lambda s: [tokens.index(c) for c in s]
    lex = {"ho": [encode("ɔ")], "è": [encode("ɛ")], "casa": [encode("kasa")]}
    lm = {"l'ho", "c'è", "l'inesistente", "d'è", "l'casa"}
    result, audit = adapt(lex, tokens, lm, pronunciation_variants=False, contractions=True)
    assert result["l'ho"] == [encode("lɔ")]
    assert result["c'è"] == [encode("tʃɛ")]
    assert "l'è" not in result
    assert audit["unsupported_lm_contractions"] == ["d'è", "l'casa", "l'inesistente"]
    assert audit["added_words"] == ["c'è", "l'ho"]


@pytest.mark.parametrize("number,word", [(0,"zero"),(2,"due"),(6,"sei"),(21,"ventuno"),
    (23,"ventitré"),(28,"ventotto"),(108,"centotto"),(180,"centottanta"),
    (183,"centottantatré"),(1000,"mille"),(2000,"duemila"),(2026,"duemilaventisei"),
    (23000,"ventitremila"),(999999,"novecentonovantanovemilanovecentonovantanove")])
def test_italian_cardinals(number, word):
    assert integer_italian(number) == word


def test_spoken_form_does_not_guess_dates_decimals_identifiers_or_contractions():
    result = spoken_form("L’ho visto 2 volte, voto 6. Poi 2,5 1/2 -3 02 A2 2a 2026-10-08.")
    assert [t["replacement"] for t in result["transformations"]] == ["due", "sei"]
    assert result["text"].startswith("l'ho visto due volte voto sei")
    assert result["unsupported_numbers"]


def test_strict_metrics_unchanged_and_number_effect_is_separate():
    ref = "l'ho incontrato oggi dopo 2 anni"
    metrics, _ = evaluate("lo ha incontrato dogi dopo due anni", {"text": ref, "normalized": ref},
                          {"lo", "oggi", "due"}, {"lo", "due"})
    assert metrics["wer"]["errors"] == 4
    assert metrics["cer"]["errors"] == 7
    assert metrics["spoken_form"]["wer"]["errors"] == 3
    assert metrics["spoken_form"]["cer"]["errors"] == 4
    causes = {r["word"]: r["cause"] for r in metrics["oov_causes"]}
    assert causes["oggi"] == "outside_decoding_vocabulary"
    assert causes["l'ho"] == "absent_from_supported_lexicon"
    assert metrics["spoken_form"]["reference"]["transformations"][0]["source"] == "2"


def test_graph_identity_separates_adaptation_settings(tmp_path, monkeypatch):
    from copy import deepcopy
    from sicilian_asr import preparation
    sha = "a" * 64
    manifest = {"resources": {"lm": {"path": "lm.bin", "revision": "fixed"}}}
    config = {"resources": {"ita": {"lm_resource": "lm", "arpa_path": "model.arpa"}},
              "decoder": {"vocabulary_size": 20, "ctc_topology": "compact", "lm_scale": 1.0}}
    (tmp_path / "lm.bin.members.json").write_text(json.dumps({"archive_sha256": sha, "artifacts": {"model.arpa": {}}}))
    monkeypatch.setattr(preparation, "required_resources", lambda _: ["lm"])
    monkeypatch.setattr(preparation, "verify_file", lambda *args: {"sha256": sha})
    keys = set()
    for pronunciation, contractions in [(False, False), (True, False), (False, True), (True, True)]:
        changed = deepcopy(config)
        changed["decoder"].update(pronunciation_variants=pronunciation, contractions=contractions)
        key, payload = preparation.identity(tmp_path, changed, manifest)
        assert payload["settings"]["pronunciation_variants"] == pronunciation
        assert "lexicon/prepare.py" in payload["compiler_sources"]
        keys.add(key)
    assert len(keys) == 4


def test_benchmark_aggregation_uses_total_edits_not_mean_of_rates():
    import importlib.util
    path = Path(__file__).resolve().parents[1] / "scripts/benchmark_italian.py"
    spec = importlib.util.spec_from_file_location("benchmark_italian", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    rows = []
    for reference, hypothesis in [("uno", "due"), ("uno due tre", "uno due tre")]:
        metrics, _ = evaluate(hypothesis, {"normalized": reference}, {"uno", "due", "tre"}, {"uno", "due", "tre"})
        rows.append({"metrics": metrics, "actor": "speaker", "sentence": reference})
    result = module.summarize(rows)
    assert result["aggregate"]["strict"]["wer"]["rate"] == .25
    assert result["by_actor"]["speaker"]["recordings"] == 2
    assert len(result["by_sentence"]) == 2
