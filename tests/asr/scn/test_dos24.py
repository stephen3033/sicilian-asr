import pytest

from asr.scn.dataset import passage_hashes
from asr.scn.dos24 import Converter
from asr.scn.dos24_corpus import DuplicateFilter, leakage, sentence_spans


def lexical(*words, pos="NOUN"):
    return {
        w: [
            {
                "lemma": w,
                "pos": pos,
                "features": {},
                "source": "lexical-fixture",
                "evidence": w,
            }
        ]
        for w in words
    }


@pytest.mark.parametrize(
    "source,target",
    [
        ("cavaddu", "cavaḍḍu"),
        ("cavaddru", "cavaḍḍu"),
        ("friddu", "friddu"),
        ("addisijari", "addisijari"),
        ("chiu", "cchiù"),
        ("'nzirtari", "nzirtari"),
        ("magaria", "magarìa"),
        ("aju avi annu", "aju avi annu"),
        ("àju àvi ànnu", "àju àvi ànnu"),
        ("gatta jatta pagari pajari pavari", "gatta jatta pagari pajari pavari"),
        ("‘n", "'n"),
        ("talìu", "talìu"),
        ("taliari passijari", "taliari passijari"),
    ],
)
def test_published_examples_counterexamples_and_optional_forms(source, target):
    converter = Converter()
    result = converter.convert(source)
    assert result["converted_text"] == target
    assert not result["unresolved"]
    assert converter.convert(target)["converted_text"] == target


def test_productive_lexical_noun_suffix_is_not_a_caption_whitelist():
    converter = Converter(lexical("microorganizzazioni", "microorganizzazzioni"))
    result = converter.convert("microorganizzazioni")
    assert result["converted_text"] == "microorganizzazzioni"
    assert result["lm_eligible"]
    assert "M01" in result["edits"][0]["rules"]
    assert not Converter().convert("wordwithoutanalysis")["lm_eligible"]


def test_productive_adjective_inflection_and_explicit_stress():
    converter = Converter(lexical("novu", pos="ADJ") | lexical("pirchì", pos="ADV"))
    assert converter.convert("novi")["lm_eligible"]
    assert converter.convert("pirchí")["converted_text"] == "pirchì"


def test_no_universal_dd_ci_or_synonym_replacement():
    converter = Converter(lexical("addari"))
    result = converter.convert("addari ci e ni")
    assert result["converted_text"] == "addari ci e ni"
    assert {r["reason"] for r in result["unresolved"]} == {
        "dental_or_retroflex",
        "grammatical_function",
    }
    result = converter.convert(
        "ci e ni", grammatical={0: "locative", 3: "copula", 5: "partitive"}
    )
    assert result["converted_text"] == "cci è nni"
    assert converter.convert("ci", grammatical={0: "dative"})["converted_text"] == "ci"


def test_stress_and_hiatus_require_missing_facts():
    converter = Converter(lexical("àncora", "ancòra", "poesia"))
    assert converter.convert("ancora")["unresolved"][0]["reason"] == "stress_ambiguity"
    assert converter.convert("poesia")["unresolved"][0]["reason"] == "hiatus"


def test_resource_stress_marks_are_not_automatically_dos24_accents():
    converter = Converter(lexical("ca", "cà", "si", "sì", "fari", "fàri", "capìri", "facèmu", "pìnzanu"))
    result = converter.convert("ca si fari capiri facemu pinzanu")
    assert result["converted_text"] == "ca si fari capiri facemu pìnzanu"
    assert [e["original"] for e in result["edits"]] == ["pinzanu"]


def test_final_vowel_rule_does_not_rewrite_monosyllables_or_borrowed_unknowns():
    index = lexical("o", "u", "me", "mi", "italianu")
    index["o"][0]["lemma"] = "u"
    index["me"][0]["lemma"] = "mi"
    converter = Converter(index)
    assert converter.convert("o me")["converted_text"] == "o me"
    assert converter.convert("italiano")["converted_text"] == "italiano"
    assert not converter.convert("italiano")["lm_eligible"]


def test_span_edits_reconstruct_original_unicode_and_preserve_names_quotes():
    converter = Converter(lexical("pirchì"))
    original = "pirchi\u0301, cavaddu; José «cavaddru»."
    result = converter.convert(original)
    reconstructed = original
    for edit in reversed(result["edits"]):
        assert original[edit["start"] : edit["end"]] == edit["original"]
        assert edit["rules"]
        reconstructed = (
            reconstructed[: edit["start"]]
            + edit["replacement"]
            + reconstructed[edit["end"] :]
        )
    assert reconstructed == result["converted_text"]
    assert "José «cavaddru»" in reconstructed
    assert not result["lm_eligible"]


def test_sentence_exclusion_cannot_manufacture_adjacent_ngrams():
    text = "cani friddu. unknownword. cavaddu novu."
    converter = Converter()
    retained = [
        converter.convert(text[a:b])["converted_text"]
        for a, b in sentence_spans(text)
        if converter.convert(text[a:b])["lm_eligible"]
    ]
    assert retained == ["cani friddu.", "cavaḍḍu novu."]
    assert all("friddu cavaḍḍu" not in sentence for sentence in retained)


def test_dedup_checks_similarity_before_adding_and_leakage_after_conversion():
    filt = DuplicateFilter(0.8)
    text = "unu dui tri quattru cincu sei setti ottu novi deci"
    assert filt.check(text, "article-1") is None
    assert filt.check(text.upper(), "article-2")["representative"] == "article-1"
    assert filt.check(text + " undici", "article-3")["reason"] == "near_duplicate"
    assert leakage("cavaḍḍu novu", set(), {"cavaḍḍu novu"})
    assert not leakage("cavaddu novu", set(), {"cavaḍḍu novu"})
    assert leakage(
        " ".join(map(str, range(13))), passage_hashes(list(map(str, range(12)))), set()
    )
