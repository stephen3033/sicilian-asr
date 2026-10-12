import hashlib
import io

import pytest

from asr.ita.lexicon import adapt, convert, lexicon_statistics
from common.errors import ResourceError
from ipa.inventory import inventory
from tests.support import ROOT


class Response(io.BytesIO):
    status = 200

    def __init__(self, body: bytes, content_length=None):
        super().__init__(body)
        self.headers = {
            "Content-Length": str(
                len(body) if content_length is None else content_length
            )
        }

    def geturl(self):
        return "https://example.org/artifact"


def specification(body=b"upstream bytes\n"):
    return {
        "status": "available",
        "path": "resources/test.data",
        "revision": "a" * 40,
        "url": "https://example.org/artifact",
        "sha256": hashlib.sha256(body).hexdigest(),
    }


def test_lexicon_statistics_preserves_duplicate_rows(tmp_path):
    path = tmp_path / "test.dict"
    # Artificial unit-test fixture only, never installed as a linguistic resource.
    path.write_text("test\tt e s t\ntest\tt e s t\ntest\tt ɛ s t\n", encoding="utf-8")
    assert lexicon_statistics(path) == {
        "rows": 3,
        "vocabulary": 1,
        "distinct_word_pronunciation_pairs": 2,
    }


def test_malformed_lexicon_is_rejected(tmp_path):
    path = tmp_path / "test.dict"
    path.write_text("not a TSV row\n")
    with pytest.raises(ResourceError, match="Malformed"):
        lexicon_statistics(path)


def test_pronunciation_alternatives_preserve_length_and_source():
    tokens = list(dict.fromkeys("ɔdʒːianŋktrgof"))
    encode = lambda s: [tokens.index(c) for c in s]
    lex = {
        "oggi": [encode("ɔdʒːi")],
        "anni": [encode("anːi")],
        "incontro": [encode("iŋkontro")],
        "ano": [encode("ano")],
        "lungo": [encode("aŋgo")],
    }
    result, audit = adapt(
        lex, tokens, set(), pronunciation_variants=True, contractions=False
    )
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
    result, audit = adapt(
        lex, tokens, lm, pronunciation_variants=False, contractions=True
    )
    assert result["l'ho"] == [encode("lɔ")]
    assert result["c'è"] == [encode("tʃɛ")]
    assert "l'è" not in result
    assert audit["unsupported_lm_contractions"] == ["d'è", "l'casa", "l'inesistente"]
    assert audit["added_words"] == ["c'è", "l'ho"]


def test_lexicon_alternatives_exclusions_and_unicode(tmp_path):
    token_path = ROOT / ".cache/models/zipa-cr-ns-large-800k/tokens.txt"
    model_path = token_path.with_name("unigram_127.model")
    if not model_path.exists():
        pytest.skip("Pinned tokenizer not acquired")
    tokens, tokenizer = inventory(token_path, model_path)
    path = tmp_path / "lexicon.dict"
    path.write_text(
        "caffè\tk a fː ɛ\ncaffe\u0300\tk a fː ɛ\ncaffè\tk a fː e\ngatto\tɡ a tː o\ntest\ttz a\naltro\tbv a\n"
    )
    lex, audit = convert(path, tokens, tokenizer)
    assert len(lex["caffè"]) == 2
    assert audit["duplicate_normalized_pairs_removed"] == 1
    assert audit["excluded_rows"] == 2
    assert audit["normalization_collisions"]["caffè"] == ["caffe\u0300", "caffè"]
    assert "".join(tokens[i] for i in lex["gatto"][0]) == "gatːo"
    assert all(3 not in p for pronunciations in lex.values() for p in pronunciations)
