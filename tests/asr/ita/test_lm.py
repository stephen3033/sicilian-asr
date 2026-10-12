from asr.ita.lm import select_vocabulary
from asr.wfst.arpa import entries, restrict_arpa
from tests.support import ROOT


def test_vocabulary_selection_and_unchanged_arpa_weights(tmp_path):
    arpa = ROOT / "tests/fixtures/tiny.arpa"
    words, _ = select_vocabulary(
        arpa, {"uno": [], "una": [], "due": [], "assente": []}, 2
    )
    assert words == ["due", "uno"]
    restricted = tmp_path / "filtered.arpa"
    report = restrict_arpa(arpa, restricted, words)
    old = {(n, w): line for n, w, p, line in entries(arpa)}
    for n, w, _, line in entries(restricted):
        assert old[n, w] == line
    assert report["retained_ngrams"][1] == 5
