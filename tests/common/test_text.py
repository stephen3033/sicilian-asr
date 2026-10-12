from common.text import normalize


def test_normalization_preserves_digits_accents_apostrophes_unknown():
    assert (
        normalize("  È L’ANNO 2026: caffe\u0300, l'acqua! <unk> ")
        == "è l'anno 2026 caffè l'acqua <unk>"
    )
    assert normalize("'ciao'—due-tre") == "ciao due tre"
