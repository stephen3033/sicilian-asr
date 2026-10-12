import pytest

from asr.ita.spoken import integer_italian, spoken_form


@pytest.mark.parametrize(
    "number,word",
    [
        (0, "zero"),
        (2, "due"),
        (6, "sei"),
        (21, "ventuno"),
        (23, "ventitré"),
        (28, "ventotto"),
        (108, "centotto"),
        (180, "centottanta"),
        (183, "centottantatré"),
        (1000, "mille"),
        (2000, "duemila"),
        (2026, "duemilaventisei"),
        (23000, "ventitremila"),
        (999999, "novecentonovantanovemilanovecentonovantanove"),
    ],
)
def test_italian_cardinals(number, word):
    assert integer_italian(number) == word


def test_spoken_form_does_not_guess_dates_decimals_identifiers_or_contractions():
    result = spoken_form(
        "L’ho visto 2 volte, voto 6. Poi 2,5 1/2 -3 02 A2 2a 2026-10-08."
    )
    assert [t["replacement"] for t in result["transformations"]] == ["due", "sei"]
    assert result["text"].startswith("l'ho visto due volte voto sei")
    assert result["unsupported_numbers"]
