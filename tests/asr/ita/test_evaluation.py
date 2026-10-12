from asr.ita.evaluation import evaluate


def test_strict_metrics_unchanged_and_number_effect_is_separate():
    ref = "l'ho incontrato oggi dopo 2 anni"
    metrics, _ = evaluate(
        "lo ha incontrato dogi dopo due anni",
        {"text": ref, "normalized": ref},
        {"lo", "oggi", "due"},
        {"lo", "due"},
    )
    assert metrics["wer"]["errors"] == 4
    assert metrics["cer"]["errors"] == 7
    assert metrics["spoken_form"]["wer"]["errors"] == 3
    assert metrics["spoken_form"]["cer"]["errors"] == 4
    causes = {r["word"]: r["cause"] for r in metrics["oov_causes"]}
    assert causes["oggi"] == "outside_decoding_vocabulary"
    assert causes["l'ho"] == "absent_from_supported_lexicon"
    assert metrics["spoken_form"]["reference"]["transformations"][0]["source"] == "2"
