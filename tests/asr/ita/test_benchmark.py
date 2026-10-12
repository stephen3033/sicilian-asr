from asr.ita.evaluation import evaluate


def test_benchmark_aggregation_uses_total_edits_not_mean_of_rates():
    from asr.ita.benchmark import summarize

    rows = []
    for reference, hypothesis in [("uno", "due"), ("uno due tre", "uno due tre")]:
        metrics, _ = evaluate(
            hypothesis,
            {"normalized": reference},
            {"uno", "due", "tre"},
            {"uno", "due", "tre"},
        )
        rows.append({"metrics": metrics, "actor": "speaker", "sentence": reference})
    result = summarize(rows)
    assert result["aggregate"]["strict"]["wer"]["rate"] == 0.25
    assert result["by_actor"]["speaker"]["recordings"] == 2
    assert len(result["by_sentence"]) == 2
