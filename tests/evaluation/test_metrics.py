from evaluation.metrics import edits, evaluate


def test_alignment_counts_and_empty_reference():
    counts, _ = edits(["a", "b", "c"], ["a", "d", "c", "e"])
    assert (
        counts["substitutions"],
        counts["insertions"],
        counts["deletions"],
        counts["reference_units"],
    ) == (1, 1, 0, 3)
    assert edits(["a", "b"], ["a"])[0]["deletions"] == 1
    assert edits([], ["a"])[0]["rate"] is None
    metrics, _ = evaluate("a <unk>", {"normalized": "a b 2"}, {"a", "b"}, {"a"})
    assert metrics["oov_full_supported_lexicon"]["tokens"] == 1
    assert metrics["oov_decoding_vocabulary"]["tokens"] == 2
    assert metrics["wer"]["hypothesis_units"] == 2
    assert metrics["cer"]["reference_units"] == 3
