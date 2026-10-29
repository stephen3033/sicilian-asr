import shutil

import k2
import pytest

from asr.wfst.cache import load_graph
from asr.wfst.compact import decode
from asr.wfst.diagnostics import (
    GrammarScorer,
    SequenceScorer,
    retained_lattice,
    sequences_in_lattice,
    sequences_in_lattice_product,
)
from asr.wfst.graph import compile_graph
from common.errors import ResourceError
from tests.asr.wfst.test_decoding import SETTINGS, TOKENS, acoustic
from tests.support import ROOT


def test_factor_scores_match_compilation_with_pronunciation_costs_and_bonus(tmp_path):
    shutil.copyfile(ROOT / "tests/fixtures/tiny.arpa", tmp_path / "restricted.arpa")
    lexicon = {"<unk>": [[2]], "uno": [[4]], "una": [[4]], "due": [[5]]}
    settings = {
        **SETTINGS,
        "lm_order": 3,
        "word_bonus": 0.2,
        "pronunciation_settings": {"uno": [{"cost": 0.7, "exact": True}]},
    }
    compile_graph(tmp_path, lexicon, TOKENS, settings)
    graph, words = load_graph(tmp_path)
    output = acoustic([3, 4, 0, 5])
    decoded = decode(graph, output, words, settings)
    scorer = SequenceScorer(lexicon, TOKENS, settings, tmp_path / "restricted.arpa")
    components = scorer.decompose(decoded["text"].split(), decoded)
    assert components["verified"]
    assert components["pronunciation_score"] == pytest.approx(-0.7)
    assert components["word_bonus_score"] == pytest.approx(0.4)
    exact = scorer.score(decoded["text"].split(), output)
    assert exact["score"] == pytest.approx(decoded["score"], abs=0.001)
    lattice = retained_lattice(graph, output, settings)
    candidates = [["uno", "due"], ["una", "due"], ["due", "uno"]]
    membership = sequences_in_lattice(lattice, candidates, scorer.word_ids)
    assert membership == sequences_in_lattice_product(
        lattice, candidates, scorer.word_ids
    )
    assert membership[0]["present"]
    # Epsilon removal can merge a dominated homophone/backoff path.
    # Absence alone is not evidence of beam-search failure.
    assert not membership[1]["present"]
    assert not membership[2]["present"]
    alternatives = sequences_in_lattice(
        retained_lattice(graph, acoustic([4]), {**settings, "output_beam": 30}),
        [["uno"], ["una"]],
        scorer.word_ids,
    )
    assert all(row["present"] for row in alternatives)
    assert alternatives[0]["score"] > alternatives[1]["score"]
    assert scorer.score(candidates[0], output, pronunciation_scale=0)[
        "acoustic_score"
    ] == pytest.approx(
        scorer.score(candidates[1], output, pronunciation_scale=0)["acoustic_score"]
    )
    with pytest.raises(ResourceError, match="lack lexical support"):
        scorer.score(["missing"], output)


def test_blank_only_counterfactual_and_backoff_include_eos(tmp_path):
    lexicon = {"<unk>": [[2]], "uno": [[4]], "una": [[4]], "due": [[5]]}
    scorer = SequenceScorer(
        lexicon, TOKENS, SETTINGS, ROOT / "tests/fixtures/tiny.arpa"
    )
    empty = scorer.score([], acoustic([0, 0]))
    assert empty["text"] == "" and len(empty["lm_terms"]) == 1
    assert empty["lm_terms"][0]["word"] == "</s>"
    score = scorer.score(["una", "due", "uno"], acoustic([4, 0, 5, 0, 4]))
    assert len(score["lm_terms"]) == 4
    assert score["lm_terms"][-1]["word"] == "</s>"


def test_compiled_grammar_uses_openfst_start_and_tropical_backoff(tmp_path):
    path = tmp_path / "G.fst.txt"
    path.write_text("\n5 6 4 4 3\n5 7 9 0 0.1\n7 6 4 4 1\n6 0.2\n")
    scorer = GrammarScorer(path, {"uno": 4, "#0": 9})
    assert scorer.start == 5
    assert scorer.score(["uno"]) == pytest.approx(-1.3)


def test_exact_compiled_constraint_proves_a_synthetic_beam_search_failure():
    from pipeline.scn_residual_search import constrained_compiled_score

    scorer = SequenceScorer(
        {"<unk>": [[2]], "uno": [[4]], "una": [[4]], "due": [[5]]},
        TOKENS,
        SETTINGS,
        ROOT / "tests/fixtures/tiny.arpa",
    )
    graph = k2.Fsa.from_str(
        "0 1 4 4 0\n0 2 5 2 -5\n1 3 6 0 0\n2 4 6 0 0\n3 5 4 0 0\n4 6 5 0 0\n5 7 6 0 0\n6 7 6 0 0\n7 8 -1 -1 0\n8",
        acceptor=False,
    )
    graph = k2.create_fsa_vec([k2.arc_sort(graph)])
    output = acoustic([4, 5])
    output.log_probs[0, 0, 5] = -1.5
    output.log_probs[0, 1, 4] = -20
    output.log_probs = output.log_probs.log_softmax(-1)
    narrow = decode(graph, output, scorer.words, {**SETTINGS, "search_beam": 1})
    assert narrow["text"] == "uno"
    corrected = constrained_compiled_score(graph, output, ["due"], scorer)
    assert corrected["present"] and corrected["score"] > narrow["score"] + 10


def test_native_word_trie_matches_independent_products_with_empty_and_duplicate_paths():
    lattice = k2.Fsa.from_str(
        "0 1 0 0 -0.3\n0 2 4 1 -1\n1 2 4 1 -0.2\n1 4 -1 -1 -4\n2 3 5 2 -0.1\n2 4 -1 -1 -0.7\n3 4 -1 -1 0\n4",
        acceptor=False,
    )
    vec = k2.create_fsa_vec([lattice])
    sequences = [[], ["a"], ["a", "b"], ["a"], ["b"], ["a", "b", "a"]]
    native = sequences_in_lattice(vec, sequences, {"a": 1, "b": 2})
    product = sequences_in_lattice_product(vec, sequences, {"a": 1, "b": 2})
    for a, b in zip(native, product):
        assert a["present"] == b["present"]
        assert (
            a["score"] == pytest.approx(b["score"])
            if a["present"]
            else a["score"] is None
        )
