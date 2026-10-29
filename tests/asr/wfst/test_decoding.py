"""Synthetic distributions and a controlled trigram: no checkpoint or real speech."""

import itertools
import math
import shutil

import k2
import kenlm
import pytest
import torch

from asr.wfst.compact import compact_frames, compact_topology, decode
from asr.wfst.graph import compile_graph, lexicon_fst
from common.errors import ResourceError
from ipa.types import AcousticOutput
from tests.support import ROOT

TOKENS = ["<blk>", "<sos/eos>", "<unk>", "▁", "a", "b"]


SETTINGS = {
    "lm_scale": 1.0,
    "acoustic_scale": 1.0,
    "search_beam": 20.0,
    "output_beam": 8.0,
    "min_active_states": 0,
    "max_active_states": 10000,
}


def acoustic(ids):
    p = torch.full((1, len(ids), len(TOKENS)), -80.0)
    p[0, range(len(ids)), ids] = 0
    return AcousticOutput(TOKENS, 0, {}, p.log_softmax(-1), torch.tensor([len(ids)]))


def load(path):
    return k2.Fsa.from_dict(torch.load(path, weights_only=False))


def test_compressed_compilation_preserves_graph_scores_and_decoding(tmp_path):
    from asr.wfst.cache import load_graph
    from asr.wfst.storage import load_tensor, tensor_path
    from common.utils import fingerprints

    directories = [tmp_path / "plain", tmp_path / "compressed"]
    results = []
    for directory, compressed in zip(directories, (False, True)):
        directory.mkdir()
        shutil.copyfile(
            ROOT / "tests/fixtures/tiny.arpa", directory / "restricted.arpa"
        )
        report = compile_graph(
            directory,
            {"<unk>": [[2]], "uno": [[4]], "una": [[4]], "due": [[5]]},
            TOKENS,
            {**SETTINGS, "lm_order": 3, "compress_artifacts": compressed},
        )
        assert report["status"] == "complete"
        assert all(
            fingerprints(directory / name)["sha256"] == value
            for name, value in report["artifacts"].items()
        )
        graph, words = load_graph(directory)
        results.append(decode(graph, acoustic([3, 4, 0, 5]), words, SETTINGS))
    assert results[0]["text"] == results[1]["text"]
    assert results[0]["score"] == results[1]["score"]
    for stage in ("T", "L", "G", "LG", "TLG"):
        values = [
            load_tensor(tensor_path(directory, stage)) for directory in directories
        ]
        graphs = [k2.Fsa.from_dict(value) for value in values]
        assert torch.equal(graphs[0].arcs.values(), graphs[1].arcs.values())
        assert torch.equal(graphs[0].scores, graphs[1].scores)
        assert graphs[0].aux_labels.tolist() == graphs[1].aux_labels.tolist()


@pytest.fixture(params=[2, 3])
def compiled(tmp_path, request):
    shutil.copyfile(ROOT / "tests/fixtures/tiny.arpa", tmp_path / "restricted.arpa")
    if request.param == 2:
        arpa = tmp_path / "restricted.arpa"
        text = arpa.read_text().split("\\3-grams:")[0].replace("ngram 3=2\n", "")
        unigrams, bigrams = text.split("\\2-grams:")
        # Highest-order entries do not carry backoff weights in a bigram LM.
        bigrams = "\n".join(
            "\t".join(row.split("\t")[:2]) for row in bigrams.splitlines()
        )
        arpa.write_text(unigrams + "\\2-grams:" + bigrams + "\n\\end\\\n")
    lexicon = {"<unk>": [[2]], "uno": [[4], [5, 4]], "una": [[4]], "due": [[5]]}
    report = compile_graph(
        tmp_path, lexicon, TOKENS, {**SETTINGS, "lm_order": request.param}
    )
    words = {
        int(i): w
        for w, i in (
            line.split() for line in (tmp_path / "words.txt").read_text().splitlines()
        )
    }
    return tmp_path, load(tmp_path / "TLG.pt"), words, report


def outputs_for_exact_frames(graph, original_ids):
    """Enumerate published Compact CTC paths with original frames fixed.

    Neutral frames permit all labels at weight zero. The hub cannot be reached
    on a real frame, so these never hallucinate a new emitted token.
    """
    arcs = graph.arcs.values().tolist()
    aux = graph.aux_labels.tolist()
    paths = {(0, ())}
    for frame in [v for i in original_ids for v in (i, None)]:
        following = set()
        for state, labels in paths:
            for (source, dest, label, _), out in zip(arcs, aux):
                if source == state and label >= 0 and (frame is None or label == frame):
                    following.add((dest, labels + ((out,) if out > 0 else ())))
        paths = following
    accepting = {arc[0] for arc in arcs if arc[2] == -1}
    return {labels for state, labels in paths if state in accepting}


def test_published_compact_ctc_semantics_and_termination():
    topology = compact_topology(len(TOKENS))
    assert outputs_for_exact_frames(topology, [4]) == {(4,)}
    assert outputs_for_exact_frames(topology, [4, 4]) == {(4,), (4, 4)}
    assert outputs_for_exact_frames(topology, [4, 0, 4]) == {(4, 4)}
    assert outputs_for_exact_frames(topology, [0, 0]) == {()}
    assert outputs_for_exact_frames(topology, [3, 4, 5]) == {(3, 4, 5)}
    assert outputs_for_exact_frames(topology, [4, 5]) == {(4, 5)}
    # Final arcs originate only at the hub: last neutral frame returns there.
    assert {row[0] for row in topology.arcs.values().tolist() if row[2] == -1} == {0}


def test_auxiliary_frames_preserve_originals_and_lengths():
    output = acoustic([4, 5, 0])
    output.lengths[0] = 2
    original = output.log_probs.clone()
    padded, lengths = compact_frames(output)
    assert torch.equal(output.log_probs, original)
    assert torch.equal(padded[:, ::2, :-1], original)
    assert torch.isneginf(padded[:, ::2, -1]).all()
    assert (padded[:, 1::2] == 0).all()
    assert lengths.tolist() == [4]


def test_full_graph_homophones_alternatives_unknown_and_prefix(compiled):
    path, graph, words, report = compiled
    assert report["status"] == "complete"
    assert decode(graph, acoustic([4]), words, SETTINGS)["text"] == "uno"
    assert decode(graph, acoustic([5, 4]), words, SETTINGS)["text"] == "uno"
    assert decode(graph, acoustic([3, 4, 0, 5]), words, SETTINGS)["text"] == "uno due"
    assert decode(graph, acoustic([2]), words, SETTINGS)["text"] == "<unk>"
    # Finite probabilities permit substitutions, so check forbidden exact paths
    # by symbolic intersection rather than expecting every noisy input to fail.
    for sequence in ([4, 6, 3, 6, 5, 6], [1, 6]):
        path_fsa = k2.linear_fsa(sequence)
        intersection = k2.connect(
            k2.intersect(graph, path_fsa, treat_epsilons_specially=False)
        )
        assert intersection.num_arcs == 0
    impossible = k2.linear_fsa([4, 6, 4, 6])
    impossible.aux_labels = impossible.labels.clone()
    with pytest.raises(ResourceError, match="lattice"):
        decode(impossible, acoustic([4]), words, SETTINGS)
    # Every intermediate graph survives serialization with scores and labels intact.
    for name in ["T", "L", "G", "LG", "TLG"]:
        fsa = load(path / f"{name}.pt")
        assert fsa.num_arcs == report["stages"][name]["arcs"]


def test_decoder_uses_full_distribution_and_lm_scale(compiled):
    _, graph, words, _ = compiled
    output = acoustic([5])
    output.log_probs[0, 0, 4] = -0.5
    output.log_probs = output.log_probs.log_softmax(-1)
    assert output.log_probs.argmax(-1).item() == 5
    # Same distribution, stronger language preference for uno vs due.
    graph.scores = graph.scores * 0.1
    assert decode(graph, output, words, SETTINGS)["text"] == "due"
    graph.scores = graph.scores * 20
    assert decode(graph, output, words, SETTINGS)["text"] == "uno"


@pytest.mark.parametrize("word_bonus", [-1.5, 1.5])
def test_word_bonus_scores_words_without_rewarding_backoff_or_boundaries(
    tmp_path, word_bonus
):
    shutil.copyfile(ROOT / "tests/fixtures/tiny.arpa", tmp_path / "restricted.arpa")
    lexicon = {"<unk>": [[2]], "uno": [[4]], "una": [[4]], "due": [[5]]}
    settings = {**SETTINGS, "lm_order": 3, "lm_scale": 0.5, "word_bonus": word_bonus}
    compile_graph(tmp_path, lexicon, TOKENS, settings)
    graph = load(tmp_path / "TLG.pt")
    words = {
        int(i): w
        for w, i in (
            line.split() for line in (tmp_path / "words.txt").read_text().splitlines()
        )
    }
    lm = kenlm.Model(str(tmp_path / "restricted.arpa"))
    # Includes a known bigram, a backoff sequence, and the unknown-word route.
    for phones in ([4], [4, 0, 5], [5, 0, 5], [2]):
        result = decode(graph, acoustic(phones), words, settings)
        expected = 0.5 * lm.score(result["text"]) * math.log(10)
        expected += word_bonus * len(result["text"].split())
        assert result["grammar_score"] == pytest.approx(expected, abs=3e-5)


@pytest.mark.parametrize("seed_cost,alternate_cost", [(0.0, 0.5), (1.5, 0.0)])
def test_penalized_phone_alternatives_preserve_words_and_new_homophones(
    tmp_path, seed_cost, alternate_cost
):
    shutil.copyfile(ROOT / "tests/fixtures/tiny.arpa", tmp_path / "restricted.arpa")
    lexicon = {"<unk>": [[2]], "uno": [[4]], "una": [[4]], "due": [[5]]}
    settings = {
        **SETTINGS,
        "lm_order": 3,
        "phone_alternatives": {"4": [{"token": 5, "cost": alternate_cost}]},
        "phone_seed_costs": {"4": seed_cost},
    }
    compile_graph(tmp_path, lexicon, TOKENS, settings)
    graph = load(tmp_path / "TLG.pt")
    words = {
        int(i): w
        for w, i in (
            line.split() for line in (tmp_path / "words.txt").read_text().splitlines()
        )
    }
    lm = kenlm.Model(str(tmp_path / "restricted.arpa"))
    result = decode(graph, acoustic([5]), words, settings)
    assert result["text"] == "uno"
    assert result["grammar_score"] == pytest.approx(
        lm.score("uno") * math.log(10) - alternate_cost, abs=3e-5
    )
    canonical = decode(graph, acoustic([4]), words, settings)
    assert canonical["text"] == "uno"
    assert canonical["grammar_score"] == pytest.approx(
        lm.score("uno") * math.log(10) - seed_cost, abs=3e-5
    )
    pair = decode(graph, acoustic([5, 0, 5]), words, settings)
    assert pair["text"] == "uno due"
    assert pair["grammar_score"] == pytest.approx(
        lm.score("uno due") * math.log(10) - alternate_cost, abs=3e-5
    )


def test_position_costs_distinguish_word_ending_before_disambiguation(tmp_path):
    shutil.copyfile(ROOT / "tests/fixtures/tiny.arpa", tmp_path / "restricted.arpa")
    lexicon = {"<unk>": [[2]], "uno": [[4, 4]], "una": [[4, 4]], "due": [[5]]}
    settings = {
        **SETTINGS,
        "lm_order": 3,
        "phone_alternatives": {"4": [{"token": 5, "cost": 7.0}]},
        "phone_position_costs": {
            "final": {"4": [{"token": 4, "cost": 1.0}, {"token": 5, "cost": 0.0}]},
            "nonfinal": {"4": [{"token": 4, "cost": 0.0}, {"token": 5, "cost": 2.0}]},
        },
    }
    compile_graph(tmp_path, lexicon, TOKENS, settings)
    graph = load(tmp_path / "TLG.pt")
    words = {
        int(i): w
        for w, i in (
            line.split() for line in (tmp_path / "words.txt").read_text().splitlines()
        )
    }
    lm = kenlm.Model(str(tmp_path / "restricted.arpa"))
    for phones, cost in [([4, 0, 5], 0), ([4, 0, 4], 1), ([5, 0, 4], 3)]:
        result = decode(graph, acoustic(phones), words, settings)
        assert result["text"] == "uno"
        assert result["grammar_score"] == pytest.approx(
            lm.score("uno") * math.log(10) - cost, abs=3e-5
        )


def test_optional_label_preserves_word_scores_and_creates_new_homophones(tmp_path):
    shutil.copyfile(ROOT / "tests/fixtures/tiny.arpa", tmp_path / "restricted.arpa")
    lexicon = {"<unk>": [[2]], "uno": [[4, 5]], "una": [[4]], "due": [[5, 4]]}
    settings = {**SETTINGS, "lm_order": 3, "phone_optional_costs": {"5": 0.5}}
    compile_graph(tmp_path, lexicon, TOKENS, settings)
    graph = load(tmp_path / "TLG.pt")
    words = {
        int(i): w
        for w, i in (
            line.split() for line in (tmp_path / "words.txt").read_text().splitlines()
        )
    }
    lm = kenlm.Model(str(tmp_path / "restricted.arpa"))
    for phones, expected, cost in [
        ([4], "uno", 0.5),
        ([4, 5], "uno", 0),
        ([5, 4], "due", 0),
        ([4, 0, 4], "uno due", 1.0),
    ]:
        result = decode(graph, acoustic(phones), words, settings)
        assert result["text"] == expected
        assert result["grammar_score"] == pytest.approx(
            lm.score(expected) * math.log(10) - cost, abs=3e-5
        )


def test_optional_labels_cannot_form_epsilon_word_cycles():
    with pytest.raises(ResourceError, match="nonoptional acoustic label"):
        lexicon_fst(
            {"word": [[4, 5]]},
            {"word": 1, "#0": 2},
            len(TOKENS),
            phone_optional_costs={"4": 0.5, "5": 1.0},
        )


def test_word_context_costs_change_only_the_named_written_word(tmp_path):
    shutil.copyfile(ROOT / "tests/fixtures/tiny.arpa", tmp_path / "restricted.arpa")
    lexicon = {"<unk>": [[2]], "uno": [[4]], "una": [[4]], "due": [[5]]}
    settings = {
        **SETTINGS,
        "lm_order": 3,
        "phone_context_costs": {
            "una": {"1:0:4": [{"token": 4, "cost": 20}, {"token": 5, "cost": 0}]}
        },
    }
    compile_graph(tmp_path, lexicon, TOKENS, settings)
    words = {
        int(i): w
        for w, i in (
            line.split() for line in (tmp_path / "words.txt").read_text().splitlines()
        )
    }
    graph = load(tmp_path / "TLG.pt")
    assert decode(graph, acoustic([4]), words, SETTINGS)["text"] == "uno"
    assert torch.isfinite(graph.scores).all()
    with pytest.raises(ResourceError, match="does not match"):
        lexicon_fst(
            lexicon,
            {"<unk>": 1, "uno": 2, "una": 3, "due": 4, "#0": 5},
            len(TOKENS),
            phone_context_costs={"una": {"2:1:4": [{"token": 4, "cost": 0}]}},
        )


def test_exact_observed_paths_bypass_seed_mappings_and_pay_prior_once(tmp_path):
    shutil.copyfile(ROOT / "tests/fixtures/tiny.arpa", tmp_path / "restricted.arpa")
    lexicon = {"<unk>": [[2]], "uno": [[4, 4]], "una": [[4, 4]], "due": [[5]]}
    settings = {
        **SETTINGS,
        "lm_order": 3,
        "phone_alternatives": {"4": [{"token": 5, "cost": 0}]},
        "phone_seed_costs": {"4": 2.0},
        "phone_position_costs": {"final": {"4": [{"token": 4, "cost": 3.0}]}},
        "phone_optional_costs": {"4": 0},
        "pronunciation_settings": {
            "uno": [{"cost": 0.7, "exact": True}],
            "una": [{"cost": 0.9, "exact": True}],
        },
    }
    compile_graph(tmp_path, lexicon, TOKENS, settings)
    graph = load(tmp_path / "TLG.pt")
    words = {
        int(i): w
        for w, i in (
            line.split() for line in (tmp_path / "words.txt").read_text().splitlines()
        )
    }
    lm = kenlm.Model(str(tmp_path / "restricted.arpa"))
    result = decode(graph, acoustic([4, 0, 4]), words, settings)
    assert result["text"] == "uno"
    assert result["grammar_score"] == pytest.approx(
        lm.score("uno") * math.log(10) - 0.7, abs=3e-5
    )
    # Neither substitution nor omission applies to empirical exact-label paths.
    assert decode(graph, acoustic([5, 0, 5]), words, settings)["text"] == "due due"


def test_diagnostics_explain_scores_and_preserve_decoding(compiled):
    path, graph, words, _ = compiled
    output = acoustic([3, 4, 0, 5])
    original = output.log_probs.clone()
    baseline = decode(graph, output, words, SETTINGS)
    result = decode(graph, output, words, {**SETTINGS, "diagnostic_paths": 16})
    assert result["text"] == baseline["text"]
    assert result["score"] == pytest.approx(
        result["acoustic_score"] + result["grammar_score"], abs=1e-5
    )
    lm = kenlm.Model(str(path / "restricted.arpa"))
    assert result["grammar_score"] == pytest.approx(
        lm.score("uno due") * math.log(10), abs=2e-5
    )
    assert result["acoustic_score"] == pytest.approx(0, abs=1e-5)
    assert result["pronunciation"] == "▁ab"
    assert result["diagnostics"]["candidates"][0]["text"] == result["text"]
    assert torch.equal(output.log_probs, original)
    # A retained graph vector avoids a multi-GB copy for every recording.
    assert (
        decode(k2.create_fsa_vec([graph]), output, words, SETTINGS)["text"]
        == result["text"]
    )


def test_arpa_scores_backoff_and_boundaries_match_kenlm(compiled):
    path, _, words, _ = compiled
    model = kenlm.Model(str(path / "restricted.arpa"))
    G = load(path / "G.pt")
    identifiers = {w: i for i, w in words.items()}
    labels = G.labels.clone()
    labels[labels == identifiers["#0"]] = 0
    G.labels = labels
    G = k2.arc_sort(k2.remove_epsilon(G))
    # Exhaust all short sequences, including unseen bigrams and unknown paths.
    for length in range(3):
        for sequence in itertools.product(
            ["uno", "una", "due", "<unk>"], repeat=length
        ):
            # Independent tropical path scoring also handles the empty sentence
            # without the pinned k2 singleton-arc composition stride bug.
            current = {0: 0.0}
            for label in [identifiers[w] for w in sequence] + [-1]:
                following = {}
                for arc, weight in zip(G.arcs.values().tolist(), G.scores.tolist()):
                    source, dest, symbol, _ = arc
                    if source in current and symbol == label:
                        following[dest] = max(
                            following.get(dest, -math.inf), current[source] + weight
                        )
                current = following
            score = max(current.values(), default=-math.inf)
            assert score == pytest.approx(
                model.score(" ".join(sequence), bos=True, eos=True) * math.log(10),
                abs=2e-5,
            )
