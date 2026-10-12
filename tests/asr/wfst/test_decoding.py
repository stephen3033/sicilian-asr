from tests.support import ROOT

"""Synthetic distributions and a controlled trigram: no checkpoint or real speech."""


import itertools
import math
import shutil

import k2
import kenlm
import pytest
import torch

from asr.wfst.compact import compact_frames, compact_topology, decode
from asr.wfst.graph import compile_graph
from common.errors import ResourceError
from ipa.types import AcousticOutput

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
