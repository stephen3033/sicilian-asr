"""Constrained, factor-separated WFST scores over unchanged acoustic frames.

Reference constraints are diagnostic only. They never create lexicon evidence.
"""

import math
from collections import defaultdict
from functools import lru_cache

import k2
import torch

from asr.wfst.compact import compact_frames, compact_topology
from asr.wfst.graph import lexicon_fst
from common.errors import ResourceError


def tolerance(score):
    return max(0.001, 0.00001 * abs(score))


def path_summary(path, frames, tokens, words):
    labels = path.labels.tolist()
    if len(labels) != frames.shape[1] + 1:
        raise ResourceError("Constrained path does not cover all acoustic frames")
    acoustic = sum(
        float(frames[0, i, label]) for i, label in enumerate(labels) if label >= 0
    )
    values = (
        path.aux_labels.values
        if isinstance(path.aux_labels, k2.RaggedTensor)
        else path.aux_labels
    )
    word_ids = [i for i in values.tolist() if i > 0]
    phone_ids = []
    if hasattr(path, "tokens"):
        values = (
            path.tokens.values
            if isinstance(path.tokens, k2.RaggedTensor)
            else path.tokens
        )
        phone_ids = [i for i in values.tolist() if 0 < i < len(tokens)]
    return {
        "text": " ".join(words[i] for i in word_ids),
        "acoustic_score": acoustic,
        "path_score": float(path.scores.double().sum()),
        "pronunciation_token_ids": phone_ids,
        "frame_labels": labels[:-1],
    }


class SequenceScorer:
    """Exact best acoustic/lexical path for a fixed written word sequence."""

    def __init__(self, lexicon, tokens, settings, arpa, compiled_grammar=None):
        import kenlm

        self.lexicon, self.tokens, self.settings = lexicon, tokens, settings
        self.word_ids = {
            w: i for i, w in enumerate(["<eps>", *sorted(lexicon), "<s>", "</s>", "#0"])
        }
        self.words = {i: w for w, i in self.word_ids.items()}
        self.lm = kenlm.Model(str(arpa))
        self.topology = compact_topology(len(tokens))
        self.compiled_grammar = (
            GrammarScorer(compiled_grammar, self.word_ids) if compiled_grammar else None
        )

    def grammar(self, sequence):
        rows = list(self.lm.full_scores(" ".join(sequence), bos=True, eos=True))
        if len(rows) != len(sequence) + 1:
            raise ResourceError("LM token/score count differs")
        kenlm_score = sum(r[0] for r in rows) * math.log(10)
        compiled = (
            self.compiled_grammar.score(sequence)
            if self.compiled_grammar
            else kenlm_score
        )
        return {
            "unscaled_log_probability": compiled,
            "kenlm_log_probability": kenlm_score,
            "compiled_minus_kenlm": compiled - kenlm_score,
            "terms": [
                {
                    "word": w,
                    "log_probability": r[0] * math.log(10),
                    "ngram_order": r[1],
                    "oov": r[2],
                }
                for w, r in zip([*sequence, "</s>"], rows)
            ],
        }

    def lexical_graph(self, sequence, pronunciation_scale=1.0):
        if not sequence:
            # Avoid the pinned k2 singleton-arc stride bug. The ordinary
            # lexicon permits an empty utterance with one optional prefix.
            return k2.arc_sort(
                k2.Fsa.from_str(
                    f"0 1 {self.tokens.index('▁')} 0 0\n0 2 -1 -1 0\n1 2 -1 -1 0\n2",
                    acceptor=False,
                )
            )
        unknown = set(sequence) - set(self.lexicon)
        if unknown:
            raise ResourceError(
                f"Constrained words lack lexical support: {sorted(unknown)}"
            )
        lexicon = {w: self.lexicon[w] for w in set(sequence)}
        settings = self.settings
        lexical, _ = lexicon_fst(
            lexicon,
            self.word_ids,
            len(self.tokens),
            self.tokens.index("▁"),
            settings.get("phone_alternatives"),
            settings.get("phone_seed_costs"),
            settings.get("phone_position_costs"),
            settings.get("phone_optional_costs"),
            {
                w: v
                for w, v in settings.get("pronunciation_settings", {}).items()
                if w in lexicon
            },
            {
                w: v
                for w, v in (settings.get("phone_context_costs") or {}).items()
                if w in lexicon
            },
        )
        labels = lexical.labels.clone()
        labels[labels >= len(self.tokens) + 1] = 0
        lexical.labels = labels
        lexical.scores = lexical.scores * pronunciation_scale
        constraint = k2.linear_fsa([self.word_ids[w] for w in sequence])
        lexical = k2.connect(k2.compose(k2.arc_sort(lexical), k2.arc_sort(constraint)))
        lexical = k2.arc_sort(k2.connect(k2.remove_epsilon(lexical)))
        return lexical

    def graph(self, sequence, pronunciation_scale=1.0):
        return k2.arc_sort(
            k2.connect(
                k2.compose(
                    self.topology,
                    self.lexical_graph(sequence, pronunciation_scale),
                    inner_labels="tokens",
                )
            )
        )

    def pronunciation_score(self, sequence, phone_ids):
        if not sequence and phone_ids in ([], [self.tokens.index("▁")]):
            return 0.0
        graph = self.lexical_graph(sequence)
        matched = k2.connect(k2.intersect(graph, k2.arc_sort(k2.linear_fsa(phone_ids))))
        if matched.num_arcs == 0:
            raise ResourceError(
                "Decoded pronunciation is not licensed by lexical factors"
            )
        return float(
            k2.create_fsa_vec([k2.top_sort(matched)]).get_tot_scores(
                log_semiring=False, use_double_scores=True
            )[0]
        )

    def decompose(self, sequence, decoded):
        grammar = self.grammar(sequence)
        pronunciation = self.pronunciation_score(
            sequence, decoded["pronunciation_token_ids"]
        )
        components = {
            "acoustic_score": decoded["acoustic_score"],
            "pronunciation_score": pronunciation,
            "lm_score": grammar["unscaled_log_probability"] * self.settings["lm_scale"],
            "word_bonus_score": len(sequence) * self.settings.get("word_bonus", 0.0),
        }
        total = sum(components.values())
        return {
            **components,
            "score": total,
            "compiled_score": decoded["score"],
            "residual": total - decoded["score"],
            "verified": abs(total - decoded["score"]) <= tolerance(decoded["score"]),
            "lm_terms": grammar["terms"],
            "compiled_minus_kenlm": grammar["compiled_minus_kenlm"],
        }

    def score(self, sequence, output, *, pronunciation_scale=1.0, include_frames=False):
        graph = self.graph(sequence, pronunciation_scale)
        frames, lengths = compact_frames(
            output, self.settings.get("acoustic_scale", 1.0)
        )
        dense = k2.DenseFsaVec(
            frames, torch.tensor([[0, 0, int(lengths[0])]], dtype=torch.int32)
        )
        lattice = k2.intersect_dense(
            k2.create_fsa_vec([graph]),
            dense,
            output_beam=1000000,
            max_states=15000000,
            max_arcs=100000000,
        )
        total = lattice.get_tot_scores(log_semiring=False, use_double_scores=True)
        if not torch.isfinite(total).all():
            raise ResourceError("No complete constrained acoustic path")
        path = k2.shortest_path(lattice, use_double_scores=True)
        result = path_summary(path, frames, self.tokens, self.words)
        if result["text"].split() != sequence:
            raise ResourceError("Constrained scorer emitted a different sequence")
        grammar = self.grammar(sequence)
        lm = grammar["unscaled_log_probability"] * self.settings["lm_scale"]
        bonus = len(sequence) * self.settings.get("word_bonus", 0.0)
        result.update(
            {
                "pronunciation_score": result["path_score"] - result["acoustic_score"],
                "lm_score": lm,
                "word_bonus_score": bonus,
                "score": result["path_score"] + lm + bonus,
                "lm_terms": grammar["terms"],
                "compiled_minus_kenlm": grammar["compiled_minus_kenlm"],
                "method": "Exact dense intersection of word-constrained lexical graph; tropical compiled G sequence score including final cost when supplied, otherwise KenLM with EOS",
            }
        )
        if not include_frames:
            result.pop("frame_labels")
        return result


class GrammarScorer:
    """Score the actual G backoff FST, including permitted fallback routes.

    An ARPA backoff FST may permit a cheaper fallback even when an explicit n-gram
    exists. KenLM uses the explicit n-gram in that case. Report, never hide, the
    difference; changing the intended grammar is a separate intervention.
    """

    def __init__(self, path, word_ids):
        self.word_ids = word_ids
        self.arcs, self.backoff, self.finals = (
            defaultdict(lambda: defaultdict(list)),
            defaultdict(list),
            {},
        )
        self.start = None
        for line in path.read_text().splitlines():
            fields = line.split()
            if not fields:
                continue
            if self.start is None:
                self.start = int(fields[0])
            if len(fields) <= 2:
                self.finals[int(fields[0])] = (
                    -float(fields[1]) if len(fields) == 2 else 0.0
                )
            else:
                source, target, label = map(int, fields[:3])
                score = -float(fields[4]) if len(fields) == 5 else 0.0
                if label in (0, word_ids["#0"]):
                    self.backoff[source].append((target, score))
                else:
                    self.arcs[source][label].append((target, score))
        self.closure = lru_cache(None)(self._closure)

    def _closure(self, state):
        result = {state: 0.0}
        for target, score in self.backoff[state]:
            for destination, remaining in self.closure(target):
                result[destination] = max(
                    result.get(destination, -math.inf), score + remaining
                )
        return tuple(result.items())

    def score(self, sequence):
        states = {self.start: 0.0}
        for word in sequence:
            following = {}
            for state, score in states.items():
                for source, backoff in self.closure(state):
                    for target, weight in self.arcs[source][self.word_ids[word]]:
                        following[target] = max(
                            following.get(target, -math.inf), score + backoff + weight
                        )
            states = following
        return max(
            (
                score + weight + self.finals[target]
                for state, score in states.items()
                for target, weight in self.closure(state)
                if target in self.finals
            ),
            default=-math.inf,
        )


def retained_lattice(graph, output, settings):
    frames, lengths = compact_frames(output, settings["acoustic_scale"])
    dense = k2.DenseFsaVec(
        frames, torch.tensor([[0, 0, int(lengths[0])]], dtype=torch.int32)
    )
    return k2.intersect_dense_pruned(
        graph,
        dense,
        search_beam=settings["search_beam"],
        output_beam=settings["output_beam"],
        min_active_states=settings["min_active_states"],
        max_active_states=settings["max_active_states"],
        allow_partial=False,
    )


def sequence_in_lattice(lattice, sequence, word_ids):
    return sequences_in_lattice(lattice, [sequence], word_ids)[0]


def sequences_in_lattice(lattice, sequences, word_ids):
    from asr.wfst.trie_scores import query

    return query(lattice, sequences, word_ids)


def sequences_in_lattice_product(lattice, sequences, word_ids):
    """Match a prefix trie once, retaining a separate score per terminal case."""
    prefixes, arcs, ends = {(): 0}, [], []
    for case, sequence in enumerate(sequences):
        prefix = ()
        for word in sequence:
            longer = (*prefix, word_ids[word])
            if longer not in prefixes:
                prefixes[longer] = len(prefixes)
                arcs.append((prefixes[prefix], prefixes[longer], word_ids[word], 0, -1))
            prefix = longer
        ends.append((prefixes[prefix], case))
    final = len(prefixes)
    arcs += [(state, final, -1, 0, case) for state, case in ends]
    arcs.sort()
    constraint = k2.Fsa.from_str(
        "\n".join(" ".join(map(str, a[:4])) for a in arcs) + f"\n{final}", acceptor=True
    )
    constraint.residual_case = torch.tensor([a[4] for a in arcs], dtype=torch.int32)
    matched = k2.connect(
        k2.intersect(k2.arc_sort(k2.invert(lattice[0])), k2.arc_sort(constraint))
    )
    if matched.num_arcs == 0:
        return [{"present": False, "score": None} for _ in sequences]
    matched = k2.create_fsa_vec([k2.top_sort(matched)])
    forward = matched.get_forward_scores(log_semiring=False, use_double_scores=True)
    final_arcs = matched.labels == -1
    sources = matched.arcs.values()[:, 0].long()[final_arcs]
    cases = matched.residual_case[final_arcs].long()
    totals = forward[sources] + matched.scores[final_arcs].double()
    scores = torch.full((len(sequences),), -torch.inf, dtype=torch.float64)
    scores.scatter_reduce_(0, cases, totals, reduce="amax", include_self=True)
    return [
        {"present": math.isfinite(s), "score": s if math.isfinite(s) else None}
        for s in scores.tolist()
    ]
