"""Compact CTC with the auxiliary-frame construction of Laptev et al. (2022).

Topology follows NeMo v1.23 build_compact_topo (Apache-2.0), including a
separate blank state and termination from the hub after an auxiliary frame.
https://github.com/NVIDIA/NeMo/blob/v1.23.0/nemo/collections/asr/parts/k2/topologies.py
"""

import k2
import torch

from sicilian_asr.acoustic.zipa import AcousticOutput
from sicilian_asr.resources import ResourceError

TOPOLOGY_RULES = "laptev-compact-nemo-blank-state-aux-neutral-frames-v1"


def compact_topology(vocabulary_size: int) -> k2.Fsa:
    if vocabulary_size < 2:
        raise ResourceError("Compact CTC needs blank and at least one token")
    auxiliary = vocabulary_size
    final = vocabulary_size + 1
    arcs = [f"0 {i+1} {i} {i} 0" for i in range(vocabulary_size)]
    arcs.append(f"0 {final} -1 -1 0")
    for i in range(vocabulary_size):
        arcs.extend([f"{i+1} 0 {auxiliary} 0 0", f"{i+1} {i+1} {i} 0 0"])
    arcs.append(str(final))
    return k2.arc_sort(k2.Fsa.from_str("\n".join(arcs), acceptor=False))


def compact_frames(output: AcousticOutput, acoustic_scale: float = 1) -> tuple[torch.Tensor, torch.Tensor]:
    output.validate()
    if output.blank_id != 0 or not 0 < acoustic_scale < float("inf"):
        raise ResourceError("Compact CTC requires blank 0 and a finite positive acoustic scale")
    p = output.log_probs
    padded = torch.zeros((p.shape[0], 2*p.shape[1], p.shape[2]+1), dtype=p.dtype)
    padded[:, ::2, :-1] = p * acoustic_scale
    padded[:, ::2, -1] = -torch.inf
    return padded, output.lengths * 2


def decode(graph: k2.Fsa, output: AcousticOutput, words: dict[int, str], settings: dict) -> dict:
    if output.log_probs.shape[0] != 1:
        raise ResourceError("Sequential decoder expects one segment per call")
    frames, lengths = compact_frames(output, settings["acoustic_scale"])
    supervisions = torch.tensor([[0, 0, int(lengths[0])]], dtype=torch.int32)
    dense = k2.DenseFsaVec(frames, supervisions)
    graph_vec = k2.create_fsa_vec([graph]) if len(graph.shape) == 2 else graph
    lattice = k2.intersect_dense_pruned(graph_vec, dense,
        search_beam=settings["search_beam"], output_beam=settings["output_beam"],
        min_active_states=settings["min_active_states"], max_active_states=settings["max_active_states"], allow_partial=False)
    totals = lattice.get_tot_scores(log_semiring=False, use_double_scores=True)
    if not torch.isfinite(totals).all():
        raise ResourceError("Empty/nonfinite WFST lattice; no complete path within decoder beams")
    best = k2.shortest_path(lattice, use_double_scores=True)
    labels = best.aux_labels.values if isinstance(best.aux_labels, k2.RaggedTensor) else best.aux_labels
    ids = [i for i in labels.tolist() if i > 0]
    if any(i not in words or words[i] in {"#0", "<s>", "</s>"} for i in ids):
        raise ResourceError("Decoder emitted an invalid word ID")
    result = {"text": " ".join(words[i] for i in ids), "word_ids": ids, "score": float(totals[0]),
              "original_frames": int(output.lengths[0]), "decoder_frames": int(lengths[0]), "lattice_arcs": lattice.num_arcs}

    def summarize_path(fsa, arc_ids):
        index = torch.tensor(arc_ids, dtype=torch.int32)
        input_ids = fsa.labels[index.long()]
        valid = input_ids >= 0
        if len(input_ids) != int(lengths[0]) + 1:
            raise ResourceError("Unexpected decoder path length in score diagnostics")
        acoustic = float(frames[0, torch.arange(len(input_ids))[valid], input_ids[valid].long()].double().sum())
        total = float(fsa.scores[index.long()].double().sum())
        def values(attribute):
            if isinstance(attribute, k2.RaggedTensor):
                selected, _ = attribute.index(index, axis=0, need_value_indexes=False)
                return selected.values.tolist()
            return attribute[index.long()].tolist()
        word_ids = [i for i in values(fsa.aux_labels) if i > 0]
        phone_ids = [i for i in values(fsa.tokens) if 0 < i < len(output.tokens)] if hasattr(fsa, "tokens") else []
        return {"text": " ".join(words[i] for i in word_ids), "score": total,
                "acoustic_score": acoustic, "grammar_score": total - acoustic,
                "pronunciation": "".join(output.tokens[i] for i in phone_ids),
                "pronunciation_token_ids": phone_ids}

    chosen = summarize_path(best, list(range(best.num_arcs)))
    result.update({k: v for k, v in chosen.items() if k not in {"text", "score"}})
    count = settings.get("diagnostic_paths", 0)
    if count:
        candidates = {chosen["text"]: chosen}
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(0)
            sampled = k2.random_paths(lattice, num_paths=count, use_double_scores=True).tolist()[0]
        for path in sampled:
            candidate = summarize_path(lattice, path)
            previous = candidates.get(candidate["text"])
            if previous is None or candidate["score"] > previous["score"]:
                candidates[candidate["text"]] = candidate
        result["diagnostics"] = {"candidate_method": "best path plus sampled lattice paths; not exhaustive N-best",
            "sampled_paths": count, "seed": 0, "acoustic_scale": settings["acoustic_scale"],
            "lm_scale": settings["lm_scale"], "scores": "natural log; scales already applied",
            "candidates": sorted(candidates.values(), key=lambda c: (-c["score"], c["text"]))}
    return result
