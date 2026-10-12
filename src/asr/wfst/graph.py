"""Static T∘L∘G compilation, adapting Icefall's Apache-2.0 compile_hlg recipe."""

import time
from collections import Counter
from pathlib import Path

import k2
import kaldilm
import torch

from asr.wfst.compact import compact_topology
from common.errors import ResourceError
from common.utils import fingerprints, write_json

GRAPH_RULES = "icefall-disambig-lg-determinize-prefix-once-v1"


def lexicon_fst(
    lexicon: dict[str, list[list[int]]],
    word_ids: dict[str, int],
    size: int,
    prefix_id: int = 3,
) -> tuple[k2.Fsa, int]:
    pairs = [(w, tuple(p)) for w in sorted(lexicon) for p in lexicon[w]]
    counts = Counter(p for _, p in pairs)
    prefixes = {p[:n] for _, p in pairs for n in range(1, len(p))}
    used = Counter()
    # 0 is a once-only start; 1 is the word loop. No inserted interword token.
    arcs = [[0, 1, 0, 0, 0], [0, 1, prefix_id, 0, 0]]
    next_state = 2
    disambig = size + 1  # size is reserved for Compact CTC's auxiliary label.
    for word, original in pairs:
        pronunciation = list(original)
        if counts[original] > 1 or original in prefixes:
            used[original] += 1
            pronunciation.append(disambig + used[original])
        current = 1
        for index, token in enumerate(pronunciation):
            target = 1 if index == len(pronunciation) - 1 else next_state
            if target == next_state:
                next_state += 1
            arcs.append(
                [current, target, token, word_ids[word] if index == 0 else 0, 0]
            )
            current = target
    # Propagate the grammar's backoff marker from every word-emitting state.
    for state in sorted({a[0] for a in arcs if a[3] > 0}):
        arcs.append([state, state, disambig, word_ids["#0"], 0])
    arcs.append([1, next_state, -1, -1, 0])
    text = "\n".join(" ".join(map(str, a)) for a in sorted(arcs)) + f"\n{next_state}"
    return k2.arc_sort(k2.Fsa.from_str(text, acceptor=False)), max(
        used.values(), default=0
    )


def compile_graph(
    directory: Path, lexicon: dict, tokens: list[str], settings: dict
) -> dict:
    report = {"status": "compiling", "stages": {}, "artifacts": {}}
    start = time.perf_counter()

    def save(name, graph):
        if graph.num_arcs == 0:
            raise ResourceError(f"Empty graph at {name}")
        path = directory / f"{name}.pt"
        torch.save(graph.as_dict(), path)
        report["stages"][name] = {
            "states": graph.shape[0],
            "arcs": graph.num_arcs,
            "elapsed_seconds": time.perf_counter() - start,
        }
        report["artifacts"][path.name] = fingerprints(path)["sha256"]
        write_json(directory / "compilation.json", report)
        print(f"{name}: {graph.shape[0]:,} states, {graph.num_arcs:,} arcs", flush=True)

    try:
        vocabulary = sorted(lexicon)
        word_ids = {
            w: i for i, w in enumerate(["<eps>"] + vocabulary + ["<s>", "</s>", "#0"])
        }
        (directory / "words.txt").write_text(
            "".join(f"{w} {i}\n" for w, i in word_ids.items()), encoding="utf-8"
        )
        L, maximum = lexicon_fst(lexicon, word_ids, len(tokens), tokens.index("▁"))
        table = tokens + ["<compact_eps>"] + [f"#{i}" for i in range(maximum + 1)]
        (directory / "tokens.txt").write_text(
            "".join(f"{t} {i}\n" for i, t in enumerate(table)), encoding="utf-8"
        )
        T = compact_topology(len(tokens))
        save("T", T)
        save("L", L)
        fst = kaldilm.arpa2fst(
            str(directory / "restricted.arpa"),
            disambig_symbol="#0",
            read_symbol_table=str(directory / "words.txt"),
            max_order=settings.get("lm_order", 3),
        )
        (directory / "G.fst.txt").write_text(fst)
        G = k2.arc_sort(k2.Fsa.from_openfst(fst, acceptor=False))
        del fst
        G.scores = G.scores * settings["lm_scale"]
        save("G", G)
        print("Composing and determinizing LG", flush=True)
        LG = k2.connect(k2.compose(L, G))
        del L, G
        LG = k2.connect(k2.determinize(LG))
        labels = LG.labels.clone()
        labels[labels >= len(tokens) + 1] = 0
        LG.labels = labels
        aux = (
            LG.aux_labels.values
            if isinstance(LG.aux_labels, k2.RaggedTensor)
            else LG.aux_labels
        )
        aux[aux >= word_ids["#0"]] = 0
        LG = k2.connect(k2.remove_epsilon(LG))
        if isinstance(LG.aux_labels, k2.RaggedTensor):
            LG.aux_labels = LG.aux_labels.remove_values_eq(0)
        LG = k2.arc_sort(LG)
        save("LG", LG)
        TLG = k2.arc_sort(k2.connect(k2.compose(T, LG, inner_labels="tokens")))
        save("TLG", TLG)
        report["status"] = "complete"
        report["seconds"] = time.perf_counter() - start
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        write_json(directory / "compilation.json", report)
    return report
