"""Static T∘L∘G compilation, adapting Icefall's Apache-2.0 compile_hlg recipe."""

import math
import time
from collections import Counter
from pathlib import Path

import k2
import kaldilm

from asr.wfst.compact import compact_topology
from asr.wfst.storage import save_tensor
from common.errors import ResourceError
from common.utils import fingerprints, write_json

GRAPH_RULES = "icefall-disambig-lg-determinize-prefix-once-v1"


def lexicon_fst(
    lexicon: dict[str, list[list[int]]],
    word_ids: dict[str, int],
    size: int,
    prefix_id: int = 3,
    phone_alternatives: dict | None = None,
    phone_seed_costs: dict | None = None,
    phone_position_costs: dict | None = None,
    phone_optional_costs: dict | None = None,
    pronunciation_settings: dict | None = None,
    phone_context_costs: dict | None = None,
) -> tuple[k2.Fsa, int]:
    optional = {}
    for token, cost in (phone_optional_costs or {}).items():
        token = int(token)
        if not 4 <= token < size or not math.isfinite(cost) or cost < 0:
            raise ResourceError(
                "Optional phone costs require lexical labels and finite nonnegative costs"
            )
        optional[token] = cost
    seed_costs = {}
    for token, cost in (phone_seed_costs or {}).items():
        token = int(token)
        if not 4 <= token < size or not math.isfinite(cost) or cost < 0:
            raise ResourceError(
                "Seed phone costs require lexical labels and finite nonnegative costs"
            )
        seed_costs[token] = cost
    alternatives = {}
    for source, options in (phone_alternatives or {}).items():
        source = int(source)
        if not 4 <= source < size:
            raise ResourceError("Phone alternatives must use lexical acoustic labels")
        alternatives[source] = []
        seen = {source}
        for option in options:
            target, cost = option["token"], option["cost"]
            if type(target) is not int or not 4 <= target < size or target in seen:
                raise ResourceError("Invalid or duplicate alternative phone label")
            if not math.isfinite(cost) or cost < 0:
                raise ResourceError(
                    "Alternative phone costs must be finite and nonnegative"
                )
            alternatives[source].append((target, cost))
            seen.add(target)
    positions = {}
    for position, mapping in (phone_position_costs or {}).items():
        if position not in ("final", "nonfinal"):
            raise ResourceError("Phone position must be final or nonfinal")
        positions[position] = {}
        for source, options in mapping.items():
            source = int(source)
            if not 4 <= source < size:
                raise ResourceError("Position costs must use lexical acoustic labels")
            seen = set()
            parsed = []
            for option in options:
                target, cost = option["token"], option["cost"]
                if type(target) is not int or not 4 <= target < size or target in seen:
                    raise ResourceError("Invalid or duplicate positional phone label")
                if not math.isfinite(cost) or cost < 0:
                    raise ResourceError("Position costs must be finite and nonnegative")
                seen.add(target)
                parsed.append((target, cost))
            if source not in seen:
                raise ResourceError("Position costs must retain the seed fallback")
            positions[position][source] = parsed
    contexts = {}
    for word, mapping in (phone_context_costs or {}).items():
        if word not in lexicon:
            raise ResourceError("Context costs contain an unknown written word")
        contexts[word] = {}
        for coordinate, options in mapping.items():
            length, offset, source = map(int, coordinate.split(":"))
            if not (0 <= offset < length and 4 <= source < size):
                raise ResourceError("Invalid word-context coordinate")
            if not any(len(p) == length and p[offset] == source for p in lexicon[word]):
                raise ResourceError(
                    "Word-context coordinate does not match a pronunciation"
                )
            seen, parsed = set(), []
            for option in options:
                token, cost = option["token"], option["cost"]
                if (
                    type(token) is not int
                    or not 4 <= token < size
                    or token in seen
                    or not math.isfinite(cost)
                    or cost < 0
                ):
                    raise ResourceError("Invalid word-context label/cost")
                seen.add(token)
                parsed.append((token, cost))
            if source not in seen:
                raise ResourceError("Context costs must retain the seed fallback")
            contexts[word][coordinate] = parsed
    expanded_paths = bool(alternatives or positions or optional or contexts)
    pronunciation_settings = pronunciation_settings or {}
    if set(pronunciation_settings) - set(lexicon):
        raise ResourceError("Pronunciation settings contain unknown words")
    pairs = []
    for word in sorted(lexicon):
        settings = pronunciation_settings.get(word, [{} for _ in lexicon[word]])
        if len(settings) != len(lexicon[word]):
            raise ResourceError("Pronunciation settings must match the lexicon paths")
        for pronunciation, setting in zip(lexicon[word], settings):
            cost, exact = setting.get("cost", 0.0), setting.get("exact", False)
            if not math.isfinite(cost) or cost < 0 or type(exact) is not bool:
                raise ResourceError("Invalid pronunciation cost or exact-label flag")
            pairs.append((word, tuple(pronunciation), cost, exact))
    if any(
        not pronunciation or (not exact and all(t in optional for t in pronunciation))
        for _, pronunciation, _, exact in pairs
    ):
        raise ResourceError(
            "Every pronunciation must retain a nonoptional acoustic label"
        )
    counts = Counter(p for _, p, _, _ in pairs)
    prefixes = {p[:n] for _, p, _, _ in pairs for n in range(1, len(p))}
    used = Counter()
    # 0 is a once-only start; 1 is the word loop. No inserted interword token.
    arcs = [[0, 1, 0, 0, 0], [0, 1, prefix_id, 0, 0]]
    next_state = 2
    disambig = size + 1  # size is reserved for Compact CTC's auxiliary label.
    for pair_index, (word, original, pronunciation_cost, exact) in enumerate(pairs, 1):
        pronunciation = list(original)
        if expanded_paths:
            # Alternatives create new homophones/prefixes, so every pronunciation
            # needs a unique disambiguation symbol before LG determinization.
            pronunciation.append(disambig + pair_index)
        elif counts[original] > 1 or original in prefixes:
            used[original] += 1
            pronunciation.append(disambig + used[original])
        current = 1
        for index, token in enumerate(pronunciation):
            target = 1 if index == len(pronunciation) - 1 else next_state
            if target == next_state:
                next_state += 1
            options = (
                [(token, 0)]
                if exact
                else [(token, seed_costs.get(token, 0))] + alternatives.get(token, [])
            )
            if not exact and index < len(original):
                # Determine position before the appended disambiguation symbol.
                position = "final" if index == len(original) - 1 else "nonfinal"
                options = positions.get(position, {}).get(token, options)
                options = contexts.get(word, {}).get(
                    f"{len(original)}:{index}:{token}", options
                )
                if token in optional:
                    # A length-label omission consumes no acoustic token. It
                    # must never make an entire word an epsilon-only cycle.
                    options = options + [(0, optional[token])]
            for alternate, cost in options:
                arcs.append(
                    [
                        current,
                        target,
                        alternate,
                        word_ids[word] if index == 0 else 0,
                        -cost - (pronunciation_cost if index == 0 else 0),
                    ]
                )
            current = target
    # Propagate the grammar's backoff marker from every word-emitting state.
    for state in sorted({a[0] for a in arcs if a[3] > 0}):
        arcs.append([state, state, disambig, word_ids["#0"], 0])
    arcs.append([1, next_state, -1, -1, 0])
    text = "\n".join(" ".join(map(str, a)) for a in sorted(arcs)) + f"\n{next_state}"
    maximum = len(pairs) if expanded_paths else max(used.values(), default=0)
    return k2.arc_sort(k2.Fsa.from_str(text, acceptor=False)), maximum


def compile_graph(
    directory: Path, lexicon: dict, tokens: list[str], settings: dict
) -> dict:
    report = {"status": "compiling", "stages": {}, "artifacts": {}}
    start = time.perf_counter()

    def save(name, graph):
        if graph.num_arcs == 0:
            raise ResourceError(f"Empty graph at {name}")
        path = save_tensor(
            graph.as_dict(),
            directory,
            name,
            compressed=settings.get("compress_artifacts", False),
        )
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
        L, maximum = lexicon_fst(
            lexicon,
            word_ids,
            len(tokens),
            tokens.index("▁"),
            settings.get("phone_alternatives"),
            settings.get("phone_seed_costs"),
            settings.get("phone_position_costs"),
            settings.get("phone_optional_costs"),
            settings.get("pronunciation_settings"),
            settings.get("phone_context_costs"),
        )
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
        word_bonus = settings.get("word_bonus", 0.0)
        if not math.isfinite(word_bonus):
            raise ResourceError("Word bonus must be finite")
        # Vocabulary IDs are contiguous after <eps>. Reward lexical emissions,
        # including <unk>, but never backoff or sentence-boundary arcs.
        lexical = (G.labels > 0) & (G.labels <= len(lexicon))
        G.scores = G.scores * settings["lm_scale"] + lexical * word_bonus
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
