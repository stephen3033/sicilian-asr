"""Matched raw/converted KenLM controls with a shared vocabulary and pruning."""

import math
import os
import subprocess
from collections import Counter

from asr.scn.dataset import digest
from asr.scn.orthography import normalize
from asr.wfst.arpa import entries, read_unigrams
from common.errors import ResourceError
from common.utils import write_json
from pipeline.experiment import SPECIAL, read, seal, sha, verified_files


def mixture(channel, wiki, share, prior_tokens=0):
    """Whole independent sentences; channel repeats change weight, never evidence."""
    if not channel or not 0 <= share < 1 or (share and not wiki) or prior_tokens < 0:
        raise ValueError(
            "A mixture requires channel text and nonempty requested Wikipedia"
        )
    channel_tokens = sum(len(t.split()) for t in channel)
    wiki_tokens = sum(len(t.split()) for t in wiki) if share else 0
    if not channel_tokens:
        raise ValueError("Empty channel tokens")
    wiki_copies = max(1, math.ceil(share*(channel_tokens+prior_tokens)/((1-share)*wiki_tokens))) if share else 0
    target_channel = max(channel_tokens, math.ceil(wiki_tokens*wiki_copies*(1-share)/share-prior_tokens)) if share else channel_tokens
    copies, remaining = divmod(target_channel, channel_tokens)
    partial, partial_tokens = [], 0
    for text in sorted(channel, key=digest):
        if partial_tokens >= remaining:
            break
        partial.append(text)
        partial_tokens += len(text.split())
    effective_channel = channel_tokens*copies + partial_tokens
    weighted = channel * copies + partial + wiki * wiki_copies
    return weighted, {
        "channel_unique_sentences": len(set(channel)),
        "channel_source_tokens": channel_tokens,
        "wikipedia_unique_sentences": len(set(wiki)) if share else 0,
        "wikipedia_source_tokens": wiki_tokens,
        "channel_copies": copies,
        "channel_partial_sentences": len(partial),
        "channel_effective_tokens": effective_channel,
        "wikipedia_copies": wiki_copies,
        "wikipedia_effective_tokens": wiki_tokens * wiki_copies,
        "shared_unigram_prior_tokens": prior_tokens,
        "requested_wikipedia_share": share,
        "effective_wikipedia_share": wiki_tokens * wiki_copies
        / (wiki_tokens * wiki_copies + effective_channel + prior_tokens),
    }


def evaluate_lm(arpa, texts, vocabulary, channel_texts):
    import kenlm

    model = kenlm.Model(str(arpa))
    grams = {n: set() for n in (2, 3)}
    for n, words, _, _ in entries(arpa):
        if n in grams:
            grams[n].add(words)
    channel = {
        n: {
            tuple(ws[i : i + n])
            for text in channel_texts
            for ws in [["<s>", *text.split(), "</s>"]]
            for i in range(len(ws) - n + 1)
        }
        for n in (2, 3)
    }
    logprob = 0.0
    units = tokens = oov = 0
    totals, seen, new_totals, new_seen = Counter(), Counter(), Counter(), Counter()
    for text in texts:
        raw = normalize(text).split()
        mapped = [w if w in vocabulary else "<unk>" for w in raw]
        logprob += model.score(" ".join(mapped), bos=True, eos=True)
        units += len(mapped) + 1
        tokens += len(raw)
        oov += sum(w not in vocabulary for w in raw)
        ws = ["<s>", *mapped, "</s>"]
        for n in (2, 3):
            for i in range(len(ws) - n + 1):
                gram = tuple(ws[i : i + n])
                totals[n] += 1
                seen[n] += gram in grams[n]
                if gram not in channel[n]:
                    new_totals[n] += 1
                    new_seen[n] += gram in grams[n]
    return {
        "perplexity": 10 ** (-logprob / units) if units else None,
        "oov_tokens": oov,
        "tokens": tokens,
        "oov_rate": oov / tokens if tokens else None,
        "context_coverage": {
            n: seen[n] / totals[n] if totals[n] else None for n in totals
        },
        "unseen_channel_context_coverage": {
            n: {"covered": new_seen[n], "total": new_totals[n]} for n in (2, 3)
        },
    }


def train_model(root, directory, channel, wiki, share, order, vocabulary, prune):
    tool = read(root / ".cache/scn/tools/receipt.json")["tools"]["lmplz"]
    if sha(root / tool["path"]) != tool["sha256"]:
        raise ResourceError("KenLM estimator changed")
    vocabulary = set(vocabulary) - SPECIAL
    # The same one-token prior for EVERY word keeps absent vocabulary supported
    # without a corpus-dependent prior. Include it in effective mixture shares.
    fallback = sorted(vocabulary)
    texts, weighting = mixture(channel, wiki, share, prior_tokens=len(fallback))
    # KenLM 0.3 rejects literal <unk> during estimation. Its fixed-vocabulary
    # option prunes complete OOV n-grams; never remove tokens and bridge context.
    identity = {
        "channel": digest(channel),
        "wiki": digest(wiki),
        "weighting": weighting,
        "vocabulary": digest(sorted(vocabulary)),
        "order": order,
        "prune": prune,
        "tool": tool,
        "shared_unigram_prior": fallback,
        "oov_policy": "Estimate source sequences intact, prune OOV n-grams with limit_vocab_file; held-out OOV uses the common <unk> symbol. No whitespace-skipping of special symbols.",
        "implementation": sha(__file__),
    }
    receipt = directory / "model.json"
    if receipt.exists():
        saved = read(receipt)
        if saved["key"] != digest(identity):
            raise ResourceError("LM inputs changed")
        verified_files(directory, saved)
        return saved
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "training.txt").open("w") as stream:
        for text in texts:
            stream.write(text + "\n")
        for word in fallback:
            stream.write(word + "\n")
    (directory / "vocabulary.txt").write_text("\n".join(sorted(vocabulary)) + "\n")
    with (directory / "build.log").open("w") as log:
        subprocess.run(
            [
                str(root / tool["path"]),
                "-o",
                str(order),
                "--discount_fallback",
                "--prune",
                *map(str, prune[:order]),
                "-S",
                "512M",
                "-T",
                str(directory),
                "--text",
                str(directory / "training.txt"),
                "--arpa",
                str(directory / "model.arpa"),
                "--limit_vocab_file",
                str(directory / "vocabulary.txt"),
            ],
            stdout=log,
            stderr=log,
            check=True,
            env={**os.environ, "OMP_NUM_THREADS": "2"},
        )
    if set(read_unigrams(directory / "model.arpa")) - SPECIAL != vocabulary:
        raise ResourceError("LM/lexicon inventories differ")
    counts = Counter(n for n, _, _, _ in entries(directory / "model.arpa"))
    write_json(
        directory / "statistics.json",
        {
            "ngram_counts": dict(counts),
            "weighting": weighting,
            "unigram_fallback_tokens": len(fallback),
            "effective_tokens_including_fallback": sum(len(t.split()) for t in texts)
            + len(fallback),
        },
    )
    return seal(
        directory,
        "model.json",
        identity,
        ["training.txt", "vocabulary.txt", "model.arpa", "statistics.json"],
    )


def train_controls(root, output, bundle, corpus, recipe):
    channel = list(
        dict.fromkeys(
            normalize(r["text"]) for r in read(bundle / "channel.json")["training"]
        )
    )
    import json

    admitted = [json.loads(line) for line in (corpus / "admitted.jsonl").open()]
    vocabulary = set(read(bundle / "graph/lexicon_selected.json")) - SPECIAL
    heldout = [r["reference"] for r in read(bundle / "recordings.json")]
    configs = [(f"channel-o{o}", "channel", 0, o) for o in recipe["orders"]]
    configs += [
        (f"{kind}-w{int(share * 100)}-o{o}", kind, share, o)
        for kind in ("converted", "raw")
        for share in recipe["wikipedia_shares"]
        for o in recipe["orders"]
    ]
    attempts = []
    for prune in recipe["pruning_attempts"]:
        models = []
        for label, kind, share, order in configs:
            directory = output / ("prune-" + "-".join(map(str, prune))) / label
            wiki = [r[kind] for r in admitted] if kind != "channel" else []
            print("LM", label, prune, flush=True)
            train_model(root, directory, channel, wiki, share, order, vocabulary, prune)
            stats = read(directory / "statistics.json")
            models.append(
                {
                    "id": label,
                    "kind": kind,
                    "share": share,
                    "order": order,
                    "directory": str(directory),
                    "receipt_sha256": sha(directory / "model.json"),
                    "source_passage_ids_sha256": digest([r["id"] for r in admitted])
                    if share
                    else None,
                    "unique_articles": len({r["document_id"] for r in admitted})
                    if share
                    else 0,
                    **stats,
                    "heldout": evaluate_lm(
                        directory / "model.arpa", heldout, vocabulary, channel
                    ),
                }
            )
            write_json(output / "progress.json", {"models": models, "prune": prune})
        maximum = max(
            sum(v for k, v in m["ngram_counts"].items() if int(k) > 1) for m in models
        )
        attempts.append({"prune": prune, "maximum_context_ngrams": maximum})
        if maximum <= recipe["maximum_context_ngrams"]:
            break
    else:
        raise ResourceError("All shared pruning policies exceed the n-gram budget")
    converted = sorted(
        [m for m in models if m["kind"] == "converted"],
        key=lambda m: (m["heldout"]["perplexity"], m["id"]),
    )[:2]
    selected = [m["id"] for m in converted]
    selected += [m["id"].replace("converted", "raw") for m in converted]
    selected += [
        min(
            (m for m in models if m["kind"] == "channel"),
            key=lambda m: (m["heldout"]["perplexity"], m["id"]),
        )["id"]
    ]
    summary = {
        "status": "complete",
        "models": models,
        "selected_for_decode": selected,
        "shared_prune": prune,
        "pruning_attempts": attempts,
        "selection": "Held-out perplexity, then stable ID; development selection only.",
        "corpus_receipt_sha256": sha(corpus / "corpus.receipt.json"),
    }
    write_json(output / "models.json", summary)
    return summary
