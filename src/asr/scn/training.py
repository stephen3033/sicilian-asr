"""Train actual decoder LMs; the historical lm.py probes remain independent."""

import json
import math
import subprocess
import time
from collections import Counter

from asr.scn.dataset import digest, read_jsonl
from asr.scn.lexicon import build, seed_entries
from asr.wfst.arpa import read_unigrams
from common.errors import ResourceError
from common.resources import load_resources
from common.utils import fingerprints, local_path, write_json
from ipa.inventory import inventory


def verify_bundle(root, receipt_path):
    path = local_path(root, str(receipt_path))
    receipt = json.loads(path.read_text())
    if receipt.get("status") != "complete" or receipt.get("key") != digest(
        receipt["identity"]
    ):
        raise ResourceError("Incomplete or invalid Sicilian model receipt")
    for name, expected in receipt["artifacts"].items():
        if fingerprints(local_path(path.parent, name))["sha256"] != expected:
            raise ResourceError(f"Changed trained Sicilian artifact: {name}")
    return path.parent, receipt


def score(arpa, texts, vocabulary):
    import kenlm

    model = kenlm.Model(str(arpa))
    loss, total, oov, words = 0.0, 0, 0, 0
    for text in texts:
        tokens = text.split()
        loss -= model.score(text, bos=True, eos=True) * math.log(10)
        total += len(tokens) + 1
        words += len(tokens)
        oov += sum(w not in vocabulary for w in tokens)
    return {
        "perplexity": math.exp(loss / total) if total else None,
        "development_tokens": words,
        "oov_tokens": oov,
        "oov_rate": oov / words if words else None,
    }


def train_candidates(root, output, recipe):
    corpus = json.loads((output / "corpus.json").read_text())
    for name, key in (
        ("training.jsonl", "training_sha256"),
        ("development.json", "development_sha256"),
    ):
        if fingerprints(output / name)["sha256"] != corpus[key]:
            raise ResourceError(f"Changed corpus artifact: {name}; rebuild corpus")
    rows = read_jsonl(output / "training.jsonl")
    dev = json.loads((output / "development.json").read_text())
    tools = json.loads((root / ".cache/scn/tools/receipt.json").read_text())
    estimator = local_path(root, tools["tools"]["lmplz"]["path"])
    if fingerprints(estimator)["sha256"] != tools["tools"]["lmplz"]["sha256"]:
        raise ResourceError("KenLM estimator changed")
    manifest = load_resources(root)["resources"]
    tokens, tokenizer = inventory(
        local_path(root, manifest["zipa_tokens"]["path"]),
        local_path(root, manifest["zipa_tokenizer"]["path"]),
    )
    seeds = seed_entries(
        [
            root / ".cache/scn/acquired/wikipron-broad.tsv",
            root / ".cache/scn/acquired/wikipron-narrow.tsv",
        ]
    )
    all_lexicon, evidence = build(
        {w for r in rows for w in r["tokens"]}, seeds, tokens, tokenizer
    )
    write_json(output / "pronunciations.json", evidence)
    write_json(output / "lexicon.json", all_lexicon)
    channel = [r for r in rows if r["cohort"] == "channel"]
    background = [r for r in rows if r["cohort"] == "background"]
    # Stable reviewed source order, no final-test terms or frequencies.
    rank = {
        s: i
        for i, s in enumerate(recipe["preferred_sources"] + recipe["expansion_sources"])
    }
    background.sort(
        key=lambda r: (rank.get(r["source_id"], 99), r["document_id"], r["id"])
    )
    base = sum(len(r["tokens"]) for r in channel)
    sizes = sorted(
        set([base] + [min(n, corpus["tokens"]) for n in recipe["checkpoints"]])
    )
    results, seen = [], set()
    for size in sizes:
        selected, count = list(channel), base
        for row in background:
            if count >= size:
                break
            selected.append(row)
            count += len(row["tokens"])
        signature = digest([r["id"] for r in selected])
        if signature in seen:
            continue
        seen.add(signature)
        frequency = Counter(w for r in selected for w in r["tokens"])
        vocabulary = sorted(set(frequency) & set(all_lexicon))
        if len(vocabulary) < 2:
            raise ResourceError("Too few supported words to train")
        for order in recipe["orders"]:
            identity = {
                "training": digest(selected),
                "vocabulary": digest(vocabulary),
                "lexicon": digest({w: all_lexicon[w] for w in vocabulary}),
                "order": order,
                "kenlm": tools,
                "normalization": corpus["normalization"],
                "source_code": {
                    p: fingerprints(root / "src" / p)["sha256"]
                    for p in (
                        "asr/scn/training.py",
                        "asr/scn/lexicon.py",
                        "asr/scn/orthography.py",
                    )
                },
            }
            key = digest(identity)
            directory = root / ".cache/scn/models" / key
            directory.mkdir(parents=True, exist_ok=True)
            receipt_path = directory / "model.json"
            if not receipt_path.exists():
                started = time.perf_counter()
                (directory / "training.txt").write_text(
                    "".join(r["text"] + "\n" for r in selected)
                )
                (directory / "vocabulary.txt").write_text("\n".join(vocabulary) + "\n")
                write_json(
                    directory / "lexicon.json", {w: all_lexicon[w] for w in vocabulary}
                )
                command = [
                    str(estimator),
                    "-o",
                    str(order),
                    "-S",
                    "512M",
                    "-T",
                    str(directory / "tmp"),
                    "--text",
                    str(directory / "training.txt"),
                    "--arpa",
                    str(directory / "model.arpa"),
                    "--limit_vocab_file",
                    str(directory / "vocabulary.txt"),
                ]
                fallback = False
                with (directory / "training.log").open("w") as log:
                    process = subprocess.run(command, stdout=log, stderr=log)
                if (
                    process.returncode
                    and "discount" in (directory / "training.log").read_text().lower()
                ):
                    fallback = True
                    with (directory / "training.log").open("a") as log:
                        process = subprocess.run(
                            command + ["--discount_fallback", "0.5", "1", "1.5"],
                            stdout=log,
                            stderr=log,
                        )
                if process.returncode:
                    raise ResourceError(f"LM training failed: {directory}/training.log")
                unigrams = set(read_unigrams(directory / "model.arpa")) - {
                    "<s>",
                    "</s>",
                    "<unk>",
                }
                if unigrams != set(vocabulary):
                    raise ResourceError(
                        "Trained LM vocabulary does not match supported lexicon"
                    )
                receipt = {
                    "status": "complete",
                    "key": key,
                    "identity": identity,
                    "order": order,
                    "training_records": len(selected),
                    "training_tokens": count,
                    "vocabulary_size": len(vocabulary),
                    "discount_fallback": fallback,
                    "training_seconds": time.perf_counter() - started,
                    "artifacts": {
                        p.name: fingerprints(p)["sha256"]
                        for p in directory.iterdir()
                        if p.is_file()
                    },
                }
                write_json(receipt_path, receipt)
            _, receipt = verify_bundle(root, str(receipt_path.relative_to(root)))
            evaluation = score(
                directory / "model.arpa",
                [r["normalized"] for r in dev],
                set(vocabulary),
            )
            result = {
                "receipt": str(receipt_path.relative_to(root)),
                "order": order,
                "tokens": count,
                "vocabulary_size": len(vocabulary),
                **evaluation,
                "arpa_bytes": (directory / "model.arpa").stat().st_size,
                "training_seconds": receipt["training_seconds"],
            }
            print(result, flush=True)
            results.append(result)
            write_json(output / "candidates.json", results)
    return results
