"""Training-only support-floor selection and matched LM construction repairs.

Vocabulary support is a probability floor, never a fabricated sentence. All
explicit contexts come from retained training passages. Backoff weights are
recomputed after changing the unigram distribution.
"""

import math
import os
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

from asr.scn.orthography import normalize
from asr.wfst.arpa import entries
from common.artifacts import digest, read, seal, sha, verified_files
from common.errors import ResourceError
from common.utils import write_json
from pipeline.experiment import SPECIAL
from pipeline.scn_lm_experiment import evaluate_lm


class BackoffModel:
    def __init__(self, path):
        self.probabilities, self.backoffs = {}, {}
        for n, words, logp, line in entries(Path(path)):
            self.probabilities[words] = 10**logp
            fields = line.split()
            if len(fields) == n + 2:
                self.backoffs[words] = 10 ** float(fields[-1])
        self.order = max(map(len, self.probabilities))

    def probability(self, word, history=()):
        history = tuple(history[-(self.order - 1) :])
        key = (*history, word)
        if key in self.probabilities:
            return self.probabilities[key]
        if history:
            return self.backoffs.get(history, 1) * self.probability(word, history[1:])
        return self.probabilities.get(("<unk>",), 0)

    def followers(self):
        result = defaultdict(dict)
        for words, p in self.probabilities.items():
            if len(words) > 1:
                result[words[:-1]][words[-1]] = p
        return result

    def audit(self):
        unigram_mass = sum(
            p for ws, p in self.probabilities.items() if len(ws) == 1 and ws[0] != "<s>"
        )
        deviations, masses = [], {}
        for history, following in sorted(
            self.followers().items(), key=lambda r: len(r[0])
        ):
            lower_mass = masses.get(history[1:], unigram_mass)
            seen_lower = sum(self.probability(w, history[1:]) for w in following)
            mass = sum(following.values()) + self.backoffs.get(history, 1) * (
                lower_mass - seen_lower
            )
            masses[history] = mass
            deviations.append(abs(mass - 1))
        return {
            "unigram_mass": unigram_mass,
            "explicit_histories": len(masses),
            "maximum_conditional_mass_error": max(deviations, default=0),
            "histories_outside_1e_5": sum(d > 1e-5 for d in deviations),
            "ngram_counts": dict(Counter(map(len, self.probabilities))),
        }

    def supported(self, vocabulary, floor_mass):
        if not 0 < floor_mass < 1:
            raise ValueError("Vocabulary floor mass must be between zero and one")
        vocabulary = set(vocabulary) - SPECIAL
        before = self.audit()
        total = before["unigram_mass"]
        if total <= 0:
            raise ResourceError("Empty unigram probability mass")
        for key in list(self.probabilities):
            if len(key) == 1 and key != ("<s>",):
                self.probabilities[key] *= (1 - floor_mass) / total
        for word in vocabulary:
            key = (word,)
            self.probabilities[key] = self.probabilities.get(key, 0) + floor_mass / len(
                vocabulary
            )
        # Histories with no explicit followers use the new lower-order model.
        self.backoffs = {
            k: 1.0
            for k in self.probabilities
            if len(k) < self.order and k[-1] != "</s>"
        }
        rescaled = []
        for history, following in sorted(
            self.followers().items(), key=lambda r: len(r[0])
        ):
            explicit_mass = sum(following.values())
            lower_seen = sum(self.probability(w, history[1:]) for w in following)
            denominator = 1 - lower_seen
            if denominator < -1e-8:
                raise ResourceError("Invalid lower-order probability mass")
            # Roundoff or vocabulary restriction can exhaust explicit mass.
            # Reserve positive backoff support unless all words are explicit.
            target = 1.0 if denominator <= 1e-12 else min(explicit_mass, 1 - 1e-10)
            if abs(target - explicit_mass) > 1e-14:
                for w, p in following.items():
                    self.probabilities[(*history, w)] = p * target / explicit_mass
                rescaled.append(list(history))
            self.backoffs[history] = (
                (1 - target) / denominator if denominator > 1e-12 else 1.0
            )
        after = self.audit()
        if (
            abs(after["unigram_mass"] - 1) > 1e-7
            or after["maximum_conditional_mass_error"] > 1e-7
        ):
            raise ResourceError("Support-floor LM is not normalized")
        return {
            "before": before,
            "after": after,
            "floor_mass": floor_mass,
            "rescaled_explicit_histories": rescaled,
            "policy": "Normalize unigrams, mix a uniform lexical support floor, preserve real explicit conditional probabilities where feasible, recompute backoff weights bottom-up. No artificial sentence or n-gram counts.",
        }

    def write(self, path):
        counts = Counter(map(len, self.probabilities))
        with Path(path).open("w") as stream:
            stream.write("\\data\\\n")
            for n in range(1, self.order + 1):
                stream.write(f"ngram {n}={counts[n]}\n")
            for n in range(1, self.order + 1):
                stream.write(f"\n\\{n}-grams:\n")
                for ws in sorted(k for k in self.probabilities if len(k) == n):
                    p = self.probabilities[ws]
                    if p <= 0 or not math.isfinite(p):
                        raise ResourceError("Invalid explicit LM probability")
                    row = f"{math.log10(p):.12g}\t{' '.join(ws)}"
                    if ws in self.backoffs:
                        row += f"\t{math.log10(self.backoffs[ws]):.12g}"
                    stream.write(row + "\n")
            stream.write("\n\\end\\\n")


def estimate(root, directory, texts, vocabulary, order=3):
    tool = read(root / ".cache/scn/tools/receipt.json")["tools"]["lmplz"]
    if sha(root / tool["path"]) != tool["sha256"]:
        raise ResourceError("KenLM estimator identity changed")
    identity = {
        "texts": digest(texts),
        "vocabulary": digest(sorted(vocabulary)),
        "order": order,
        "tool": tool,
        "prune": [0] * order,
        "implementation": sha(__file__),
    }
    receipt = directory / "estimate.json"
    if receipt.exists():
        saved = read(receipt)
        if saved["key"] != digest(identity):
            raise ResourceError("LM estimation inputs changed")
        verified_files(directory, saved)
        return directory / "estimated.arpa"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "training.txt").write_text("\n".join(texts) + "\n")
    (directory / "vocabulary.txt").write_text(
        "\n".join(sorted(set(vocabulary) - SPECIAL)) + "\n"
    )
    with (directory / "estimate.log").open("w") as log:
        subprocess.run(
            [
                str(root / tool["path"]),
                "-o",
                str(order),
                "--discount_fallback",
                "--prune",
                *(["0"] * order),
                "-S",
                "512M",
                "-T",
                str(directory),
                "--text",
                str(directory / "training.txt"),
                "--arpa",
                str(directory / "estimated.arpa"),
                "--limit_vocab_file",
                str(directory / "vocabulary.txt"),
            ],
            stdout=log,
            stderr=log,
            check=True,
            env={**os.environ, "OMP_NUM_THREADS": "2"},
        )
    seal(
        directory,
        receipt.name,
        identity,
        ["estimated.arpa", "training.txt", "vocabulary.txt"],
    )
    return directory / "estimated.arpa"


def joined_cues(rows, max_seconds=20, max_gap=0.75):
    """Join only adjacent admitted cue IDs, retaining a reversible group map."""
    groups, group = [], []
    for row in rows:
        index = int(row["id"].rsplit(":", 1)[1])
        if group:
            previous = group[-1]
            adjacent = (
                row["recording_id"] == previous["recording_id"]
                and index == int(previous["id"].rsplit(":", 1)[1]) + 1
                and -0.01 <= row["start_seconds"] - previous["end_seconds"] <= max_gap
                and row["end_seconds"] - group[0]["start_seconds"] <= max_seconds
            )
            if not adjacent:
                groups.append(group)
                group = []
        group.append(row)
    if group:
        groups.append(group)
    return [
        {
            "text": " ".join(normalize(r["text"]) for r in g),
            "cue_ids": [r["id"] for r in g],
            "recording_id": g[0]["recording_id"],
        }
        for g in groups
    ]


def build_models(root, bundle, output, recipe):
    directory = output / "lm"
    directory.mkdir(parents=True, exist_ok=True)
    identity = {
        "bundle": sha(bundle / "bundle.json"),
        "recipe": recipe["lm"],
        "implementation": sha(__file__),
        "source_study": recipe["corpus_study"],
    }
    receipt = directory / "models.receipt.json"
    if receipt.exists():
        saved = read(receipt)
        if saved["key"] != digest(identity):
            raise ResourceError("Residual LM inputs changed")
        verified_files(directory, saved)
        return read(directory / "models.json")
    vocabulary = set(read(bundle / "graph/lexicon_selected.json")) - SPECIAL
    channel = read(bundle / "channel.json")
    training = channel["training"]
    videos = sorted({r["recording_id"] for r in training})
    if len(videos) != 6 or any(r["split"] != "train" for r in training):
        raise ResourceError("LM floor selection requires six training recordings")
    folds, losses = [], defaultdict(lambda: [0.0, 0])
    import kenlm

    for video in videos:
        texts = list(
            dict.fromkeys(
                normalize(r["text"]) for r in training if r["recording_id"] != video
            )
        )
        validation = [
            normalize(r["text"]) for r in training if r["recording_id"] == video
        ]
        fold_dir = directory / "folds" / video
        estimated = estimate(root, fold_dir, texts, vocabulary)
        metrics = []
        for mass in recipe["lm"]["floor_masses"]:
            model = BackoffModel(estimated)
            audit = model.supported(vocabulary, mass)
            arpa = fold_dir / f"floor-{mass:g}.arpa"
            model.write(arpa)
            lm = kenlm.Model(str(arpa))
            nll = units = 0
            for text in validation:
                mapped = [w if w in vocabulary else "<unk>" for w in text.split()]
                nll -= lm.score(" ".join(mapped)) * math.log(10)
                units += len(mapped) + 1
            losses[mass][0] += nll
            losses[mass][1] += units
            metrics.append(
                {
                    "floor_mass": mass,
                    "negative_log_likelihood": nll,
                    "units": units,
                    "perplexity": math.exp(nll / units),
                    "audit": audit,
                }
            )
        folds.append(
            {
                "heldout": video,
                "training_recordings": [v for v in videos if v != video],
                "training_sha256": digest(texts),
                "validation_sha256": digest(validation),
                "metrics": metrics,
            }
        )
    selected = min(losses, key=lambda m: (losses[m][0] / losses[m][1], m))
    source = root / recipe["corpus_study"] / "lm/prune-0-0-0"
    models, audits = [], []
    channel_texts = list(dict.fromkeys(normalize(r["text"]) for r in training))
    heldout = [r["reference"] for r in read(bundle / "recordings.json")]
    for original in ("channel-o3", "converted-w50-o3", "raw-w50-o3"):
        parent = source / original
        saved = read(parent / "model.json")
        verified_files(parent, saved)
        lines = (parent / "training.txt").read_text().splitlines()
        prior = saved["identity"]["shared_unigram_prior"]
        if lines[-len(prior) :] != prior or set(prior) != vocabulary:
            raise ResourceError(
                "Cannot separate the matched source corpus from its artificial vocabulary sentences"
            )
        real = lines[: -len(prior)]
        name = original + "-support"
        target = directory / name
        estimated = estimate(root, target, real, vocabulary)
        model = BackoffModel(estimated)
        audit = model.supported(vocabulary, selected)
        model.write(target / "model.arpa")
        if (
            sum(v for k, v in audit["after"]["ngram_counts"].items() if int(k) > 1)
            > recipe["lm"]["maximum_context_ngrams"]
        ):
            raise ResourceError("Repaired LM exceeds the common context budget")
        stats = {
            "removed_artificial_sentences": len(prior),
            "real_sentences": len(real),
            "real_tokens": sum(len(t.split()) for t in real),
            "real_training_sha256": digest(real),
            "parent_receipt_sha256": sha(parent / "model.json"),
            "parent_weighting": saved["identity"]["weighting"],
            "audit": audit,
        }
        wiki_tokens = saved["identity"]["weighting"]["wikipedia_effective_tokens"]
        stats["effective_wikipedia_share_without_prior"] = (
            wiki_tokens / stats["real_tokens"]
        )
        write_json(target / "statistics.json", stats)
        models.append(
            {
                "id": name,
                "parent": original,
                "directory": str(target),
                "kind": original.split("-")[0],
                "boundary_mode": "original_cues",
                "statistics": stats,
                "heldout": evaluate_lm(
                    target / "model.arpa", heldout, vocabulary, channel_texts
                ),
            }
        )
        old = BackoffModel(parent / "model.arpa")
        audits.append(
            {
                "id": original,
                "audit": old.audit(),
                "artificial_sentence_count": len(prior),
                "real_sentence_count": len(real),
                "artificial_boundary_fraction": len(prior) / len(lines),
                "absent_in_real_corpus_words": len(
                    vocabulary - set(" ".join(real).split())
                ),
            }
        )
    # A boundary-only control keeps every retained cue token and its count.
    # This branch deliberately has no repetition reweighting or deduplication.
    for mode, rows in (
        (
            "cues",
            [{"text": normalize(r["text"]), "cue_ids": [r["id"]]} for r in training],
        ),
        ("joined", joined_cues(training)),
    ):
        target = directory / f"channel-boundaries-{mode}"
        estimated = estimate(root, target, [r["text"] for r in rows], vocabulary)
        model = BackoffModel(estimated)
        audit = model.supported(vocabulary, selected)
        model.write(target / "model.arpa")
        write_json(target / "groups.json", rows)
        models.append(
            {
                "id": target.name,
                "parent": "channel-boundary-control",
                "kind": "channel",
                "directory": str(target),
                "boundary_mode": mode,
                "audit": audit,
                "tokens": sum(len(r["text"].split()) for r in rows),
                "heldout": evaluate_lm(
                    target / "model.arpa", heldout, vocabulary, channel_texts
                ),
            }
        )
    result = {
        "status": "complete",
        "selected_floor_mass": selected,
        "training_folds": folds,
        "floor_selection": "Minimum pooled held-out training-recording token NLL, including EOS; no development text used.",
        "models": models,
        "original_lm_audit": audits,
        "baseline_lm_audit": BackoffModel(bundle / "graph/restricted.arpa").audit(),
        "boundary_policy": "Same admitted cue tokens/counts. Join consecutive cue IDs within one training recording, <=0.75-second gap and <=20-second group. Excluded IDs and recording boundaries always break groups.",
        "oov_policy": "Same vocabulary and <unk>; estimate intact source sequences with limit_vocab_file, then normalize all retained conditional distributions. Never bridge OOV tokens.",
    }
    write_json(directory / "models.json", result)
    artifacts = [
        str(p.relative_to(directory))
        for p in directory.rglob("*")
        if p.is_file() and p != receipt
    ]
    seal(directory, receipt.name, identity, artifacts)
    return result
