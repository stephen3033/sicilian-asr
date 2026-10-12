"""Temporary count-based diagnostics. Never exports a decoder LM or ARPA file."""

from collections import Counter, defaultdict
import json
import math
import re

from sicilian_asr.resources import ResourceError, fingerprints, write_json
from sicilian_asr.scn_acquisition import load_catalog
from sicilian_asr.scn_corpus import CATEGORIES


class ProbeLM:
    """Interpolated Witten–Bell with a normalized uniform base, including UNK."""

    def __init__(self, rows, max_order=4):
        self.max_order = max_order
        frequency = Counter(w for row in rows for w in row["tokens"])
        self.frequency = frequency
        self.vocabulary = {w for w, n in frequency.items() if n >= 2} | {"<unk>", "</s>"}
        self.counts = [Counter() for _ in range(max_order)]
        for row in rows:
            words = [w if w in self.vocabulary else "<unk>" for w in row["tokens"]]
            padded = ["<s>"] * (max_order - 1) + words + ["</s>"]
            for i in range(max_order - 1, len(padded)):
                for n in range(1, max_order + 1):
                    self.counts[n - 1][tuple(padded[i-n+1:i+1])] += 1
        self.totals, self.types = [], []
        for counter in self.counts:
            totals, types = Counter(), Counter()
            for gram, count in counter.items():
                totals[gram[:-1]] += count
                types[gram[:-1]] += 1
            self.totals.append(totals)
            self.types.append(types)

    def probabilities(self, history, word):
        value = 1 / len(self.vocabulary)
        values = []
        for n in range(1, self.max_order + 1):
            h = tuple(history[-(n-1):]) if n > 1 else ()
            total, types = self.totals[n-1][h], self.types[n-1][h]
            if total:
                value = (self.counts[n-1][h + (word,)] + types * value) / (total + types)
            values.append(value)
        return values

    def evaluate(self, rows, orders=(1, 2, 3, 4)):
        loss, unseen = Counter(), Counter()
        lexical, oov, predictions = 0, 0, 0
        for row in rows:
            words = row["tokens"]
            lexical += len(words)
            oov += sum(w not in self.vocabulary for w in words)
            padded = ["<s>"] * (self.max_order-1) + [w if w in self.vocabulary else "<unk>" for w in words] + ["</s>"]
            for i in range(self.max_order-1, len(padded)):
                probabilities = self.probabilities(padded[max(0,i-self.max_order+1):i], padded[i])
                for n in orders:
                    loss[n] -= math.log(probabilities[n-1])
                    unseen[n] += tuple(padded[i-n+1:i+1]) not in self.counts[n-1]
                predictions += 1
        return {"records": len(rows), "tokens": lexical, "oov_tokens": oov,
                "oov_rate": oov / lexical if lexical else None,
                "predictions_including_eos": predictions,
                "orders": {str(n): {"perplexity": math.exp(loss[n]/predictions) if predictions else None,
                                     "unseen_ngram_rate_after_unk_mapping": unseen[n]/predictions if predictions else None} for n in orders}}


def diagnose(rows, pronunciation_candidates=None):
    split = {s: [r for r in rows if r["split"] == s] for s in ("train", "development", "test")}
    counts = {s: {"records": len(v), "documents": len({r['document_id'] for r in v}),
                  "tokens": sum(len(r['tokens']) for r in v), "groups": len({r['split_group'] for r in v})} for s,v in split.items()}
    if not all(split.values()):
        return {"status": "insufficient_independent_splits", "splits": counts, "selected_order": None}
    model = ProbeLM(split["train"])
    dev = model.evaluate(split["development"])
    best = min(v["perplexity"] for v in dev["orders"].values())
    selected = min(int(n) for n, v in dev["orders"].items() if v["perplexity"] <= best * 1.01)
    ranked = [w for w,_ in model.frequency.most_common()]
    dev_words = [w for r in split["development"] for w in r["tokens"]]
    candidate_coverage = None if pronunciation_candidates is None else {
        "unvalidated_wikipron_word_types": len(pronunciation_candidates),
        "development_token_coverage": sum(w in pronunciation_candidates for w in dev_words) / len(dev_words),
        "development_type_coverage": len(set(dev_words) & pronunciation_candidates) / len(set(dev_words)),
        "dos24_or_pronunciation_correctness_verified": False,
    }
    coverage = {}
    for size in (500, 1000, 5000, 10000, 25000, 50000):
        vocab = set(ranked[:size])
        total = sum(len(r["tokens"]) for r in split["development"])
        hit = sum(w in vocab for r in split["development"] for w in r["tokens"])
        coverage[str(size)] = {"actual_vocabulary": len(vocab), "development_token_coverage": hit / total}
    return {"status": "auxiliary_diagnostic_only", "splits": counts,
            "training_vocabulary_types_min_count_2": len(model.vocabulary)-2,
            "training_raw_word_types": len(model.frequency), "training_singleton_word_types": sum(v==1 for v in model.frequency.values()),
            "ngram_sparsity": {str(n): {"types": len(c), "singleton_types": sum(v==1 for v in c.values()),
                                          "singleton_fraction": sum(v==1 for v in c.values())/len(c)} for n,c in enumerate(model.counts,1)},
            "development": dev, "selected_order": selected, "selected_order_test": model.evaluate(split["test"], (selected,)),
            "coverage_by_top_training_words": coverage,
            "pronunciation_candidate_coverage": candidate_coverage,
            "largest_document_group_records": max(Counter(r["split_group"] for r in rows).values()),
            "source_distribution": dict(Counter(r["source_id"] for r in rows))}


def probe(root):
    path = root / ".cache/scn/corpus/analyzed.jsonl"
    statistics = json.loads((root / "resources/scn/corpus/statistics.json").read_text())
    if fingerprints(path)["sha256"] != statistics["analyzed_sha256"]:
        raise ResourceError("Analyzed records differ from their receipt")
    catalog = load_catalog(root)["sources"]
    rows = []
    pronunciation_candidates = set()
    excluded = Counter()
    for line in path.open():
        r = json.loads(line)
        if catalog[r["source_id"]].get("extractor") == "ipa_tsv" and len(r["tokens"]) == 1:
            pronunciation_candidates.add(r["tokens"][0])
        reason = ("non_text_role" if r["data_role"] != "text" else "exact_duplicate" if r.get("duplicate_of")
                  else "transcript_validation_pending" if not catalog[r["source_id"]].get("probe_eligible", True)
                  else "pdf_layout_validation_pending" if catalog[r["source_id"]].get("extractor") == "pdf"
                  else "markup_or_list_review_pending" if len(r["tokens"]) > 250 or re.search(r"</?[A-Za-z]|\{\||\[\[|https?://", r["text"])
                  else "italian_screen_pending" if r["language_screen"]["screen"] != "unresolved" else None)
        if reason:
            excluded[reason] += 1
        else:
            # Keep only information needed by the temporary count models.
            rows.append({k:r[k] for k in ("id", "tokens", "source_id", "document_id", "split_group", "split", "orthography")})
    results = {}
    for category in CATEGORIES:
        print("PROBE " + category, flush=True)
        results[category] = diagnose([r for r in rows if r["orthography"] == category], pronunciation_candidates)
    # Two predeclared domain checks; remove any documents linked to the held source.
    holdout = {}
    auxiliary = [r for r in rows if r["orthography"] == "uncertain_mixed"]
    for source in ("scn_google_smoldoc", "scn_firefox_l10n"):
        held = [r for r in auxiliary if r["source_id"] == source]
        groups = {r["split_group"] for r in held}
        rest = [r for r in auxiliary if r["split_group"] not in groups and r["split"] == "train"]
        if held and rest:
            print("SOURCE HOLDOUT " + source, flush=True)
            holdout[source] = {"train_records":len(rest), "excluded_groups":len(groups), "evaluation": ProbeLM(rest).evaluate(held)}
        else:
            holdout[source] = {"status":"insufficient_independent_sources"}
    write_json(root / "resources/scn/lm/diagnostics.json", {
        "analyzed_sha256": statistics["analyzed_sha256"],
        "method": "Interpolated Witten-Bell orders 1-4; uniform base over train words count>=2 plus UNK/EOS; train-only vocabulary; EOS scored",
        "split": "SHA256(document/duplicate-connected group) mod 10: development=0, test=1, train=2..9; short exact duplicates removed but do not connect documents",
        "selection": "Lowest order within 1% of best development perplexity; test perplexity computed for selected order only",
        "final_lm_trained": False, "excluded_records":dict(excluded), "cohorts":results, "source_holdouts":holdout,
        "dos24_recommendation": "No empirical order selection possible until licensed, validated DOS24 text yields independent splits." if not any(r['orthography']=='dos24' for r in rows) else "Review DOS24 diagnostics and speaker/domain evaluation before final training.",
        "limitations": ["Auxiliary orthographies are never DOS24 substitutes", "Perplexities across different vocabularies/cohorts are not comparable", "These are count probes, not the existing Italian LM estimator", "Lexical screening cannot establish language purity", "MinHash duplicate detection is approximate"]})
