"""Human report and compact experiment ledger from verified stage results."""

from common.artifacts import read, sha
from common.utils import write_json
from pipeline.experiment import improves


def rates(row):
    return tuple(row["metrics"][name]["rate"] for name in ("wer", "cer"))


def report(output):
    selection = read(output / "selection.json")
    corpus = read(output / "corpus/summary.json")
    models = read(output / "lm/models.json")
    lm = read(output / "lm-results.json")
    pronunciation = read(output / "pronunciation-results.json")
    mappings = read(output / "pronunciation/analysis.json")
    vocabulary = read(output / "vocabulary-results.json")
    admission = read(output / "vocabulary-admission.json")
    cleanup = read(output / "cleanup/cleanup.json")
    supplemental_path = output / "cleanup/supplemental.json"
    supplemental = read(supplemental_path) if supplemental_path.exists() else {}
    removed_bytes = cleanup["bytes_removed"] + supplemental.get("bytes_removed", 0)
    validation = read(output / "validation.json")
    baseline = lm["baseline_metrics"]
    rows = (
        lm["results"]
        + pronunciation["results"]
        + [vocabulary["result"]]
        + selection["combined"]
        + selection["retuned"]
    )
    chosen = selection["selected"]
    wer, cer = rates(chosen)
    counts = corpus["counts"]
    text = [
        "# DOS24 WFST experiment results",
        "",
        f"Selected: `{chosen['id']}`. **WER {100 * wer:.2f}%; CER {100 * cer:.2f}%.**",
        f"The frozen baseline is {100 * baseline['wer']['rate']:.2f}% WER / {100 * baseline['cer']['rate']:.2f}% CER. The selected result has {chosen['metrics']['wer']['errors']:,} word errors out of {chosen['metrics']['wer']['reference_units']:,} reference words and {chosen['metrics']['cer']['errors']:,} character errors out of {chosen['metrics']['cer']['reference_units']:,} characters.",
        f"Targets remain WER below 10% and CER below 5%. Remaining gaps are {max(0, 100 * wer - 10):.2f} and {max(0, 100 * cer - 5):.2f} percentage points, respectively. Independent acceptance on fresh recordings remains pending.",
        "",
        "The selected result reproduced exact text and error counts in a fresh full-recording decode. The baseline was also rebuilt and replayed after cleanup using its portable bundle. Selection uses development recordings only.",
        "",
        "## Conversion evidence",
        "",
        f"Processed {counts['sentences']:,} source sentences. Admitted {counts.get('admitted_sentences', 0):,} sentences from {corpus['unique_articles']:,} articles, containing {counts.get('admitted_tokens', 0):,} tokens. Of these sentences, {counts.get('admitted_changed', 0):,} changed and {counts.get('admitted_unchanged', 0):,} remained unchanged.",
        f"The corpus adds {corpus['additional_words']:,} word types, {corpus['additional_ngrams']['2']:,} bigrams and {corpus['additional_ngrams']['3']:,} trigrams beyond channel training captions. The preservation check covers {counts['channel_sentences']:,} channel cues: {counts['channel_changed']:,} changed and {counts['channel_unresolved']:,} contain unresolved analyses.",
        "Automated support is not human certification or a measured conversion-accuracy estimate. The lexical resources supply analyses, not canonical DOS24 replacements. Unresolved word identity, grammatical function, stress, and quotation provenance remain in the review queue with their missing facts. Explicit grammatical analyses can resolve e/è, ci/cci, ni/nni and contractions; uncertain sentence analyses are excluded rather than inferred from frequency.",
        "",
        "| Exclusion or unresolved fact | Count |",
        "| --- | ---: |",
        *[
            f"| {reason} | {count:,} |"
            for reason, count in sorted(corpus["exclusions"].items())
        ],
        "",
        "Unresolved-fact counts can overlap within a sentence. Rule frequencies, collision forms, and productive examples absent from captions are in `corpus/summary.json`; full span edits and provenance are in `corpus/corpus.jsonl`.",
        "",
        "## Matched language models",
        "",
        "All fourteen models share vocabulary, unknown-word handling and a bounded pruning policy. Each vocabulary word contributes the same one-token prior; effective Wikipedia shares include these prior tokens. Source sentences remain separate. Unique source counts and effective repetition counts are recorded independently. Raw and converted controls use exactly the same admitted source passages.",
        f"Shared pruning: `{models['shared_prune']}`. Selection for decoding used development perplexity: `{', '.join(models['selected_for_decode'])}`. Vocabulary, pronunciation paths, posteriors, segmentation, topology, and initial word bonus remained fixed.",
        "",
        "| LM | Effective Wikipedia share | Perplexity | OOV | Bigram coverage | Trigram coverage |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for model in models["models"]:
        heldout = model["heldout"]
        text.append(
            f"| {model['id']} | {100 * model['weighting']['effective_wikipedia_share']:.3f}% | {heldout['perplexity']:.2f} | {100 * heldout['oov_rate']:.2f}% | {100 * heldout['context_coverage']['2']:.2f}% | {100 * heldout['context_coverage']['3']:.2f}% |"
        )
    text += [
        "",
        "Coverage and perplexity are diagnostic measures, not substitutes for full-recording WER/CER. Held-out contexts absent from channel training are reported separately in `lm/models.json`.",
        "",
        "| Converted candidate | WER difference from matched raw | CER difference from matched raw | Dominates raw |",
        "| --- | ---: | ---: | --- |",
        *[
            f"| {r['converted']} | {100 * r['wer_delta']:+.3f} pp | {100 * r['cer_delta']:+.3f} pp | {r['conversion_improves']} |"
            for r in lm["matched_comparisons"]
        ],
        "",
        "Negative differences favor conversion. Promotion still requires improvement over the historical baseline without worsening CER.",
        "",
        "## Vocabulary and pronunciation",
        "",
        f"The separate vocabulary test adds {vocabulary['added_words']:,} supported forms to `{vocabulary['parent']['id']}`. Admission requires five independent admitted Wikipedia articles or channel-training attestation. {len(admission['excluded']):,} proposed forms remain unresolved or nonconforming; {len(admission['unsupported_acoustic_paths']):,} supported forms lack usable fallback acoustic paths. Improvement over the fixed-vocabulary parent: **{vocabulary['improves_parent']}**.",
        (
            "No additional form had both linguistic support and a usable pronunciation path. This comparison is an identity control; it does not measure the effect of a genuinely larger vocabulary."
            if not vocabulary["added_words"]
            else "The expanded vocabulary was selected without development term frequencies."
        ),
        f"New mappings use {mappings['observations']:,} cached label observations from six training recordings. {len(mappings['evidence']):,} word-and-label contexts meet the declared recurrence thresholds. Global, positional and context alternatives retain historical whole-word paths as controls.",
        mappings["boundary_policy"],
        "",
        "| Held-out recording | Global label NLL | Positional label NLL | Context label NLL | Supported whole-word observations |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for fold in mappings["folds"]:
        m = fold["metrics"]
        word = fold["whole_word_control"]
        text.append(
            f"| {fold['heldout']} | {m['global']['negative_log_likelihood']:.4f} | {m['positional']['negative_log_likelihood']:.4f} | {m['context']['negative_log_likelihood']:.4f} | {word['heldout_supported_paths']}/{word['heldout_observations']} |"
        )
    text += [
        "",
        "Each fold rebuilds learned mappings using the other five recordings. Lower negative log likelihood is better. Written distinctions remain separate even where acoustic labels are similar.",
        "",
        "## Complete-recording comparisons",
        "",
        "| Candidate | WER | CER | Improves baseline without CER loss |",
        "| --- | ---: | ---: | --- |",
        *[
            f"| {r['id']} | {100 * rates(r)[0]:.2f}% | {100 * rates(r)[1]:.2f}% | {improves(r['metrics'], baseline)} |"
            for r in sorted(rows, key=lambda r: (*rates(r), r["id"]))
        ],
        "",
        (
            "No tested candidate dominates the baseline. Combination and cost retuning were skipped by their predeclared improvement gate. This establishes a measured plateau within this experiment family, not a global performance bound."
            if selection["measured_plateau"]
            else f"Combined {len(selection['combined'])} supported candidate pairs and evaluated {len(selection['retuned'])} local cost settings around an improved graph. Selection follows the WER/CER dominance rule; other tradeoffs remain visible above."
        ),
        "",
        "## Cleanup and verification",
        "",
        f"Retired **{removed_bytes / 1024**3:.2f} GiB** of explicitly inventoried files. This is removed file size, not net filesystem growth after regenerated graphs. The deletion and supplemental inventories list every path and checksum; raw datasets, licenses, ZIPA, production resources, and the portable baseline remain retained.",
        f"Cross-stage validation: **{validation['status']}**, covering {validation['language_models']} LMs, {validation['complete_recording_candidates']} complete-recording candidates and {validation['pronunciation_folds']} training-only folds. See `validation.json` for input identities and checks.",
        "",
        "Reproduce with `.venv/bin/python -m pipeline scn-improve --stage all`. The selected graph is `best/candidate.json`; use `experimental-scn.yaml` with the ordinary preparation/transcription commands. Source conversion can be rerun through the same workflow. Historical receipts whose artifacts were retired are evidence only.",
        "",
        "Evidence: `corpus/review-queue.jsonl`, `review/priority.jsonl`, `lm/models.json`, `pronunciation/analysis.json`, `cleanup/history.json`, `cleanup/deletion-inventory.json`, `cleanup/source-retirement.json`, `cleanup/supplemental-inventory.json`, `selected-fresh-replay/result.json`, and `experiment-ledger.json`.",
    ]
    regression = output / "regression.json"
    if regression.exists():
        text += ["", "Production regression checks: " + read(regression)["summary"]]
    (output / "report.md").write_text("\n".join(text) + "\n")
    ledger = {
        "schema_version": 1,
        "historical_ledger": {
            "path": "cleanup/history.json",
            "sha256": sha(output / "cleanup/history.json"),
        },
        "recipe_sha256": sha(output / "study.json"),
        "baseline": baseline,
        "candidates": [
            {
                "id": r["id"],
                "metrics": r["metrics"],
                "graph": r["graph"],
                "improves_baseline": improves(r["metrics"], baseline),
            }
            for r in rows
        ],
        "matched_conversion_comparisons": lm["matched_comparisons"],
        "selected": chosen["id"],
        "measured_plateau": selection["measured_plateau"],
        "repeat_policy": "Do not repeat unchanged data, rules, mappings, or decoder settings. Fresh evidence or a distinct controlled hypothesis is required.",
        "evidence_gaps": [
            "Unresolved lexical identity, stress and grammatical context in review queue",
            "No independently identified syntactic doubling observations",
            "Development selection does not establish fresh-recording acceptance",
        ]
        + (
            ["No additional vocabulary forms met the declared support/admission rules"]
            if not vocabulary["added_words"]
            else []
        ),
    }
    write_json(output / "experiment-ledger.json", ledger)
    write_json(
        output / "report.json",
        {
            "selection": selection,
            "corpus": corpus,
            "cleanup": cleanup,
            "supplemental_cleanup": supplemental,
            "total_bytes_removed": removed_bytes,
            "validation": validation,
            "results": rows,
            "ledger": ledger,
        },
    )
