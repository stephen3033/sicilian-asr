"""Versioned DOS24 corpus, language-model and pronunciation experiment runner."""

import argparse
import json
import shutil
from pathlib import Path

from asr.scn.dataset import digest
from asr.scn.orthography import normalize
from common.errors import ResourceError
from common.utils import write_json
from pipeline.experiment import (
    compile_graph,
    decode_bundle,
    improves,
    read,
    seal,
    sha,
)

DOS24_STAGES = (
    "export",
    "replay",
    "cleanup-inventory",
    "cleanup",
    "convert",
    "label-analysis",
    "lm",
    "evaluate",
    "vocabulary",
    "pronunciation",
    "combine",
    "validate",
    "report",
)

RESIDUAL_STAGES = (
    "residual-prepare",
    "residual-score",
    "residual-models",
    "residual-calibrate",
    "residual-audio",
    "residual-repairs",
    "residual-tests",
    "residual-cleanup",
    "residual-validate",
    "residual-report",
)
STAGES = (*DOS24_STAGES, *RESIDUAL_STAGES)


def completed(path):
    result = read(path)
    if result.get("status") != "complete":
        raise ResourceError(f"Prerequisite stage is incomplete: {path}")
    return result


def candidate(bundle, output, label, lexicon, arpa, compile_settings, decoder):
    tokens = read(bundle / "graph/compile_input.json")["tokens"]
    graph = compile_graph(
        output / "graphs" / label, lexicon, arpa, tokens, compile_settings
    )
    result = decode_bundle(bundle, graph, output / "decodes" / label, decoder)
    return {
        "id": label,
        "status": "complete",
        "graph": str(graph),
        "decoder": decoder,
        "compile_settings": compile_settings,
        "metrics": result["metrics"],
        "result": str(output / "decodes" / label / "result.json"),
    }


def evaluate(bundle, output, recipe):
    study = completed(output / "lm/models.json")
    models = {m["id"]: m for m in study["models"]}
    lexicon = read(bundle / "graph/lexicon_selected.json")
    base_settings = read(bundle / "graph/compile_input.json")["settings"]
    base_decoder = read(bundle / "decoder.json")
    results = []
    for model_id in study["selected_for_decode"]:
        model = models[model_id]
        for scale in recipe["lm_scales"]:
            label = f"{model_id}-lm{scale:g}"
            settings = {**base_settings, "lm_order": model["order"], "lm_scale": scale}
            print("Decode", label, flush=True)
            row = candidate(
                bundle,
                output,
                label,
                lexicon,
                Path(model["directory"]) / "model.arpa",
                settings,
                {**base_decoder, "lm_scale": scale},
            )
            results.append(
                {**row, "lm": model_id, "kind": model["kind"], "share": model["share"]}
            )
            write_json(
                output / "lm-results.json", {"status": "running", "results": results}
            )
    pairs = []
    for converted in (r for r in results if r["kind"] == "converted"):
        raw = next(
            r for r in results if r["id"] == converted["id"].replace("converted", "raw")
        )
        pairs.append(
            {
                "converted": converted["id"],
                "raw": raw["id"],
                "conversion_improves": improves(converted["metrics"], raw["metrics"]),
                "wer_delta": converted["metrics"]["wer"]["rate"]
                - raw["metrics"]["wer"]["rate"],
                "cer_delta": converted["metrics"]["cer"]["rate"]
                - raw["metrics"]["cer"]["rate"],
            }
        )
    report = {
        "status": "complete",
        "results": results,
        "matched_comparisons": pairs,
        "baseline_metrics": read(bundle / "expected.json")["metrics"],
        "selection_scope": "Development recordings only; no independent acceptance claim.",
    }
    write_json(output / "lm-results.json", report)
    return report


def rank(row):
    return row["metrics"]["wer"]["rate"], row["metrics"]["cer"]["rate"], row["id"]


def vocabulary(bundle, output, recipe, root):
    import sentencepiece as spm

    from asr.scn.dos24 import Converter
    from asr.scn.lexicon import build
    from pipeline.scn_lm_experiment import train_model

    parent = min(
        (
            r
            for r in completed(output / "lm-results.json")["results"]
            if r["kind"] == "converted"
        ),
        key=rank,
    )
    model = next(
        m for m in read(output / "lm/models.json")["models"] if m["id"] == parent["lm"]
    )
    lexicon = read(Path(parent["graph"]) / "lexicon_selected.json")
    support = read(output / "corpus/article-support.json")
    converter = Converter(read(output / "corpus/index.json")["analyses"])
    channel = read(bundle / "channel.json")
    training_words = {
        w
        for r in channel["recordings"]
        if r["split"] == "train"
        for w in normalize(r["text"]).split()
    }
    proposed = {
        w
        for w, articles in support.items()
        if len(articles) >= recipe["minimum_articles"]
    } | training_words
    eligible, excluded = [], []
    for word in sorted(proposed - set(lexicon)):
        converted, _, _, issue = converter.word(word)
        if issue or converted != word:
            excluded.append(
                {
                    "word": word,
                    "reason": "unresolved_or_nonconforming",
                    "analysis": issue,
                }
            )
        else:
            eligible.append(word)
    tokens = read(bundle / "graph/compile_input.json")["tokens"]
    tokenizer = spm.SentencePieceProcessor(model_file=str(bundle / "tokenizer.model"))
    additions, evidence = build(eligible, {}, tokens, tokenizer)
    expanded = {**lexicon, **additions}
    write_json(
        output / "vocabulary-admission.json",
        {
            "eligible": eligible,
            "added": sorted(additions),
            "excluded": excluded,
            "unsupported_acoustic_paths": evidence["excluded"],
            "pronunciations": evidence,
            "policy": "At least five independent admitted Wikipedia articles or channel-training attestation. No development term frequencies.",
        },
    )
    admitted = [json.loads(line) for line in (output / "corpus/admitted.jsonl").open()]
    texts = list(dict.fromkeys(normalize(r["text"]) for r in channel["training"]))
    directory = output / "lm/expanded"
    prune = read(output / "lm/models.json")["shared_prune"]
    train_model(
        root,
        directory,
        texts,
        [r["converted"] for r in admitted],
        model["share"],
        model["order"],
        expanded,
        prune,
    )
    stats = read(directory / "statistics.json")
    if (
        sum(v for k, v in stats["ngram_counts"].items() if int(k) > 1)
        > recipe["maximum_context_ngrams"]
    ):
        raise ResourceError("Expanded-vocabulary LM exceeds shared n-gram budget")
    result = candidate(
        bundle,
        output,
        "expanded-vocabulary",
        expanded,
        directory / "model.arpa",
        parent["compile_settings"],
        parent["decoder"],
    )
    report = {
        "status": "complete",
        "parent": parent,
        "result": result,
        "added_words": len(additions),
        "improves_parent": improves(result["metrics"], parent["metrics"]),
    }
    write_json(output / "vocabulary-results.json", report)
    return report


def pronunciation(bundle, output):
    analysis = completed(output / "pronunciation/analysis.json")
    lexicon = read(bundle / "graph/lexicon_selected.json")
    base = read(bundle / "graph/compile_input.json")["settings"]
    decoder = read(bundle / "decoder.json")
    global_settings = {
        **base,
        "phone_alternatives": analysis["global_alternatives"],
        "phone_seed_costs": analysis["global_seed_costs"],
        "phone_position_costs": None,
    }
    settings = {
        "channel-global": global_settings,
        "channel-positional": {
            **global_settings,
            "phone_position_costs": analysis["positional_costs"],
        },
        "channel-context": {
            **base,
            "phone_context_costs": analysis["phone_context_costs"],
        },
    }
    results = []
    for name, value in settings.items():
        results.append(
            candidate(
                bundle,
                output,
                name,
                lexicon,
                bundle / "graph/restricted.arpa",
                value,
                decoder,
            )
        )
        write_json(
            output / "pronunciation-results.json",
            {"status": "running", "results": results},
        )
    report = {
        "status": "complete",
        "results": results,
        "control": read(bundle / "expected.json")["metrics"],
        "policy": "Frozen baseline LM/vocabulary and existing whole-word paths. New global, positional and written-word context costs use only the six training recordings.",
    }
    write_json(output / "pronunciation-results.json", report)
    return report


def export_candidate(bundle, graph, decoder, result, destination):
    destination.mkdir(parents=True, exist_ok=True)
    shutil.copytree(graph, destination / "graph", dirs_exist_ok=True)
    write_json(
        destination / "vocabulary.json",
        {"words": sorted(read(graph / "lexicon_selected.json"))},
    )
    write_json(
        destination / "pronunciation-settings.json",
        {
            key: value
            for key, value in read(graph / "compile_input.json")["settings"].items()
            if key.startswith("phone_")
            or key
            in {"pronunciation_settings", "pronunciation_variants", "contractions"}
        },
    )
    # Preserve the evidence/audio bundle as a distinct portable artifact; candidate
    # contents themselves are sufficient for compilation and fresh transcription.
    write_json(destination / "decoder.json", decoder)
    write_json(
        destination / "selection.json",
        {
            "result": result,
            "comparison_bundle_sha256": sha(bundle / "bundle.json"),
            "independent_acceptance": "pending_fresh_recordings",
        },
    )
    seal(
        destination,
        "candidate.json",
        {
            "schema_version": 1,
            "baseline": sha(bundle / "bundle.json"),
            "decoder": decoder,
        },
        [
            str(p.relative_to(destination))
            for p in destination.rglob("*")
            if p.is_file() and p.name != "candidate.json"
        ],
    )


def combine(root, bundle, output, recipe):
    baseline_metrics = read(bundle / "expected.json")["metrics"]
    baseline = {
        "id": "baseline",
        "graph": str(bundle / "graph"),
        "metrics": baseline_metrics,
        "decoder": read(bundle / "decoder.json"),
        "compile_settings": read(bundle / "graph/compile_input.json")["settings"],
    }
    lm = completed(output / "lm-results.json")
    vocab = completed(output / "vocabulary-results.json")
    pron = completed(output / "pronunciation-results.json")
    conversion_winners = {
        p["converted"] for p in lm["matched_comparisons"] if p["conversion_improves"]
    }
    # Stronger raw or channel controls are measured too; promotion always obeys
    # the WER/CER dominance condition rather than a weighted aggregate score.
    candidates = [
        r
        for r in lm["results"]
        if r["kind"] != "converted" or r["id"] in conversion_winners
    ]
    if vocab["improves_parent"] and vocab["parent"]["id"] in conversion_winners:
        candidates.append(vocab["result"])
    candidates += pron["results"]
    winners = [r for r in candidates if improves(r["metrics"], baseline_metrics)]
    selected = min(winners, key=rank) if winners else baseline
    lm_winners = [
        r
        for r in candidates
        if r not in pron["results"] and improves(r["metrics"], baseline_metrics)
    ]
    pron_winners = [
        r for r in pron["results"] if improves(r["metrics"], baseline_metrics)
    ]
    combined, tuned = [], []
    if lm_winners and pron_winners:
        language_parent, pronunciation_parent = (
            min(lm_winners, key=rank),
            min(pron_winners, key=rank),
        )
        settings = {
            **language_parent["compile_settings"],
            **{
                k: v
                for k, v in pronunciation_parent["compile_settings"].items()
                if k.startswith("phone_")
            },
        }
        row = candidate(
            bundle,
            output,
            "combined",
            read(Path(language_parent["graph"]) / "lexicon_selected.json"),
            Path(language_parent["graph"]) / "restricted.arpa",
            settings,
            language_parent["decoder"],
        )
        combined.append(row)
        if improves(row["metrics"], selected["metrics"]):
            selected = row
    # Cost search is conditional on a newly improved graph, never a rerun of
    # unchanged historical sweeps. This is bounded by the versioned recipe.
    if selected["id"] != "baseline":
        parent = selected
        for delta in recipe["lm_scale_offsets"]:
            for bonus_delta in recipe["word_bonus_offsets"]:
                if delta == bonus_delta == 0:
                    continue
                settings = {
                    **parent["compile_settings"],
                    "lm_scale": round(
                        parent["compile_settings"]["lm_scale"] + delta, 6
                    ),
                    "word_bonus": parent["compile_settings"].get("word_bonus", 0)
                    + bonus_delta,
                }
                label = f"retune-lm{settings['lm_scale']:g}-b{settings['word_bonus']:g}"
                row = candidate(
                    bundle,
                    output,
                    label,
                    read(Path(parent["graph"]) / "lexicon_selected.json"),
                    Path(parent["graph"]) / "restricted.arpa",
                    settings,
                    {
                        **parent["decoder"],
                        "lm_scale": settings["lm_scale"],
                        "word_bonus": settings["word_bonus"],
                    },
                )
                tuned.append(row)
                if improves(row["metrics"], selected["metrics"]):
                    selected = row
    fresh = decode_bundle(
        bundle,
        Path(selected["graph"]),
        output / "selected-fresh-replay",
        selected["decoder"],
        fresh=True,
    )
    expected = (
        read(bundle / "expected.json")
        if selected["id"] == "baseline"
        else read(selected["result"])
    )
    if fresh["metrics"] != selected["metrics"] or {
        r["video"]: r["text"] for r in fresh["recordings"]
    } != {r["video"]: r["text"] for r in expected["recordings"]}:
        raise ResourceError(
            "Selected candidate did not reproduce full-recording scores/text"
        )
    export_candidate(
        bundle, Path(selected["graph"]), selected["decoder"], selected, output / "best"
    )
    import yaml

    config = {
        "languages": ["scn"],
        "output_language": "scn",
        "resources": {
            "scn": {"candidate_bundle_path": str((output / "best").relative_to(root))}
        },
        "decoder": {
            **selected["decoder"],
            "graph_path": str((output / "prepared-best").relative_to(root)),
        },
    }
    (output / "experimental-scn.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True)
    )
    report = {
        "status": "complete",
        "selected": selected,
        "combined": combined,
        "retuned": tuned,
        "fresh_replay": str(output / "selected-fresh-replay/result.json"),
        "measured_plateau": not winners,
        "scope": "Only the declared experiment family, not a global performance bound.",
        "targets_met": selected["metrics"]["wer"]["rate"] < 0.1
        and selected["metrics"]["cer"]["rate"] < 0.05,
    }
    write_json(output / "selection.json", report)
    return report


def report(output):
    from pipeline.scn_reporting import report as write_report

    return write_report(output)


def run(root, output, recipe, stage):
    bundle = root / recipe["baseline_bundle"]
    if recipe.get("workflow") == "residual-diagnosis":
        from pipeline.scn_residual import run as run_residual

        return run_residual(root, bundle, output, recipe, stage)
    if stage != "all" and stage not in DOS24_STAGES:
        raise ResourceError("Residual stages require a residual-diagnosis recipe")
    output.mkdir(parents=True, exist_ok=True)
    if (bundle / "bundle.json").exists():
        actual = {
            split: sorted(
                r["recording_id"]
                for r in read(bundle / "channel.json")["recordings"]
                if r["split"] == split
            )
            for split in recipe["splits"]
        }
        if actual != {
            split: sorted(videos) for split, videos in recipe["splits"].items()
        }:
            raise ResourceError("Recipe splits differ from the frozen recording bundle")
        identity = {"recipe": recipe, "baseline_sha256": sha(bundle / "bundle.json")}
        if (output / "study.json").exists() and read(output / "study.json")[
            "identity"
        ] != identity:
            raise ResourceError(
                "Experiment recipe or baseline changed; use a new output directory"
            )
        write_json(
            output / "study.json", {"key": digest(identity), "identity": identity}
        )
    write_json(output / "recipes" / (digest(recipe) + ".json"), recipe)
    stages = DOS24_STAGES if stage == "all" else (stage,)
    for current in stages:
        print("Stage", current, flush=True)
        if current == "export":
            from pipeline.scn_bundle import export_baseline

            export_baseline(root, bundle, root / recipe["legacy_directory"])
        elif current == "replay":
            from pipeline.scn_bundle import replay_baseline

            replay_baseline(bundle, output / "baseline-replay")
        elif current == "cleanup-inventory":
            from pipeline.scn_cleanup import inventory

            if not (output / "cleanup/cleanup.json").exists():
                inventory(
                    root, bundle, output / "cleanup", root / recipe["legacy_directory"]
                )
        elif current == "cleanup":
            from pipeline.scn_cleanup import apply_inventory

            if (
                not (output / "cleanup/cleanup.json").exists()
                or read(output / "cleanup/cleanup.json")["status"] != "complete"
            ):
                apply_inventory(root, bundle, output / "cleanup")
        elif current == "convert":
            from asr.scn.dos24_corpus import prepare_corpus
            from pipeline.scn_review import prioritize

            prepare_corpus(root, bundle, output / "corpus", recipe["corpus"])
            prioritize(bundle, output / "corpus", output / "review")
        elif current == "label-analysis":
            from pipeline.scn_pronunciation import analyze

            analyze(bundle, output / "pronunciation", recipe["pronunciation"])
        elif current == "lm":
            from pipeline.scn_lm_experiment import train_controls

            train_controls(root, output / "lm", bundle, output / "corpus", recipe["lm"])
        elif current == "evaluate":
            evaluate(bundle, output, recipe["lm"])
        elif current == "vocabulary":
            vocabulary(bundle, output, recipe["vocabulary"], root)
        elif current == "pronunciation":
            pronunciation(bundle, output)
        elif current == "combine":
            combine(root, bundle, output, recipe["retuning"])
        elif current == "validate":
            from pipeline.scn_validation import validate

            validate(root, bundle, output, recipe)
        elif current == "report":
            report(output)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path)
    parser.add_argument("--config", type=Path, default=Path("configs/scn-improve.json"))
    parser.add_argument("--stage", choices=("all", *STAGES), default="all")
    args = parser.parse_args(argv)
    root = args.root.resolve()
    recipe = read(root / args.config)
    if (
        recipe.get("schema_version") != 2
        or recipe.get("architecture_scope") != "zipa_wfst"
    ):
        raise ResourceError("Expected version-2 ZIPA/WFST experiment recipe")
    selected_output = args.output or Path(
        recipe.get("output_directory", "outputs/scn_dos24")
    )
    run(root, (root / selected_output).resolve(), recipe, args.stage)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
