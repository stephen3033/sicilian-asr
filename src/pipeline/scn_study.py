"""Cache independent acoustic inference and select/evaluate Sicilian decoders."""

import gc
import json
import platform
import resource
import time
from collections import defaultdict
from copy import deepcopy

import numpy as np
import torch

from asr.scn.dataset import digest, reviewed_text, video_record
from asr.scn.evaluation import evaluate
from asr.scn.prepare import prepare
from asr.wfst.cache import verify_cache
from asr.wfst.cached import decode_video
from asr.wfst.storage import tensor_path
from audio.segmentation import segment_audio
from common.environment import package_versions
from common.errors import ResourceError
from common.resources import load_resources
from common.settings import SAMPLE_RATE, SCN_RESOURCES
from common.utils import fingerprints, write_json
from ipa.diagnostics import greedy_ctc
from ipa.zipa import ZipaAcoustic
from media_input.media import extract_audio
from pipeline.config import load_config


def peak_rss_bytes():
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak if platform.system() == "Darwin" else peak * 1024


def caption_coverage(cues, duration):
    """Report union coverage, without assuming subtitle gaps are silent."""
    merged = []
    for cue in sorted(cues, key=lambda c: c["start_seconds"]):
        start = max(0, min(duration, cue["start_seconds"]))
        end = max(start, min(duration, cue["end_seconds"]))
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    gaps, cursor = [], 0
    for start, end in merged:
        if start > cursor:
            gaps.append([cursor, start])
        cursor = end
    if cursor < duration:
        gaps.append([cursor, duration])
    covered = sum(end - start for start, end in merged)
    return {
        "audio_seconds": duration,
        "caption_seconds": covered,
        "fraction": covered / duration if duration else 0,
        "uncaptioned_intervals_seconds": gaps,
        "excluded_audio_intervals_seconds": [],
        "scoring_policy": "Full audio against complete caption text; gaps are retained. Subtitle omissions, paraphrases and foreign examples affect WER. Non-speech annotations may be removed only by recorded reference review.",
    }


def read_result(directory, key):
    receipt = json.loads((directory / "receipt.json").read_text())
    if receipt["key"] != key:
        raise ResourceError("Decode cache identity mismatch")
    for name, expected in receipt["artifacts"].items():
        if fingerprints(directory / name)["sha256"] != expected:
            raise ResourceError(f"Changed decode artifact: {name}")
    return json.loads((directory / "result.json").read_text())


def posteriors(root, video, config, manifest):
    source = root / ".cache/scn/youtube" / video
    acquired = json.loads((source / "receipt.json").read_text())
    audio = source / acquired["audio"]
    expected = acquired["artifacts"][acquired["audio"]]
    if fingerprints(audio)["sha256"] != expected:
        raise ResourceError("Audio differs from acquired snapshot")
    identity = {
        "audio": expected,
        "acoustic": config["acoustic"],
        "segmentation": config["audio"],
        "versions": package_versions(),
        "resources": {
            k: v.get("sha256")
            for k, v in manifest["resources"].items()
            if k
            in {
                "zipa_checkpoint",
                "zipa_tokens",
                "zipa_tokenizer",
                "zipa_source",
                "icefall_source",
            }
        },
        "sources": {
            p: fingerprints(root / "src" / p)["sha256"]
            for p in (
                "ipa/zipa.py",
                "ipa/types.py",
                "audio/segmentation.py",
                "media_input/media.py",
            )
        },
    }
    key = digest(identity)
    directory = root / ".cache/scn/posteriors" / key
    receipt_path = directory / "acoustic.json"
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text())
        if receipt["key"] != key:
            raise ResourceError("Acoustic cache identity mismatch")
        for name, expected in receipt["artifacts"].items():
            if fingerprints(directory / name)["sha256"] != expected:
                raise ResourceError(f"Acoustic cache changed: {name}")
        return directory, receipt
    directory.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(config["acoustic"].get("threads", 4))
    started = time.perf_counter()
    samples, info = extract_audio(audio, directory / "audio.wav", None)
    segments = segment_audio(samples, config["audio"], info["timestamp_origin_seconds"])
    model = ZipaAcoustic(root, config["acoustic"], manifest)
    entries = []
    for segment in segments:
        print(f"ZIPA {video}: {segment.index + 1}/{len(segments)}", flush=True)
        result = model.infer(
            [
                torch.from_numpy(
                    samples[segment.start_sample : segment.end_sample].copy()
                )
            ]
        )
        name = f"{segment.index:05d}.npz"
        np.savez_compressed(
            directory / name,
            log_probs=result.log_probs[0, : int(result.lengths[0])].numpy(),
            tokens=np.asarray(result.tokens),
        )
        entries.append(
            {**segment.to_dict(), "path": name, "greedy_ipa": greedy_ctc(result)[0]}
        )
    receipt = {
        "key": key,
        "identity": identity,
        "video": video,
        "segments": entries,
        "duration_seconds": len(samples) / SAMPLE_RATE,
        "seconds": time.perf_counter() - started,
        "process_peak_rss_bytes": peak_rss_bytes(),
        "artifacts": {
            p.name: fingerprints(p)["sha256"]
            for p in directory.iterdir()
            if p.is_file()
        },
    }
    write_json(receipt_path, receipt)
    del model
    gc.collect()
    return directory, receipt


def evaluate_candidate(root, output, config, manifest, candidate, references, scales):
    config = deepcopy(config)
    config["resources"]["scn"]["model_receipt_path"] = candidate["receipt"]
    directory = prepare(root, config, manifest)
    prepared = verify_cache(directory, directory.name)
    vocabulary = set(json.loads((directory / "vocabulary.json").read_text())["words"])
    trials = []
    for scale in scales:
        settings = {**config["decoder"], "acoustic_scale": scale}
        reports = []
        for reference in references:
            video = reference["document_id"]
            audio_dir, acoustic = posteriors(root, video, config, manifest)
            trial_key = digest(
                {
                    "graph": prepared["key"],
                    "acoustic": acoustic["key"],
                    "settings": settings,
                    "reference": reference["normalized"],
                    "study_source": fingerprints(root / "src/pipeline/scn_study.py")[
                        "sha256"
                    ],
                    "evaluation_source": fingerprints(
                        root / "src/asr/scn/evaluation.py"
                    )["sha256"],
                    "scoring_source": fingerprints(root / "src/evaluation/metrics.py")[
                        "sha256"
                    ],
                    "orthography_source": fingerprints(
                        root / "src/asr/scn/orthography.py"
                    )["sha256"],
                }
            )
            result_dir = output / "decodes" / trial_key
            if not (result_dir / "receipt.json").exists():
                result_dir.mkdir(parents=True, exist_ok=True)
                decoded = decode_video(directory, audio_dir, acoustic, settings)
                metrics, comparison = evaluate(
                    decoded["text"],
                    {"text": reference["normalized"]},
                    vocabulary,
                    vocabulary,
                )
                decoded.update(
                    video=video,
                    metrics=metrics,
                    acoustic_key=acoustic["key"],
                    graph_key=prepared["key"],
                    reference_sha256=digest(reference),
                    acoustic_scale=scale,
                    coverage=caption_coverage(
                        reference["cues"], acoustic["duration_seconds"]
                    ),
                    acoustic_seconds=acoustic["seconds"],
                )
                write_json(result_dir / "result.json", decoded)
                (result_dir / "comparison.txt").write_text(comparison)
                (result_dir / "transcription.txt").write_text(decoded["text"] + "\n")
                write_json(
                    result_dir / "receipt.json",
                    {
                        "key": trial_key,
                        "artifacts": {
                            name: fingerprints(result_dir / name)["sha256"]
                            for name in (
                                "result.json",
                                "comparison.txt",
                                "transcription.txt",
                            )
                        },
                    },
                )
            reports.append(
                {
                    "path": str(result_dir.relative_to(root)),
                    **read_result(result_dir, trial_key),
                }
            )
        errors = sum(r["metrics"]["wer"]["errors"] for r in reports)
        words = sum(r["metrics"]["wer"]["reference_units"] for r in reports)
        char_errors = sum(r["metrics"]["cer"]["errors"] for r in reports)
        chars = sum(r["metrics"]["cer"]["reference_units"] for r in reports)
        trial = {
            **candidate,
            "graph": str(directory.relative_to(root)),
            "acoustic_scale": scale,
            "wer": errors / words if words else None,
            "cer": char_errors / chars if chars else None,
            "word_errors": errors,
            "reference_words": words,
            "character_errors": char_errors,
            "reference_characters": chars,
            "decode_seconds": sum(r["decode_seconds"] for r in reports),
            "acoustic_seconds": sum(r["acoustic_seconds"] for r in reports),
            "process_peak_rss_bytes": max(r["process_peak_rss_bytes"] for r in reports),
            "oov_rate": sum(
                r["metrics"]["oov_decoding_vocabulary"]["tokens"] for r in reports
            )
            / words
            if words
            else None,
            "oov_tokens": sum(
                r["metrics"]["oov_decoding_vocabulary"]["tokens"] for r in reports
            ),
            "results": [
                {
                    k: r[k]
                    for k in (
                        "video",
                        "path",
                        "metrics",
                        "coverage",
                        "decode_seconds",
                        "acoustic_seconds",
                    )
                }
                for r in reports
            ],
            "graph_bytes": tensor_path(directory).stat().st_size,
        }
        print({k: v for k, v in trial.items() if k != "results"}, flush=True)
        trials.append(trial)
    return trials


def development_study(root, output, recipe, config, manifest):
    if (output / "test.json").exists():
        raise ResourceError(
            "Final tests already run in this study; start a new output directory for another development experiment"
        )
    candidates = json.loads((output / "candidates.json").read_text())
    references = json.loads((output / "development.json").read_text())
    by_size = defaultdict(list)
    for candidate in candidates:
        by_size[candidate["tokens"]].append(candidate)
    trials, flat = [], 0
    previous = None
    for size, group in sorted(by_size.items()):
        best_ppl = min(c["perplexity"] for c in group)
        candidate = min(
            (c for c in group if c["perplexity"] <= 1.01 * best_ppl),
            key=lambda c: c["order"],
        )
        current = evaluate_candidate(
            root,
            output,
            config,
            manifest,
            candidate,
            references,
            recipe["acoustic_scales"],
        )
        trials.extend(current)
        write_json(output / "development-results.json", trials)
        best = min(current, key=lambda r: (r["wer"], r["cer"]))
        if (
            previous
            and previous["oov_rate"] - best["oov_rate"] < 0.005
            and previous["wer"] - best["wer"] < 0.01
        ):
            flat += 1
        else:
            flat = 0
        previous = best
        if flat >= 2 and best["oov_rate"] <= recipe["target_oov"]:
            break
    eligible = [t for t in trials if t["oov_rate"] <= recipe["target_oov"]]
    pool = eligible or trials
    best_wer = min(t["wer"] for t in pool)
    selected = min(
        (t for t in pool if t["wer"] <= best_wer + recipe["wer_tolerance"]),
        key=lambda t: (
            t["tokens"],
            t["graph_bytes"],
            t["wer"],
            abs(t["acoustic_scale"] - 1),
        ),
    )
    selection = {
        "model_receipt": selected["receipt"],
        "acoustic_scale": selected["acoustic_scale"],
        "candidate": selected,
        "coverage_target_met": bool(eligible),
        "recipe_sha256": digest(recipe),
        "training_sha256": fingerprints(output / "training.jsonl")["sha256"],
        "development_sha256": fingerprints(output / "development.json")["sha256"],
        "configuration": {
            **config,
            "decoder": {
                **config["decoder"],
                "acoustic_scale": selected["acoustic_scale"],
            },
        },
        "selection_basis": "development only; smallest within one absolute WER point of best tested eligible candidate",
    }
    write_json(output / "selection.json", selection)
    write_json(root / ".cache/scn/selected.json", selection)


def test_study(root, output, recipe, config, manifest):
    selection = json.loads((output / "selection.json").read_text())
    config = deepcopy(selection["configuration"])
    if selection["recipe_sha256"] != digest(recipe):
        raise ResourceError("Recipe changed after model selection")
    for name in ("training", "development"):
        suffix = ".jsonl" if name == "training" else ".json"
        if (
            fingerprints(output / (name + suffix))["sha256"]
            != selection[name + "_sha256"]
        ):
            raise ResourceError("Development inputs changed after selection")
    # Final references have a separate review ledger and cannot affect training.
    ledger_path = root / SCN_RESOURCES / "poc_test_reviews.json"
    review = (
        json.loads(ledger_path.read_text())
        if ledger_path.exists()
        else {"decisions": {}}
    )
    references, pending = [], []
    for video in recipe["splits"]["test"]:
        record = video_record(root, video)
        value = reviewed_text(record, review)
        if value is None:
            pending.append(record)
        else:
            references.append(
                {**record, "normalized": value[0], "corrections": value[1]}
            )
    write_json(output / "test-review-queue.json", pending)
    if pending:
        raise ResourceError(
            "Final reference review incomplete; selected model remains frozen"
        )
    write_json(output / "test-references.json", references)
    result = evaluate_candidate(
        root,
        output,
        config,
        manifest,
        selection["candidate"],
        references,
        [selection["acoustic_scale"]],
    )[0]
    result["selection_sha256"] = digest(selection)
    result["reference_ledger_sha256"] = fingerprints(ledger_path)["sha256"]
    write_json(output / "test.json", result)


def report_study(root, output, recipe, config, manifest):
    selection = json.loads((output / "selection.json").read_text())
    results = json.loads((output / "development-results.json").read_text())
    test = (
        json.loads((output / "test.json").read_text())
        if (output / "test.json").exists()
        else None
    )
    lines = [
        "# Sicilian DOS24 POC",
        "",
        "Fixed ZIPA acoustic model; development-selected word n-gram WFST.",
        "",
        "| Training tokens | Order | Vocabulary | Dev OOV | Acoustic scale | Dev WER | Dev CER | Graph MiB |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in results:
        lines.append(
            f"| {r['tokens']:,} | {r['order']} | {r['vocabulary_size']:,} | {r['oov_rate']:.2%} | {r['acoustic_scale']} | {r['wer']:.2%} | {r['cer']:.2%} | {r['graph_bytes'] / 2**20:.1f} |"
        )
    lines += [
        "",
        f"Development coverage target met: **{selection['coverage_target_met']}**.",
        f"Selected: {selection['candidate']['tokens']:,} training tokens, order {selection['candidate']['order']}, {selection['candidate']['vocabulary_size']:,} words; acoustic scale {selection['acoustic_scale']}.",
        "",
        (
            f"Largest evaluated training slice: {max(r['tokens'] for r in results):,} tokens. "
            f"Recipe checkpoints reached: {sum(n <= max(r['tokens'] for r in results) for n in recipe['checkpoints'])}/{len(recipe['checkpoints'])}. "
            f"Development OOV target: {recipe['target_oov']:.0%}."
        ),
        "",
        "The caption authorship and native-speaker representativeness are not independently established. Generated pronunciations remain provisional. DOS24 reviews are contextual project reviews, not external certification.",
    ]
    if test:
        lines += [
            "",
            f"Final {len(test['results'])}-video WER: **{test['wer']:.2%}**; CER: **{test['cer']:.2%}**; decoding-vocabulary OOV: **{test['oov_rate']:.2%}**.",
            f"First acoustic inference: {test['acoustic_seconds']:.2f} s; decoding cached posteriors: {test['decode_seconds']:.2f} s; process peak RSS: {test['process_peak_rss_bytes'] / 2**30:.2f} GiB. These exclude acquisition and tool builds.",
            "",
            "| Video | WER | CER | Caption time coverage | Comparison |",
            "| --- | ---: | ---: | ---: | --- |",
        ]
        for row in test["results"]:
            m = row["metrics"]
            lines.append(
                f"| {row['video']} | {m['wer']['rate']:.2%} | {m['cer']['rate']:.2%} | {row['coverage']['fraction']:.2%} | {row['path']}/comparison.txt |"
            )
        lines += [
            "",
            "Scoring uses full audio, including subtitle gaps and foreign examples. No audio intervals were excluded. Caption omissions/paraphrases and provisional pronunciations limit interpretation. Timing-gap details are in each result.json.",
        ]
    (output / "report.md").write_text("\n".join(lines) + "\n")
    write_json(
        output / "report.json",
        {
            "selection": selection,
            "development": results,
            "test": test,
        },
    )


def study(root, output, recipe, stage):
    manifest = load_resources(root)
    config = load_config("scn", root, manifest)
    torch.set_num_threads(config["acoustic"].get("threads", 4))
    stages = {
        "development": development_study,
        "test": test_study,
        "report": report_study,
    }
    stages[stage](root, output, recipe, config, manifest)
