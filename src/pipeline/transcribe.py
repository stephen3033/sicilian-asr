"""Sequential audio → ZIPA → Compact CTC WFST → evaluation orchestration."""

import json
import platform
import resource
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import k2
import numpy as np
import torch

from asr.wfst.cache import verify_cache
from asr.wfst.compact import decode
from audio.segmentation import segment_audio
from common.environment import package_versions
from common.errors import ResourceError
from common.settings import SAMPLE_RATE
from common.text import TEXT_RULES
from common.utils import fingerprints, local_path, write_json
from ipa.diagnostics import greedy_ctc
from ipa.zipa import ZipaAcoustic
from media_input.media import extract_audio
from media_input.reference import read_reference
from pipeline.languages import backend, text_policy


def transcribe(
    root: Path,
    config: dict,
    manifest: dict,
    source: Path,
    reference: Path | None,
    output: Path,
) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ResourceError(
            f"Output directory must be empty to prevent stale success artifacts: {output}"
        )
    started = time.perf_counter()
    metadata = {
        "status": "running",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "configuration": config,
        "platform": platform.platform(),
        "versions": package_versions(),
        "normalization": TEXT_RULES,
        "timings_seconds": {},
        "segments": [],
    }
    write_json(output / "run_metadata.json", metadata)

    def stage(name, function):
        before = time.perf_counter()
        metadata["active_stage"] = name
        write_json(output / "run_metadata.json", metadata)
        result = function()
        metadata["timings_seconds"][name] = time.perf_counter() - before
        return result

    try:
        if not config["decoder"]["enabled"]:
            raise ResourceError("Decoding is disabled")
        language, evaluate = backend(config)
        normalize, metadata["normalization"] = text_policy(config)
        identity = language.identity
        metadata["input"] = {"path": str(source.resolve()), **fingerprints(source)}
        key, payload = stage("verify_sources", lambda: identity(root, config, manifest))
        directory = local_path(root, config["decoder"]["graph_path"]) / key
        prepared = stage("verify_graph", lambda: verify_cache(directory, key))
        metadata["prepared"] = {"path": str(directory), **prepared}
        metadata["graph_statistics"] = json.loads(
            (directory / "compilation.json").read_text()
        )
        torch.set_num_threads(config["acoustic"].get("threads", 4))
        graph = stage(
            "load_graph",
            lambda: k2.Fsa.from_dict(
                torch.load(directory / "TLG.pt", map_location="cpu", weights_only=False)
            ),
        )
        words = {
            int(i): w
            for w, i in (
                line.rsplit(" ", 1)
                for line in (directory / "words.txt").read_text().splitlines()
            )
        }
        samples, audio_info = stage(
            "audio_extraction",
            lambda: extract_audio(
                source, output / "audio.wav", config["audio"].get("stream")
            ),
        )
        metadata["audio"] = audio_info
        metadata["extracted_audio"] = fingerprints(output / "audio.wav")
        segments = stage(
            "segmentation",
            lambda: segment_audio(
                samples, config["audio"], audio_info["timestamp_origin_seconds"]
            ),
        )
        model = stage(
            "load_acoustic", lambda: ZipaAcoustic(root, config["acoustic"], manifest)
        )
        metadata["preprocessing"] = model.preprocessing
        hypotheses, greedy = [], []
        for segment in segments:
            print(
                f"Segment {segment.index + 1}/{len(segments)}: {segment.start_sample / SAMPLE_RATE:.2f}–{segment.end_sample / SAMPLE_RATE:.2f} s",
                flush=True,
            )
            acoustic = stage(
                f"acoustic_{segment.index}",
                lambda: model.infer(
                    [
                        torch.from_numpy(
                            samples[segment.start_sample : segment.end_sample].copy()
                        )
                    ]
                ),
            )
            diagnostic = greedy_ctc(acoustic)[0]
            result = stage(
                f"decode_{segment.index}",
                lambda: decode(graph, acoustic, words, config["decoder"]),
            )
            hypotheses.append(result["text"])
            greedy.append(diagnostic)
            metadata["segments"].append(
                {**segment.to_dict(), **result, "ipa_greedy": diagnostic}
            )
            write_json(output / "segments.json", {"segments": metadata["segments"]})
            if config.get("exports", {}).get("probabilities", False):
                np.savez_compressed(
                    output / f"probabilities-{segment.index:05d}.npz",
                    log_probs=acoustic.log_probs[0, : int(acoustic.lengths[0])].numpy(),
                    tokens=np.asarray(acoustic.tokens),
                    blank_id=acoustic.blank_id,
                    frame_length=int(acoustic.lengths[0]),
                )
        hypothesis = " ".join(hypotheses)
        (output / "transcription.txt").write_text(hypothesis + "\n", encoding="utf-8")
        (output / "ipa_greedy.txt").write_text(
            "\n".join(greedy) + "\n", encoding="utf-8"
        )
        if reference is not None:
            metadata["reference"] = {
                "path": str(reference.resolve()),
                **fingerprints(reference),
            }
            parsed = stage("reference_parsing", lambda: read_reference(reference))
            full = set(json.loads((directory / "lexicon_full.json").read_text()))
            selected = set(
                json.loads((directory / "vocabulary.json").read_text())["words"]
            )
            metrics, comparison = stage(
                "evaluation", lambda: evaluate(hypothesis, parsed, full, selected)
            )
            write_json(output / "reference_cues.json", parsed)
            normalized = normalize(parsed["text"])
        else:
            metrics = {"status": "not_evaluated", "reason": "No reference supplied"}
            comparison, normalized = "No reference supplied.\n", ""
        (output / "reference_normalized.txt").write_text(
            normalized + "\n", encoding="utf-8"
        )
        (output / "comparison.txt").write_text(comparison, encoding="utf-8")
        write_json(output / "metrics.json", metrics)
        if config["decoder"].get("diagnostic_paths", 0):
            write_json(
                output / "diagnostics.json",
                {
                    "segments": metadata["segments"],
                    "oov_causes": metrics.get("oov_causes", []),
                    "note": "Reference is used only for scoring and OOV reporting, never to propose decoding candidates.",
                },
            )
        metadata["status"] = "complete"
        metadata["active_stage"] = None
        metadata["output_hashes"] = {
            p.name: fingerprints(p)["sha256"]
            for p in output.iterdir()
            if p.is_file() and p.name != "run_metadata.json"
        }
        metadata["limitations"] = [
            "Segment boundaries only; no word alignment.",
            "Subtitle omissions, paraphrases, and timing mismatch affect scores.",
            "<unk> does not identify arbitrary OOV speech; closed-vocabulary substitutions remain possible.",
            "No PER without IPA reference.",
        ]
        return metadata
    except BaseException as error:
        metadata.update(
            status="failed",
            error=f"{type(error).__name__}: {error}",
            traceback=traceback.format_exc(),
        )
        raise
    finally:
        metadata["elapsed_seconds"] = time.perf_counter() - started
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        metadata["peak_rss_bytes"] = (
            peak if platform.system() == "Darwin" else peak * 1024
        )
        metadata["finished_at"] = datetime.now(timezone.utc).isoformat()
        write_json(output / "run_metadata.json", metadata)
