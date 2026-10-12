"""Sequential audio → ZIPA → Compact CTC WFST → evaluation orchestration."""

from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import resource
import time
import traceback

import k2
import numpy as np
import torch

from sicilian_asr.acoustic.zipa import ZipaAcoustic, greedy_ctc
from sicilian_asr.audio import extract_audio, segment_audio
from sicilian_asr.decoding.compact import decode
from sicilian_asr.evaluation import evaluate, read_reference
from sicilian_asr.preparation import identity, package_versions, verify_cache
from sicilian_asr.resources import ResourceError, fingerprints, local_path, write_json
from sicilian_asr.text import TEXT_RULES


def transcribe(root: Path, config: dict, manifest: dict, source: Path, reference: Path | None, output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ResourceError(f"Output directory must be empty to prevent stale success artifacts: {output}")
    started = time.perf_counter()
    metadata = {"status": "running", "started_at": datetime.now(timezone.utc).isoformat(), "configuration": config,
                "platform": platform.platform(), "versions": package_versions(), "normalization": TEXT_RULES, "timings_seconds": {}, "segments": []}
    write_json(output/"run_metadata.json", metadata)

    def stage(name, function):
        before = time.perf_counter()
        metadata["active_stage"] = name
        write_json(output/"run_metadata.json", metadata)
        result = function()
        metadata["timings_seconds"][name] = time.perf_counter()-before
        return result

    try:
        if config["languages"] != ["ita"] or not config["decoder"]["enabled"]:
            raise ResourceError("Only prepared Italian CPU decoding is supported")
        metadata["input"] = {"path": str(source.resolve()), **fingerprints(source)}
        key, payload = stage("verify_sources", lambda: identity(root, config, manifest))
        directory = local_path(root, config["decoder"]["graph_path"])/key
        prepared = stage("verify_graph", lambda: verify_cache(directory, key))
        metadata["prepared"] = {"path": str(directory), **prepared}
        metadata["graph_statistics"] = json.loads((directory/"compilation.json").read_text())
        torch.set_num_threads(config["acoustic"].get("threads", 4))
        graph = stage("load_graph", lambda: k2.Fsa.from_dict(torch.load(directory/"TLG.pt", map_location="cpu", weights_only=False)))
        words = {int(i): w for w, i in (line.rsplit(" ", 1) for line in (directory/"words.txt").read_text().splitlines())}
        samples, audio_info = stage("audio_extraction", lambda: extract_audio(source, output/"audio.wav", config["audio"].get("stream")))
        metadata["audio"] = audio_info
        metadata["extracted_audio"] = fingerprints(output/"audio.wav")
        segments = stage("segmentation", lambda: segment_audio(samples, config["audio"], audio_info["timestamp_origin_seconds"]))
        model = stage("load_acoustic", lambda: ZipaAcoustic(root, config["acoustic"], manifest))
        metadata["preprocessing"] = model.preprocessing
        hypotheses, greedy = [], []
        for segment in segments:
            print(f"Segment {segment.index+1}/{len(segments)}: {segment.start_sample/16000:.2f}–{segment.end_sample/16000:.2f} s", flush=True)
            acoustic = stage(f"acoustic_{segment.index}", lambda: model.infer([torch.from_numpy(samples[segment.start_sample:segment.end_sample].copy())]))
            diagnostic = greedy_ctc(acoustic)[0]
            result = stage(f"decode_{segment.index}", lambda: decode(graph, acoustic, words, config["decoder"]))
            hypotheses.append(result["text"])
            greedy.append(diagnostic)
            metadata["segments"].append({**segment.to_dict(), **result, "ipa_greedy": diagnostic})
            write_json(output/"segments.json", {"segments": metadata["segments"]})
            if config.get("exports", {}).get("probabilities", False):
                np.savez_compressed(output/f"probabilities-{segment.index:05d}.npz", log_probs=acoustic.log_probs[0, :int(acoustic.lengths[0])].numpy(),
                                    tokens=np.asarray(acoustic.tokens), blank_id=acoustic.blank_id, frame_length=int(acoustic.lengths[0]))
        hypothesis = " ".join(hypotheses)
        (output/"transcription.txt").write_text(hypothesis+"\n", encoding="utf-8")
        (output/"ipa_greedy.txt").write_text("\n".join(greedy)+"\n", encoding="utf-8")
        if reference is not None:
            metadata["reference"] = {"path": str(reference.resolve()), **fingerprints(reference)}
            parsed = stage("reference_parsing", lambda: read_reference(reference))
            full = set(json.loads((directory/"lexicon_full.json").read_text()))
            selected = set(json.loads((directory/"vocabulary.json").read_text())["words"])
            metrics, comparison = stage("evaluation", lambda: evaluate(hypothesis, parsed, full, selected))
            write_json(output/"reference_cues.json", parsed)
            normalized = parsed["normalized"]
        else:
            metrics = {"status": "not_evaluated", "reason": "No reference supplied"}
            comparison, normalized = "No reference supplied.\n", ""
        (output/"reference_normalized.txt").write_text(normalized+"\n", encoding="utf-8")
        (output/"comparison.txt").write_text(comparison, encoding="utf-8")
        write_json(output/"metrics.json", metrics)
        if config["decoder"].get("diagnostic_paths", 0):
            write_json(output/"diagnostics.json", {"segments": metadata["segments"],
                "oov_causes": metrics.get("oov_causes", []),
                "note": "Reference is used only for scoring and OOV reporting, never to propose decoding candidates."})
        metadata["status"] = "complete"
        metadata["active_stage"] = None
        metadata["output_hashes"] = {p.name: fingerprints(p)["sha256"] for p in output.iterdir() if p.is_file() and p.name != "run_metadata.json"}
        metadata["limitations"] = ["Segment boundaries only; no word alignment.", "Subtitle omissions, paraphrases, and timing mismatch affect scores.",
                                   "<unk> does not identify arbitrary OOV speech; closed-vocabulary substitutions remain possible.", "No PER without IPA reference."]
        return metadata
    except BaseException as error:
        metadata.update(status="failed", error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
        raise
    finally:
        metadata["elapsed_seconds"] = time.perf_counter()-started
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        metadata["peak_rss_bytes"] = peak if platform.system() == "Darwin" else peak*1024
        metadata["finished_at"] = datetime.now(timezone.utc).isoformat()
        write_json(output/"run_metadata.json", metadata)
