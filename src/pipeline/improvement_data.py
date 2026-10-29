"""Source-preserving speech acquisition with recording-level leakage checks."""

import io
import json
from pathlib import Path

from asr.scn.dataset import digest, passage_hashes
from asr.scn.orthography import RULES, normalize
from common.errors import ResourceError
from common.utils import fingerprints, write_json


def verify_rows(rows):
    seen = {}
    for row in rows:
        split = row["split"]
        for key in ("recording_id", "audio_sha256"):
            value = (key, row[key])
            if value in seen and seen[value] != split:
                raise ResourceError(
                    f"Recording leakage: {key} across {seen[value]} and {split}"
                )
            seen[value] = split


def deduplicate(training, development):
    forbidden = set().union(
        *(passage_hashes(normalize(r["text"]).split()) for r in development)
    )
    forbidden_texts = {normalize(r["text"]) for r in development}
    seen, retained, excluded = set(), [], []
    for row in training:
        normalized = normalize(row["text"])
        identity = (
            row["audio_sha256"],
            row.get("start_seconds"),
            row.get("end_seconds"),
        )
        if (
            normalized in forbidden_texts
            or passage_hashes(normalized.split()) & forbidden
        ):
            excluded.append({"id": row["id"], "reason": "development_text_overlap"})
        elif identity in seen:
            excluded.append({"id": row["id"], "reason": "duplicate_audio_interval"})
        else:
            retained.append({**row, "text": normalized})
            seen.add(identity)
    return retained, excluded


def channel_rows(root, recipe, output, manifest_name="channel.json"):
    from asr.scn.dataset import video_record
    from media_input.media import extract_audio

    rows, recordings = [], []
    for split, videos in recipe["splits"].items():
        for video in videos:
            record = video_record(root, video)
            source = root / ".cache/scn/youtube" / video
            receipt = json.loads((source / "receipt.json").read_text())
            audio = output / "audio" / f"{video}.wav"
            if not audio.exists():
                audio.parent.mkdir(parents=True, exist_ok=True)
                extract_audio(source / receipt["audio"], audio)
            import soundfile as sf

            duration = sf.info(audio).duration
            common = {
                "recording_id": video,
                "audio": str(audio.resolve()),
                "audio_sha256": fingerprints(audio)["sha256"],
                "split": split,
                "cohort": "channel",
                "source_sha256": record["source_sha256"],
            }
            recordings.append(
                {**common, "id": video, "text": record["text"], "duration": duration}
            )
            if split == "train":
                # Preserve complete cue text; never truncate a transcript to fit a chunk.
                for i, cue in enumerate(record["cues"]):
                    start, end = cue["start_seconds"], cue["end_seconds"]
                    if 0 <= start < end <= duration + 0.1 and normalize(cue["text"]):
                        rows.append(
                            {
                                **common,
                                "id": f"{video}:{i}",
                                "start_seconds": start,
                                "end_seconds": min(end, duration),
                                "text": cue["text"],
                            }
                        )
    verify_rows(recordings)
    train, excluded = deduplicate(
        rows, [r for r in recordings if r["split"] == "development"]
    )
    write_json(
        output / manifest_name,
        {
            "normalization": RULES,
            "recordings": recordings,
            "training": train,
            "excluded": excluded,
        },
    )
    return recordings, train


def acquire_public(job):
    """Fetch only Sicilian training audio and pin the resolved dataset revision."""
    import numpy as np
    import soundfile as sf
    import torch
    import torchaudio.functional as audio_functional
    from datasets import Audio, load_dataset
    from huggingface_hub import HfApi

    spec = job["public_data"]
    if spec["split"] != "train" or spec["config"] != "scn_Latn":
        raise ResourceError(
            "Public adaptation acquisition is restricted to scn_Latn/train"
        )
    directory = Path(job["directory"])
    directory.mkdir(parents=True, exist_ok=True)
    receipt = directory / "receipt.json"
    if receipt.exists():
        saved = json.loads(receipt.read_text())
        for row in saved["rows"]:
            if fingerprints(Path(row["audio"]))["sha256"] != row["audio_sha256"]:
                raise ResourceError("Public audio changed")
        return saved
    pin = directory / "revision.json"
    if pin.exists():
        revision = json.loads(pin.read_text())["revision"]
    else:
        revision = (
            HfApi().dataset_info(spec["repository"], revision=spec["revision"]).sha
        )
        write_json(pin, {"repository": spec["repository"], "revision": revision})
    dataset = load_dataset(
        spec["repository"],
        spec["config"],
        split="train",
        revision=revision,
        streaming=True,
    )
    dataset = dataset.cast_column("audio", Audio(decode=False))
    rows, rejected = [], []
    for i, source in enumerate(dataset):
        from common.artifacts import space_check

        space_check(directory, reserve=job.get("reserve_bytes", 5 * 1024**3))
        if source.get("language", "scn_Latn") != "scn_Latn":
            raise ResourceError("Unexpected language in public subset")
        identifier = digest(
            [
                revision,
                source.get("speaker_id"),
                source.get("prompt_id"),
                source.get("segment_id"),
                i,
            ]
        )
        path = directory / f"{identifier}.flac"
        if not path.exists():
            encoded = source["audio"]
            audio_source = (
                io.BytesIO(encoded["bytes"])
                if encoded.get("bytes") is not None
                else encoded["path"]
            )
            wave, rate = sf.read(audio_source, dtype="float32")
            if wave.ndim == 2:
                wave = wave.mean(axis=1)
            if not np.isfinite(wave).all() or not len(wave):
                rejected.append({"id": identifier, "reason": "invalid_audio"})
                continue
            if rate != 16000:
                wave = audio_functional.resample(
                    torch.from_numpy(wave), rate, 16000
                ).numpy()
            temporary = path.with_suffix(".part")
            sf.write(temporary, wave, 16000, format="FLAC")
            temporary.replace(path)
        duration = sf.info(path).duration
        text = normalize(source["raw_text"])
        if not text:
            rejected.append({"id": identifier, "reason": "empty_transcript"})
            continue
        rows.append(
            {
                "id": identifier,
                "recording_id": f"omni:{identifier}",
                "audio": str(path.resolve()),
                "audio_sha256": fingerprints(path)["sha256"],
                "text": text,
                "start_seconds": 0,
                "end_seconds": duration,
                "split": "train",
                "cohort": "public",
                "speaker_id": source.get("speaker_id"),
                "source_revision": revision,
            }
        )
        if i % 25 == 0:
            write_json(
                directory / "progress.json",
                {"rows": len(rows), "seconds": sum(r["end_seconds"] for r in rows)},
            )
            print(f"Public training recordings: {len(rows)}", flush=True)
    saved = {
        "repository": spec["repository"],
        "revision": revision,
        "split": "train",
        "config": "scn_Latn",
        "normalization": RULES,
        "license": "CC-BY-4.0",
        "rows": rows,
        "rejected": rejected,
    }
    write_json(receipt, saved)
    return saved
