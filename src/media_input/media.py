import json
from pathlib import Path

import numpy as np
import soundfile as sf

from common.errors import ResourceError
from common.settings import MIN_AUDIO_SAMPLES, SAMPLE_RATE
from common.utils import command


def extract_audio(
    source: Path, destination: Path, stream: int | None = None
) -> tuple[np.ndarray, dict]:
    if not source.is_file():
        raise ResourceError(f"Input does not exist: {source}")
    probe = json.loads(
        command(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_streams",
                "-show_format",
                "-of",
                "json",
                str(source),
            ]
        )
    )
    audio = [s for s in probe.get("streams", []) if s["codec_type"] == "audio"]
    selected = (
        next((s for s in audio if s["index"] == stream), None)
        if stream is not None
        else next(iter(audio), None)
    )
    if selected is None:
        raise ResourceError(
            "Input has no requested audio stream (stream is an absolute FFprobe index)"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    command(
        [
            "ffmpeg",
            "-nostdin",
            "-v",
            "error",
            "-y",
            "-i",
            str(source),
            "-map",
            f"0:{selected['index']}",
            "-vn",
            "-ac",
            "1",
            "-ar",
            str(SAMPLE_RATE),
            "-c:a",
            "pcm_f32le",
            str(destination),
        ]
    )
    samples, rate = sf.read(destination, dtype="float32")
    if (
        rate != SAMPLE_RATE
        or samples.ndim != 1
        or samples.size < MIN_AUDIO_SAMPLES
        or not np.isfinite(samples).all()
    ):
        raise ResourceError(
            "Decoded audio must be finite 16 kHz mono and at least 100 ms"
        )
    origin = float(
        selected.get("start_time", probe.get("format", {}).get("start_time", 0))
    )
    return samples, {
        "probe": probe,
        "selected_stream": selected["index"],
        "timestamp_origin_seconds": origin,
        "samples": len(samples),
        "sample_rate": rate,
        "duration_seconds": len(samples) / rate,
        "ffmpeg_version": command(["ffmpeg", "-version"]).splitlines()[0],
    }
