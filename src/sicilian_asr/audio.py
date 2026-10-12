"""FFmpeg extraction and contiguous, sample-exact segment boundaries."""

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import subprocess

import numpy as np
import soundfile as sf

from sicilian_asr.resources import ResourceError


def command(args: list[str]) -> str:
    try:
        result = subprocess.run(args, capture_output=True, text=True, check=True)
    except (OSError, subprocess.CalledProcessError) as error:
        raise ResourceError(f"{args[0]} failed: {getattr(error, 'stderr', str(error))}") from error
    return result.stdout


def extract_audio(source: Path, destination: Path, stream: int | None = None) -> tuple[np.ndarray, dict]:
    if not source.is_file():
        raise ResourceError(f"Input does not exist: {source}")
    probe = json.loads(command(["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(source)]))
    audio = [s for s in probe.get("streams", []) if s["codec_type"] == "audio"]
    selected = next((s for s in audio if s["index"] == stream), None) if stream is not None else next(iter(audio), None)
    if selected is None:
        raise ResourceError("Input has no requested audio stream (stream is an absolute FFprobe index)")
    destination.parent.mkdir(parents=True, exist_ok=True)
    command(["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(source), "-map", f"0:{selected['index']}", "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_f32le", str(destination)])
    samples, rate = sf.read(destination, dtype="float32")
    if rate != 16000 or samples.ndim != 1 or samples.size < 1600 or not np.isfinite(samples).all():
        raise ResourceError("Decoded audio must be finite 16 kHz mono and at least 100 ms")
    origin = float(selected.get("start_time", probe.get("format", {}).get("start_time", 0)))
    return samples, {"probe": probe, "selected_stream": selected["index"], "timestamp_origin_seconds": origin,
                     "samples": len(samples), "sample_rate": rate, "duration_seconds": len(samples) / rate,
                     "ffmpeg_version": command(["ffmpeg", "-version"]).splitlines()[0]}


@dataclass
class Segment:
    index: int
    start_sample: int
    end_sample: int
    source_start_seconds: float
    source_end_seconds: float
    boundary: str
    forced_cut: bool

    def to_dict(self):
        return asdict(self)


def segment_audio(samples: np.ndarray, config: dict, origin: float = 0, rate: int = 16000) -> list[Segment]:
    target = float(config.get("target_seconds", 20))
    maximum = float(config.get("max_seconds", 24))
    window = float(config.get("search_seconds", 4))
    silence = float(config.get("silence_seconds", .3))
    threshold = float(config.get("silence_dbfs", -40))
    if not (0 < window < target <= maximum and silence > 0 and np.isfinite(threshold)):
        raise ResourceError("Invalid segmentation settings")
    if samples.ndim != 1 or not len(samples) or not np.isfinite(samples).all() or not np.isfinite(origin):
        raise ResourceError("Invalid samples or timestamp origin")
    hop = max(1, round(.01 * rate))
    result, start = [], 0
    while start < len(samples):
        if len(samples) - start <= round(maximum * rate):
            end, boundary = len(samples), "end"
        else:
            lo = start + round((target-window) * rate)
            hi = start + round(maximum * rate)
            centers, energies = [], []
            for offset in range(lo, hi-hop+1, hop):
                centers.append(offset + hop//2)
                energies.append(float(np.mean(samples[offset:offset+hop].astype(np.float64)**2)))
            quiet = np.asarray(energies) <= 10 ** (threshold/10)
            edges = np.diff(np.r_[False, quiet, False].astype(int))
            candidates = []
            for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
                if (b-a)*hop >= silence*rate:
                    candidates.append((centers[a] + centers[b-1])//2)
            if candidates:
                end = min(candidates, key=lambda s: (abs(s-start-target*rate), s))
                boundary = "silence"
            else:
                best = min(range(len(energies)), key=lambda i: (energies[i], abs(centers[i]-start-target*rate)))
                end, boundary = centers[best], "low_energy"
            # Avoid a final tail too short for ZIPA's subsampling convolution.
            if len(samples)-end < 1600:
                end = len(samples)-1600
        result.append(Segment(len(result), start, end, origin+start/rate, origin+end/rate, boundary, boundary == "low_energy"))
        start = end
    return result
