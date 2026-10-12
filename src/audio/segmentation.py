from dataclasses import asdict, dataclass

import numpy as np

from common.errors import ResourceError
from common.settings import DEFAULTS, MIN_AUDIO_SAMPLES, SAMPLE_RATE


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


def segment_audio(
    samples: np.ndarray, config: dict, origin: float = 0, rate: int = SAMPLE_RATE
) -> list[Segment]:
    target = float(config.get("target_seconds", DEFAULTS["audio"]["target_seconds"]))
    maximum = float(config.get("max_seconds", DEFAULTS["audio"]["max_seconds"]))
    window = float(config.get("search_seconds", DEFAULTS["audio"]["search_seconds"]))
    silence = float(config.get("silence_seconds", DEFAULTS["audio"]["silence_seconds"]))
    threshold = float(config.get("silence_dbfs", DEFAULTS["audio"]["silence_dbfs"]))
    if not (0 < window < target <= maximum and silence > 0 and np.isfinite(threshold)):
        raise ResourceError("Invalid segmentation settings")
    if (
        samples.ndim != 1
        or not len(samples)
        or not np.isfinite(samples).all()
        or not np.isfinite(origin)
    ):
        raise ResourceError("Invalid samples or timestamp origin")
    hop = max(1, round(0.01 * rate))
    result, start = [], 0
    while start < len(samples):
        if len(samples) - start <= round(maximum * rate):
            end, boundary = len(samples), "end"
        else:
            lo = start + round((target - window) * rate)
            hi = start + round(maximum * rate)
            centers, energies = [], []
            for offset in range(lo, hi - hop + 1, hop):
                centers.append(offset + hop // 2)
                energies.append(
                    float(
                        np.mean(samples[offset : offset + hop].astype(np.float64) ** 2)
                    )
                )
            quiet = np.asarray(energies) <= 10 ** (threshold / 10)
            edges = np.diff(np.r_[False, quiet, False].astype(int))
            candidates = []
            for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
                if (b - a) * hop >= silence * rate:
                    candidates.append((centers[a] + centers[b - 1]) // 2)
            if candidates:
                end = min(candidates, key=lambda s: (abs(s - start - target * rate), s))
                boundary = "silence"
            else:
                best = min(
                    range(len(energies)),
                    key=lambda i: (
                        energies[i],
                        abs(centers[i] - start - target * rate),
                    ),
                )
                end, boundary = centers[best], "low_energy"
            # Avoid a final tail too short for ZIPA's subsampling convolution.
            if len(samples) - end < MIN_AUDIO_SAMPLES:
                end = len(samples) - MIN_AUDIO_SAMPLES
        result.append(
            Segment(
                len(result),
                start,
                end,
                origin + start / rate,
                origin + end / rate,
                boundary,
                boundary == "low_energy",
            )
        )
        start = end
    return result
