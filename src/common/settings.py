"""Shared runtime defaults and locations; loading never mutates these values."""

from pathlib import Path

SAMPLE_RATE = 16000
MIN_AUDIO_SAMPLES = 1600
CONFIG_PATHS = {
    "ita": "src/asr/ita/config.yaml",
    "scn": "src/asr/scn/config.yaml",
    "ita_scn": "src/pipeline/configs/ita_scn.yaml",
}
MODEL_CONFIG = "src/ipa/config.yaml"
MANIFEST_PATHS = (
    "src/ipa/resources/manifest.yaml",
    "src/asr/ita/resources/manifest.yaml",
    "src/asr/scn/resources/manifest.yaml",
)
SCN_RESOURCES = Path("src/asr/scn/resources")

DEFAULTS = {
    "schema_version": 1,
    "acoustic": {
        "device": "cpu",
        "threads": 4,
        "sample_rate": 16000,
        "output_type": "log_probabilities",
    },
    "audio": {
        "stream": None,
        "target_seconds": 20.0,
        "max_seconds": 24.0,
        "search_seconds": 4.0,
        "silence_seconds": 0.3,
        "silence_dbfs": -40.0,
    },
    "exports": {"probabilities": False},
    "decoder": {"backend": "k2", "device": "cpu"},
    "language_switching": {"enabled": False, "implementation": "reserved"},
}
