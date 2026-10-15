"""Shared runtime defaults and locations; loading never mutates these values."""

from pathlib import Path

SAMPLE_RATE = 16000
MIN_AUDIO_SAMPLES = 1600
CONFIG_PATHS = {
    "ita": "configs/ita.yaml",
    "scn": "configs/scn.yaml",
    "ita_scn": "configs/ita_scn.yaml",
}
MODEL_CONFIG = "configs/ipa.yaml"
MANIFEST_PATHS = (
    "resources/ipa/manifest.yaml",
    "resources/ita/manifest.yaml",
    "resources/scn/manifest.yaml",
)
SCN_RESOURCES = Path("resources/scn")
SCN_CATALOG = SCN_RESOURCES / "sources/catalog.json"
SCN_EVIDENCE = Path("docs/scn/audits/evidence")
SCN_RECIPE = Path("configs/scn-poc.json")

LEGACY_CONFIG_PATHS = {
    "src/asr/ita/config.yaml": CONFIG_PATHS["ita"],
    "src/asr/scn/config.yaml": CONFIG_PATHS["scn"],
    "src/pipeline/configs/ita_scn.yaml": CONFIG_PATHS["ita_scn"],
}
# Specific document destinations precede the general resource directories.
RESOURCE_RELOCATIONS = (
    (
        "resources/ita/lexicon/phonItaliaR-LICENSE.md",
        "docs/ita/references/phonItaliaR-LICENSE.md",
    ),
    (
        "resources/ita/lexicon/phonItalia-1.10-Information.pdf",
        "docs/ita/references/phonItalia-1.10-Information.pdf",
    ),
    ("resources/ita/lm/mls/README", "docs/ita/references/mls/README"),
    ("resources/ita/lm/mls/LICENSE", "docs/ita/references/mls/LICENSE"),
)


def canonical_resource_path(relative: str) -> str:
    """Translate the two former resource layouts without changing arbitrary paths."""
    for language in ("ita", "scn"):
        prefix = f"src/asr/{language}/resources/"
        if relative.startswith(prefix):
            relative = f"resources/{language}/" + relative[len(prefix) :]
    return dict(RESOURCE_RELOCATIONS).get(relative, relative)


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
