"""Load configuration scaffolding without loading models or linguistic data."""

from pathlib import Path
import math

import yaml

from sicilian_asr.resources import ResourceError, local_path


def load_config(path: Path, root: Path, manifest: dict) -> dict:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or config.get("schema_version") != 1:
        raise ResourceError(f"Unsupported configuration schema: {path}")
    languages = config.get("languages")
    if languages not in (["ita"], ["scn"], ["ita", "scn"]):
        raise ResourceError(f"Expected ISO 639-3 languages ita/scn: {path}")
    expected_output = languages[0] if len(languages) == 1 else None
    if config.get("output_language") != expected_output:
        raise ResourceError(f"Output language conflicts with languages: {path}")
    if set(config.get("resources", {})) != set(languages):
        raise ResourceError(f"Language resource references are incomplete: {path}")
    specs = manifest["resources"]
    for section in [config["acoustic"], *config["resources"].values()]:
        for key, value in section.items():
            if key.endswith("_path") and value is not None:
                local_path(root, value)
            if key.endswith("_resource"):
                if value not in specs:
                    raise ResourceError(f"Unknown manifest resource {value}: {path}")
                corresponding_path = key.removesuffix("_resource") + "_path"
                if corresponding_path in section and section[corresponding_path] != specs[value]["path"]:
                    raise ResourceError(f"Configuration/manifest path mismatch for {value}: {path}")
    decoder = config["decoder"]
    if decoder["device"] != "cpu" or config["acoustic"]["device"] != "cpu":
        raise ResourceError("Only CPU is supported on the pinned stack")
    if decoder["enabled"]:
        if languages != ["ita"]:
            raise ResourceError("Sicilian and code-switching remain unsupported scaffolds")
        if decoder["backend"] != "k2" or decoder["ctc_topology"] != "compact":
            raise ResourceError("Italian decoding requires k2 Compact CTC")
        local_path(root, decoder["graph_path"])
        for key in ("acoustic_scale", "lm_scale", "search_beam", "output_beam"):
            value = decoder[key]
            if not isinstance(value, (float, int)) or not math.isfinite(value) or value <= 0:
                raise ResourceError(f"Invalid decoder {key}")
        if any(type(decoder[k]) is not int for k in ("min_active_states", "max_active_states")) or not 0 <= decoder["min_active_states"] <= decoder["max_active_states"] or decoder["max_active_states"] <= 0:
            raise ResourceError("Invalid active state bounds")
        size = decoder.get("vocabulary_size")
        for key in ("pronunciation_variants", "contractions"):
            if type(decoder.get(key, False)) is not bool:
                raise ResourceError(f"Invalid decoder {key}: expected boolean")
        if size is not None and (type(size) is not int or size <= 0):
            raise ResourceError("Vocabulary size must be positive or null")
        acoustic = config["acoustic"]
        if (acoustic["backend"], acoustic["sample_rate"], acoustic["feature_bins"], acoustic["vocabulary_size"], acoustic["blank_id"]) != ("zipa_pytorch", 16000, 80, 127, 0):
            raise ResourceError("Acoustic settings disagree with pinned ZIPA")
        if type(acoustic.get("threads", 4)) is not int or acoustic.get("threads", 4) <= 0:
            raise ResourceError("CPU thread count must be positive")
    if config["language_switching"]["enabled"]:
        raise ResourceError("Language switching is not implemented")
    return config
