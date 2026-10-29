"""Assemble layer settings and validate a resolved pipeline configuration."""

import json
import math
from copy import deepcopy
from pathlib import Path

import yaml

from common.errors import ResourceError
from common.settings import (
    CONFIG_PATHS,
    DEFAULTS,
    LEGACY_CONFIG_PATHS,
    MODEL_CONFIG,
    canonical_resource_path,
)
from common.utils import local_path


def merge_settings(base: dict, overrides: dict) -> dict:
    """Merge mappings; replace scalars/lists, including explicit null values."""
    result = deepcopy(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = merge_settings(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def read_settings(path: Path) -> dict:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ResourceError(f"Expected configuration mapping: {path}")
    return data


def load_config(path: str | Path, root: Path, manifest: dict) -> dict:
    """Load a built-in language name or a YAML override file."""
    name = str(path)
    overrides = {}
    if name not in CONFIG_PATHS:
        selected_path = Path(path)
        relative = str(selected_path)
        if selected_path.is_absolute() and selected_path.is_relative_to(root):
            relative = str(selected_path.relative_to(root))
        if not selected_path.is_absolute():
            selected_path = root / selected_path
        if relative in LEGACY_CONFIG_PATHS and not selected_path.is_file():
            selected_path = root / LEGACY_CONFIG_PATHS[relative]
        overrides = read_settings(selected_path)
        languages = overrides.get("languages")
        names = {("ita",): "ita", ("scn",): "scn", ("ita", "scn"): "ita_scn"}
        if not isinstance(languages, list) or tuple(languages) not in names:
            raise ResourceError(f"Expected ISO 639-3 languages ita/scn: {path}")
        name = names[tuple(languages)]
    config = merge_settings(DEFAULTS, read_settings(root / MODEL_CONFIG))
    config = merge_settings(config, read_settings(root / CONFIG_PATHS[name]))
    if name == "scn":
        selected_model = local_path(
            root, config["resources"]["scn"]["model_receipt_path"]
        )
        if selected_model.is_file():
            selection = json.loads(selected_model.read_text())
            if "acoustic_scale" in selection:
                config["decoder"]["acoustic_scale"] = selection["acoustic_scale"]
    config = merge_settings(config, overrides)
    for section in [config["acoustic"], *config["resources"].values()]:
        for key, value in section.items():
            if isinstance(value, str) and (
                key.endswith("_path") or key.endswith("_directory")
            ):
                section[key] = canonical_resource_path(value)
    if config.get("schema_version") != 1:
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
                if (
                    corresponding_path in section
                    and section[corresponding_path] != specs[value]["path"]
                ):
                    raise ResourceError(
                        f"Configuration/manifest path mismatch for {value}: {path}"
                    )
    if config.get("recognizer", {}).get("backend", "zipa_wfst") != "zipa_wfst":
        raise ResourceError("Only ZIPA/WFST recognition is supported")
    decoder = config["decoder"]
    if decoder["device"] != "cpu" or config["acoustic"]["device"] != "cpu":
        raise ResourceError("Only CPU is supported on the pinned stack")
    if decoder["enabled"]:
        if languages not in (["ita"], ["scn"]):
            raise ResourceError("Code-switching remains unsupported")
        if decoder["backend"] != "k2" or decoder["ctc_topology"] != "compact":
            raise ResourceError("Decoding requires k2 Compact CTC")
        local_path(root, decoder["graph_path"])
        for key in ("acoustic_scale", "lm_scale", "search_beam", "output_beam"):
            value = decoder[key]
            if (
                not isinstance(value, (float, int))
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ResourceError(f"Invalid decoder {key}")
        if (
            any(
                type(decoder[k]) is not int
                for k in ("min_active_states", "max_active_states")
            )
            or not 0 <= decoder["min_active_states"] <= decoder["max_active_states"]
            or decoder["max_active_states"] <= 0
        ):
            raise ResourceError("Invalid active state bounds")
        size = decoder.get("vocabulary_size")
        for key in ("pronunciation_variants", "contractions"):
            if type(decoder.get(key, False)) is not bool:
                raise ResourceError(f"Invalid decoder {key}: expected boolean")
        if size is not None and (type(size) is not int or size <= 0):
            raise ResourceError("Vocabulary size must be positive or null")
        acoustic = config["acoustic"]
        if (
            acoustic["backend"],
            acoustic["sample_rate"],
            acoustic["feature_bins"],
            acoustic["vocabulary_size"],
            acoustic["blank_id"],
        ) != ("zipa_pytorch", 16000, 80, 127, 0):
            raise ResourceError("Acoustic settings disagree with pinned ZIPA")
        if (
            type(acoustic.get("threads", 4)) is not int
            or acoustic.get("threads", 4) <= 0
        ):
            raise ResourceError("CPU thread count must be positive")
    if config["language_switching"]["enabled"]:
        raise ResourceError("Language switching is not implemented")
    return config
