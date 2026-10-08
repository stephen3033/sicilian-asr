"""Load configuration scaffolding without loading models or linguistic data."""

from pathlib import Path

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
    if config["decoder"]["device"] != "cpu" or config["decoder"]["enabled"]:
        raise ResourceError("Initialization supports disabled k2 CPU decoding only")
    if config["language_switching"]["enabled"]:
        raise ResourceError("Language switching is not implemented")
    return config
