import pytest
import yaml

from common.errors import ResourceError
from common.resources import load_resources
from pipeline.config import load_config
from tests.support import ROOT


@pytest.mark.parametrize(
    "section,key,value",
    [
        ("decoder", "device", "mps"),
        ("acoustic", "device", "cuda"),
        ("decoder", "ctc_topology", "modified"),
        ("decoder", "lm_scale", float("nan")),
        ("decoder", "vocabulary_size", 0),
        ("acoustic", "sample_rate", 48000),
    ],
)
def test_unsupported_configurations(tmp_path, section, key, value):
    manifest = load_resources(ROOT)
    config = load_config("ita", ROOT, manifest)
    config[section][key] = value
    p = tmp_path / "config.yaml"
    p.write_text(yaml.safe_dump(config))
    with pytest.raises(ResourceError):
        load_config(p, ROOT, manifest)


@pytest.mark.parametrize(
    "name,languages", [("ita", ["ita"]), ("scn", ["scn"]), ("ita_scn", ["ita", "scn"])]
)
def test_configurations_reference_manifest(name, languages):
    manifest = load_resources(ROOT)
    config = load_config(name, ROOT, manifest)
    assert config["languages"] == languages
    assert config["decoder"]["enabled"] == (name in {"ita", "scn"})
    assert not config["language_switching"]["enabled"]
    assert config["acoustic"]["blank_id"] == 0


def test_unknown_resource_is_rejected(tmp_path):
    manifest = load_resources(ROOT)
    config = load_config("ita", ROOT, manifest)
    config["resources"]["ita"]["lm_resource"] = "invented_lm"
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(ResourceError, match="Unknown manifest resource"):
        load_config(path, ROOT, manifest)


def test_layer_defaults_and_explicit_overrides_are_independent(tmp_path):
    from copy import deepcopy

    from common.settings import DEFAULTS

    original = deepcopy(DEFAULTS)
    manifest = load_resources(ROOT)
    override = tmp_path / "config.yaml"
    override.write_text(
        "languages: [ita]\naudio:\n  target_seconds: 18\n  stream: null\ndecoder:\n  vocabulary_size: null\n  contractions: false\n"
    )
    config = load_config(override, ROOT, manifest)
    assert config["audio"]["target_seconds"] == 18
    assert config["audio"]["stream"] is None
    assert config["audio"]["max_seconds"] == 24
    assert config["decoder"]["vocabulary_size"] is None
    assert config["decoder"]["contractions"] is False
    assert config["decoder"]["pronunciation_variants"] is True
    config["acoustic"]["threads"] = 99
    assert load_config("ita", ROOT, manifest)["acoustic"]["threads"] == 4
    assert DEFAULTS == original


def test_configuration_path_and_builtin_name_resolve_identically():
    manifest = load_resources(ROOT)
    assert load_config("ita", ROOT, manifest) == load_config(
        ROOT / "src/asr/ita/config.yaml", ROOT, manifest
    )
