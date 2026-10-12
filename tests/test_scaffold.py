from copy import deepcopy
from importlib import import_module
from pathlib import Path
import subprocess
import sys

import pytest
import yaml

from sicilian_asr.config import load_config
from sicilian_asr.resources import ResourceError, load_manifest


ROOT = Path(__file__).resolve().parents[1]


def test_package_imports():
    for name in ("sicilian_asr", "sicilian_asr.acoustic", "sicilian_asr.decoding", "sicilian_asr.lexicon", "sicilian_asr.lm"):
        assert import_module(name) is not None


def test_manifest_is_complete_and_all_downloads_are_pinned():
    specs = load_manifest(ROOT / "resources/manifest.yaml")["resources"]
    assert {"phonitalia_ipa", "fbk_ita_lm", "zipa_checkpoint", "scn_dos24", "scn_lexicon", "scn_vocabulary", "scn_lm"} <= specs.keys()
    for spec in specs.values():
        if spec["status"] == "available":
            assert len(spec["sha256"]) == 64
            if spec.get("provenance", "git") == "git":
                assert spec["revision"] in spec["url"]
            else:
                assert spec["md5"]
        else:
            assert spec["path"] is None
            assert spec["reason"]


@pytest.mark.parametrize("name,languages", [("ita", ["ita"]), ("scn", ["scn"]), ("ita_scn", ["ita", "scn"])])
def test_configurations_reference_manifest(name, languages):
    manifest = load_manifest(ROOT / "resources/manifest.yaml")
    config = load_config(ROOT / f"configs/{name}.yaml", ROOT, manifest)
    assert config["languages"] == languages
    assert config["decoder"]["enabled"] == (name == "ita")
    assert not config["language_switching"]["enabled"]
    assert config["acoustic"]["blank_id"] == 0


def test_unknown_resource_is_rejected(tmp_path):
    manifest = load_manifest(ROOT / "resources/manifest.yaml")
    config = yaml.safe_load((ROOT / "configs/ita.yaml").read_text())
    config["resources"]["ita"]["lm_resource"] = "invented_lm"
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(ResourceError, match="Unknown manifest resource"):
        load_config(path, ROOT, manifest)


def test_manifest_rejects_incomplete_provenance(tmp_path):
    manifest = deepcopy(load_manifest(ROOT / "resources/manifest.yaml"))
    del manifest["resources"]["phonitalia_ipa"]["license"]
    path = tmp_path / "manifest.yaml"
    path.write_text(yaml.safe_dump(manifest))
    with pytest.raises(ResourceError, match="Incomplete"):
        load_manifest(path)


def test_downloader_dry_run_does_not_acquire_resources(tmp_path):
    directory = tmp_path / "resources"
    directory.mkdir()
    (directory / "manifest.yaml").write_bytes((ROOT / "resources/manifest.yaml").read_bytes())
    process = subprocess.run(
        [sys.executable, str(ROOT / "scripts/download_resources.py"), "--root", str(tmp_path), "--all", "--dry-run"],
        capture_output=True, text=True,
    )
    assert process.returncode == 0, process.stderr
    assert "WARNING fbk_ita_lm: unavailable" in process.stdout
    assert "zipa_checkpoint:" in process.stdout
    assert not (tmp_path / ".cache").exists()
    assert list(directory.iterdir()) == [directory / "manifest.yaml"]
