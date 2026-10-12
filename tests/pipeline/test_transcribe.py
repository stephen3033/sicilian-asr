import json

import pytest

from common.errors import ResourceError
from common.resources import load_resources
from pipeline.config import load_config
from pipeline.transcribe import transcribe
from tests.support import ROOT


def test_failed_run_retains_diagnostics_without_success_or_greedy_fallback(tmp_path):
    manifest = load_resources(ROOT)
    config = load_config("ita", ROOT, manifest)
    output = tmp_path / "output"
    with pytest.raises(FileNotFoundError):
        transcribe(ROOT, config, manifest, tmp_path / "missing.mp4", None, output)
    metadata = json.loads((output / "run_metadata.json").read_text())
    assert metadata["status"] == "failed" and "error" in metadata
    assert not (output / "transcription.txt").exists()
    assert not (output / "ipa_greedy.txt").exists()
    with pytest.raises(ResourceError, match="empty"):
        transcribe(ROOT, config, manifest, tmp_path / "missing.mp4", None, output)
