import subprocess
import sys

import pytest
import torch

from asr.wfst.storage import load_tensor, save_tensor, tensor_path
from common.errors import ResourceError
from tests.support import ROOT


def test_compressed_tensor_artifacts_preserve_values_and_reduce_storage(tmp_path):
    value = {
        "arcs": torch.arange(30000, dtype=torch.int32).reshape(-1, 3),
        "scores": torch.zeros(10000),
        "metadata": {"value": 3},
    }
    plain = save_tensor(value, tmp_path, "plain")
    compressed = save_tensor(value, tmp_path, "compressed", compressed=True)
    assert compressed.stat().st_size < plain.stat().st_size
    assert tensor_path(tmp_path, "plain") == plain
    assert tensor_path(tmp_path, "compressed") == compressed
    for path in (plain, compressed):
        actual = load_tensor(path)
        assert actual["metadata"] == value["metadata"]
        assert torch.equal(actual["arcs"], value["arcs"])
        assert torch.equal(actual["scores"], value["scores"])


def test_tensor_selection_rejects_unlisted_ambiguous_siblings(tmp_path):
    with pytest.raises(FileNotFoundError):
        tensor_path(tmp_path)
    save_tensor({}, tmp_path, "TLG")
    save_tensor({}, tmp_path, "TLG", compressed=True)
    with pytest.raises(ResourceError, match="Ambiguous"):
        tensor_path(tmp_path)


def test_cache_verification_import_does_not_require_optional_torch_dependency():
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; sys.modules['torch'] = None; from asr.wfst.cache import verify_cache; from asr.wfst.storage import tensor_path",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
