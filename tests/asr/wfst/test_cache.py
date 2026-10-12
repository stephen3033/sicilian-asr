import json

import pytest

from asr.wfst.cache import verify_cache
from common.errors import ResourceError
from common.utils import fingerprints


def test_stale_and_corrupted_cache(tmp_path):
    p = tmp_path / "TLG.pt"
    p.write_bytes(b"fixture")
    (tmp_path / "prepared.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "key": "key",
                "artifacts": {"TLG.pt": fingerprints(p)["sha256"]},
            }
        )
    )
    assert verify_cache(tmp_path, "key")["status"] == "complete"
    with pytest.raises(ResourceError, match="Stale"):
        verify_cache(tmp_path, "different")
    p.write_bytes(b"changed")
    with pytest.raises(ResourceError, match="checksum"):
        verify_cache(tmp_path, "key")
