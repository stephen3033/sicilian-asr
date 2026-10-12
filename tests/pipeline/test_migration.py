import io
import json
import tarfile

import pytest

from common.errors import ResourceError
from common.utils import fingerprints, write_json
from pipeline.migration import migrate


def fixture(root):
    relative = "src/asr/ita/resources/lm/model.arpa"
    old = root / "resources/ita/lm/model.arpa"
    old.parent.mkdir(parents=True)
    old.write_bytes(b"original weights\n")
    archive = root / "model.tar.gz"
    with tarfile.open(archive, "w:gz") as stream:
        member = tarfile.TarInfo("release/model.arpa")
        member.size = old.stat().st_size
        stream.addfile(member, io.BytesIO(old.read_bytes()))
    spec = {
        "status": "available",
        "path": "model.tar.gz",
        "revision": "test",
        **fingerprints(archive),
        "members": {"release/model.arpa": relative},
    }
    receipt = {
        "archive_sha256": spec["sha256"],
        "artifacts": {str(old.relative_to(root)): fingerprints(old)},
    }
    write_json(root / "model.tar.gz.members.json", receipt)
    return {"resources": {"lm": spec}}, old, root / relative, receipt


def test_dry_run_apply_and_idempotence_preserve_verified_bytes(tmp_path):
    manifest, old, new, receipt = fixture(tmp_path)
    report = migrate(tmp_path, manifest=manifest)
    assert report["moves"] and old.exists() and not new.exists()
    migrate(tmp_path, apply=True, manifest=manifest)
    assert not old.exists() and new.read_bytes() == b"original weights\n"
    preserved = (
        tmp_path / ".cache/layout-migration/original-receipts/model.tar.gz.members.json"
    )
    assert json.loads(preserved.read_text()) == receipt
    second = migrate(tmp_path, apply=True, manifest=manifest)
    assert not second["moves"] and not second["receipts"]


@pytest.mark.parametrize("corruption", ["source", "destination", "archive"])
def test_conflicts_and_corruption_fail_before_any_move(tmp_path, corruption):
    manifest, old, new, receipt = fixture(tmp_path)
    if corruption == "destination":
        new.parent.mkdir(parents=True)
        new.write_bytes(b"conflicting data")
    elif corruption == "archive":
        (tmp_path / "model.tar.gz").write_bytes(b"broken")
    else:
        old.write_bytes(b"corrupt source")
    with pytest.raises(ResourceError):
        migrate(tmp_path, apply=True, manifest=manifest)
    assert old.exists()
    assert json.loads((tmp_path / "model.tar.gz.members.json").read_text()) == receipt
    assert not (tmp_path / ".cache/layout-migration").exists()
