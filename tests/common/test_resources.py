import hashlib
import io
import tarfile
from copy import deepcopy

import pytest
import yaml

from common.errors import ResourceError
from common.resources import (
    download_resource,
    extract_members,
    load_manifest,
    load_resources,
    verify_file,
)
from common.utils import fingerprints, local_path
from tests.support import ROOT


class Response(io.BytesIO):
    status = 200

    def __init__(self, body: bytes, content_length=None):
        super().__init__(body)
        self.headers = {
            "Content-Length": str(
                len(body) if content_length is None else content_length
            )
        }

    def geturl(self):
        return "https://example.org/artifact"


def specification(body=b"upstream bytes\n"):
    return {
        "status": "available",
        "path": "resources/test.data",
        "revision": "a" * 40,
        "url": "https://example.org/artifact",
        "sha256": hashlib.sha256(body).hexdigest(),
    }


def test_atomic_download_and_verified_reuse(tmp_path):
    spec = specification()
    result = download_resource(
        tmp_path, spec, opener=lambda *a, **k: Response(b"upstream bytes\n")
    )
    assert result["sha256"] == spec["sha256"]
    assert (tmp_path / spec["path"]).read_bytes() == b"upstream bytes\n"

    def no_network(*args, **kwargs):
        pytest.fail("Verified existing artifact should not use the network")

    assert download_resource(tmp_path, spec, opener=no_network) == result
    assert not list(tmp_path.rglob("*.part"))


def test_checksum_failure_never_creates_destination(tmp_path):
    spec = specification()
    with pytest.raises(ResourceError, match="sha256 mismatch"):
        download_resource(
            tmp_path, spec, opener=lambda *a, **k: Response(b"wrong bytes")
        )
    assert not (tmp_path / spec["path"]).exists()
    assert not list(tmp_path.rglob("*.part"))


def test_truncated_http_body_is_not_an_artifact(tmp_path):
    spec = specification()
    with pytest.raises(ResourceError, match="Incomplete HTTP body"):
        download_resource(
            tmp_path, spec, opener=lambda *a, **k: Response(b"short", 999)
        )
    assert not (tmp_path / spec["path"]).exists()
    assert not list(tmp_path.rglob("*.part"))


def test_existing_corruption_is_reported_without_overwrite(tmp_path):
    spec = specification()
    path = tmp_path / spec["path"]
    path.parent.mkdir()
    path.write_bytes(b"corrupt")
    with pytest.raises(ResourceError, match="mismatch"):
        download_resource(tmp_path, spec)
    assert path.read_bytes() == b"corrupt"


def test_unavailable_resource_cannot_be_substituted(tmp_path):
    with pytest.raises(ResourceError, match="Official download unavailable"):
        download_resource(
            tmp_path,
            {"status": "unavailable", "reason": "Official download unavailable"},
        )
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("relative", ["../escape", "/tmp/absolute"])
def test_resource_paths_cannot_escape_root(tmp_path, relative):
    with pytest.raises(ResourceError, match="escapes"):
        local_path(tmp_path, relative)


def test_academic_archive_checksums_and_derived_integrity(tmp_path):
    archive = tmp_path / "fixture.tar.gz"
    with tarfile.open(archive, "w:gz") as t:
        data = b"original arpa bytes\n"
        member = tarfile.TarInfo("release/lm.arpa")
        member.size = len(data)
        t.addfile(member, io.BytesIO(data))
    actual = fingerprints(archive)
    spec = {
        "path": "fixture.tar.gz",
        "revision": "release-1",
        "md5": actual["md5"],
        "members": {"release/lm.arpa": "derived/lm.arpa"},
    }
    assert verify_file(archive, spec)["sha256"] == actual["sha256"]
    spec["md5"] = "0" * 32
    with pytest.raises(ResourceError, match="md5 mismatch"):
        verify_file(archive, spec)
    spec["md5"] = actual["md5"]
    receipt = extract_members(tmp_path, spec, actual)
    assert receipt["archive_sha256"] == actual["sha256"]
    assert (tmp_path / "derived/lm.arpa").read_bytes() == data
    (tmp_path / "derived/lm.arpa").write_bytes(b"corrupted")
    with pytest.raises(ResourceError, match="checksum"):
        extract_members(tmp_path, spec, actual)


def test_existing_unreceipted_derivative_cannot_self_certify(tmp_path):
    archive = tmp_path / "fixture.tar.gz"
    with tarfile.open(archive, "w:gz") as t:
        member = tarfile.TarInfo("x")
        member.size = 1
        t.addfile(member, io.BytesIO(b"x"))
    (tmp_path / "derived").write_bytes(b"y")
    with pytest.raises(ResourceError, match="differs from archive"):
        extract_members(
            tmp_path,
            {"path": "fixture.tar.gz", "members": {"x": "derived"}},
            fingerprints(archive),
        )


def test_manifest_is_complete_and_all_downloads_are_pinned():
    specs = load_resources(ROOT)["resources"]
    assert {
        "phonitalia_ipa",
        "fbk_ita_lm",
        "zipa_checkpoint",
        "scn_dos24",
        "scn_lexicon",
        "scn_vocabulary",
        "scn_lm",
    } <= specs.keys()
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


def test_manifest_rejects_incomplete_provenance(tmp_path):
    manifest = deepcopy(load_resources(ROOT))
    del manifest["resources"]["phonitalia_ipa"]["license"]
    path = tmp_path / "manifest.yaml"
    path.write_text(yaml.safe_dump(manifest))
    with pytest.raises(ResourceError, match="Incomplete"):
        load_manifest(path)


def test_manifest_aggregation_rejects_duplicate_identifiers():
    from common.settings import MANIFEST_PATHS

    with pytest.raises(ResourceError, match="Duplicate resource IDs"):
        load_resources(ROOT, [MANIFEST_PATHS[0], MANIFEST_PATHS[0]])


def test_manifest_ownership():
    from common.settings import MANIFEST_PATHS

    acoustic, italian, sicilian = [
        load_manifest(ROOT / p)["resources"] for p in MANIFEST_PATHS
    ]
    assert all(name.startswith(("zipa_", "icefall_")) for name in acoustic)
    assert all(spec["language"] == "ita" for spec in italian.values())
    assert all(name.startswith("scn_") for name in sicilian)
    assert "scn_scritture_popolari" in sicilian
