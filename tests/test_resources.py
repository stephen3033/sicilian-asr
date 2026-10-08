import hashlib
import io
from pathlib import Path
import shutil

import pytest

from sicilian_asr.resources import ResourceError, download_resource, local_path
from sicilian_asr.validation import lexicon_statistics, validate_resources


ROOT = Path(__file__).resolve().parents[1]


class Response(io.BytesIO):
    status = 200

    def __init__(self, body: bytes, content_length=None):
        super().__init__(body)
        self.headers = {"Content-Length": str(len(body) if content_length is None else content_length)}

    def geturl(self):
        return "https://example.org/artifact"


def specification(body=b"upstream bytes\n"):
    return {
        "status": "available", "path": "resources/test.data", "revision": "a" * 40,
        "url": "https://example.org/artifact", "sha256": hashlib.sha256(body).hexdigest(),
    }


def test_atomic_download_and_verified_reuse(tmp_path):
    spec = specification()
    result = download_resource(tmp_path, spec, opener=lambda *a, **k: Response(b"upstream bytes\n"))
    assert result["sha256"] == spec["sha256"]
    assert (tmp_path / spec["path"]).read_bytes() == b"upstream bytes\n"

    def no_network(*args, **kwargs):
        pytest.fail("Verified existing artifact should not use the network")

    assert download_resource(tmp_path, spec, opener=no_network) == result
    assert not list(tmp_path.rglob("*.part"))


def test_checksum_failure_never_creates_destination(tmp_path):
    spec = specification()
    with pytest.raises(ResourceError, match="sha256 mismatch"):
        download_resource(tmp_path, spec, opener=lambda *a, **k: Response(b"wrong bytes"))
    assert not (tmp_path / spec["path"]).exists()
    assert not list(tmp_path.rglob("*.part"))


def test_truncated_http_body_is_not_an_artifact(tmp_path):
    spec = specification()
    with pytest.raises(ResourceError, match="Incomplete HTTP body"):
        download_resource(tmp_path, spec, opener=lambda *a, **k: Response(b"short", 999))
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
        download_resource(tmp_path, {"status": "unavailable", "reason": "Official download unavailable"})
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("relative", ["../escape", "/tmp/absolute"])
def test_resource_paths_cannot_escape_root(tmp_path, relative):
    with pytest.raises(ResourceError, match="escapes"):
        local_path(tmp_path, relative)


def test_missing_downloads_are_not_resource_ready(tmp_path):
    (tmp_path / "resources").mkdir()
    shutil.copyfile(ROOT / "resources/manifest.yaml", tmp_path / "resources/manifest.yaml")
    shutil.copytree(ROOT / "configs", tmp_path / "configs")
    report = validate_resources(tmp_path)
    assert not report["resource_ready"]
    assert not report["pipeline_implemented"]
    assert {"fbk_ita_lm", "phonitalia_ipa", "zipa_checkpoint"} <= set(report["required_missing"])
    assert report["resources"]["scn_lexicon"]["status"] == "planned"


def test_lexicon_statistics_preserves_duplicate_rows(tmp_path):
    path = tmp_path / "test.dict"
    # Artificial unit-test fixture only, never installed as a linguistic resource.
    path.write_text("test\tt e s t\ntest\tt e s t\ntest\tt ɛ s t\n", encoding="utf-8")
    assert lexicon_statistics(path) == {"rows": 3, "vocabulary": 1, "distinct_word_pronunciation_pairs": 2}


def test_malformed_lexicon_is_rejected(tmp_path):
    path = tmp_path / "test.dict"
    path.write_text("not a TSV row\n")
    with pytest.raises(ResourceError, match="Malformed"):
        lexicon_statistics(path)
