import hashlib
import io

from pipeline.validation import validate_resources
from tests.support import copy_configuration


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


def test_missing_downloads_are_not_resource_ready(tmp_path):
    copy_configuration(tmp_path)
    report = validate_resources(tmp_path)
    assert not report["resource_ready"]
    assert report["pipeline_implemented"]
    assert not report["graph_ready"]
    assert {"mls_ita_lm", "phonitalia_ipa", "zipa_checkpoint"} <= set(
        report["required_missing"]
    )
    assert report["resources"]["scn_lexicon"]["status"] == "planned"
