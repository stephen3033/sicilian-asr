import json

import pytest

from common.errors import ResourceError
from common.utils import fingerprints
from media_input.youtube import acquire, parse_json3


def test_multiline_captions_and_repetition():
    event = {"tStartMs": 200, "dDurationMs": 500, "segs": [{"utf8": "Cca\n\nsugnu."}]}
    repeated = {**event, "tStartMs": 900}
    result = parse_json3({"events": [event, event, repeated]})
    assert [r["text"] for r in result] == ["Cca\n\nsugnu.", "Cca\n\nsugnu."]
    assert result[0]["end_seconds"] == 0.7


@pytest.mark.parametrize(
    "events",
    [
        [],
        [{"tStartMs": -1, "dDurationMs": 20, "segs": [{"utf8": "text"}]}],
        [{"tStartMs": 0, "dDurationMs": float("nan"), "segs": [{"utf8": "text"}]}],
    ],
)
def test_invalid_caption_timing_and_empty_reference(events):
    with pytest.raises(ResourceError):
        parse_json3({"events": events})


@pytest.mark.parametrize(
    "requested", [{"language": "ita"}, {"channel_id": "different"}]
)
def test_cached_captions_must_match_track_and_channel(tmp_path, requested):
    source = tmp_path / "source.json"
    source.write_text(json.dumps({"channel_id": "original"}))
    (tmp_path / "receipt.json").write_text(
        json.dumps(
            {
                "video_id": "abcdefghijk",
                "language": "scn",
                "artifacts": {"source.json": fingerprints(source)["sha256"]},
            }
        )
    )
    with pytest.raises(ResourceError):
        acquire("abcdefghijk", tmp_path, **requested)
