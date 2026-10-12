import json

import pytest

from common.errors import ResourceError
from common.settings import SCN_RESOURCES
from common.utils import fingerprints
from pipeline.scn_poc import run
from pipeline.scn_study import caption_coverage, read_result


def test_caption_coverage_merges_overlaps_and_keeps_gaps():
    result = caption_coverage(
        [
            {"start_seconds": 1, "end_seconds": 4},
            {"start_seconds": 3, "end_seconds": 6},
            {"start_seconds": 8, "end_seconds": 12},
        ],
        10,
    )
    assert result["fraction"] == 0.7
    assert result["uncaptioned_intervals_seconds"] == [[0, 1], [6, 8]]
    assert result["excluded_audio_intervals_seconds"] == []


def test_decode_cache_rejects_changed_results(tmp_path):
    result = tmp_path / "result.json"
    result.write_text('{"text": "original"}')
    (tmp_path / "receipt.json").write_text(
        json.dumps(
            {
                "key": "test",
                "artifacts": {"result.json": fingerprints(result)["sha256"]},
            }
        )
    )
    assert read_result(tmp_path, "test")["text"] == "original"
    result.write_text('{"text": "changed"}')
    with pytest.raises(ResourceError, match="Changed"):
        read_result(tmp_path, "test")


@pytest.mark.parametrize("stage", ["corpus", "train", "development"])
def test_final_scores_freeze_all_development_stages(tmp_path, stage):
    resources = tmp_path / SCN_RESOURCES
    resources.mkdir(parents=True)
    (resources / "poc.json").write_text("{}")
    output = tmp_path / "outputs"
    output.mkdir()
    (output / "test.json").write_text("{}")
    with pytest.raises(ResourceError, match="frozen"):
        run(tmp_path, output, stage)
