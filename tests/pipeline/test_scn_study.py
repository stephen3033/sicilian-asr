import json

import pytest

from common.errors import ResourceError
from common.settings import SCN_RECIPE
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
    recipe = tmp_path / SCN_RECIPE
    recipe.parent.mkdir(parents=True)
    recipe.write_text("{}")
    output = tmp_path / "outputs"
    output.mkdir()
    (output / "test.json").write_text("{}")
    with pytest.raises(ResourceError, match="frozen"):
        run(tmp_path, output, stage)


@pytest.mark.parametrize("tokens, reached", [(11441, 1), (30000, 2)])
def test_report_uses_current_checkpoint_and_video_counts(tmp_path, tokens, reached):
    from pipeline.scn_study import report_study

    trial = {
        "tokens": tokens,
        "order": 3,
        "vocabulary_size": 100,
        "oov_rate": 0.1,
        "acoustic_scale": 1,
        "wer": 0.5,
        "cer": 0.3,
        "graph_bytes": 1024,
    }
    selection = {"coverage_target_met": False, "candidate": trial, "acoustic_scale": 1}
    test = {
        **trial,
        "acoustic_seconds": 1,
        "decode_seconds": 1,
        "process_peak_rss_bytes": 1024,
        "results": [
            {
                "video": str(i),
                "metrics": {"wer": {"rate": 0.5}, "cer": {"rate": 0.3}},
                "coverage": {"fraction": 0.9},
                "path": f"video-{i}",
            }
            for i in range(3)
        ],
    }
    for name, value in [
        ("selection.json", selection),
        ("development-results.json", [trial]),
        ("test.json", test),
    ]:
        (tmp_path / name).write_text(json.dumps(value))
    report_study(
        tmp_path, tmp_path, {"checkpoints": [10000, 25000], "target_oov": 0.05}, {}, {}
    )
    report = (tmp_path / "report.md").read_text()
    assert f"checkpoints reached: {reached}/2" in report
    assert "Final 3-video WER" in report
    assert "exhausted below 25k" not in report
