import numpy as np
import pytest

from audio.segmentation import segment_audio
from common.errors import ResourceError


def test_sample_exact_segmentation():
    x = np.ones(55 * 16000, dtype=np.float32) * 0.1
    x[19 * 16000 : 20 * 16000] = 0
    segments = segment_audio(x, {}, origin=4.5)
    assert segments[0].boundary == "silence"
    assert segments[0].source_start_seconds == 4.5
    assert segments[-1].end_sample == len(x)
    assert all(a.end_sample == b.start_sample for a, b in zip(segments, segments[1:]))
    assert sum(s.end_sample - s.start_sample for s in segments) == len(x)
    assert all(s.end_sample - s.start_sample <= 24 * 16000 for s in segments)
    assert any(s.forced_cut for s in segments)


def test_empty_audio_and_invalid_segmentation_settings():
    with pytest.raises(ResourceError):
        segment_audio(np.array([]), {})
    with pytest.raises(ResourceError):
        segment_audio(np.zeros(8000, dtype=np.float32), {"max_seconds": 10})
