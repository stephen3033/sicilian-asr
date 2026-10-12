import numpy as np
import pytest
import soundfile as sf

from common.errors import ResourceError
from media_input.media import extract_audio


def test_audio_conversion_and_stream_selection(tmp_path):
    p = tmp_path / "stereo.wav"
    t = np.arange(24000) / 48000
    sf.write(
        p,
        np.stack(
            [0.2 * np.sin(2 * np.pi * 220 * t), 0.1 * np.sin(2 * np.pi * 440 * t)],
            axis=1,
        ),
        48000,
    )
    audio, info = extract_audio(p, tmp_path / "mono.wav")
    assert audio.shape == (8000,) and info["selected_stream"] == 0
    assert info["duration_seconds"] == 0.5
    with pytest.raises(ResourceError, match="stream"):
        extract_audio(p, tmp_path / "bad.wav", 4)
