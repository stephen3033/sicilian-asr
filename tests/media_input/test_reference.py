import pytest

from common.errors import ResourceError
from media_input.reference import read_reference


def test_subtitles_preserve_numeric_spoken_lines(tmp_path):
    path = tmp_path / "test.srt"
    path.write_text(
        "1\n00:00:01,500 --> 00:00:03,000\n<i>Ho preso</i>\n6\n\n2\n00:00:04,000 --> 00:00:05,000\nL’acqua &amp; il tè\n"
    )
    ref = read_reference(path)
    assert ref["normalized"] == "ho preso 6 l'acqua il tè"
    assert ref["cues"][0]["start_seconds"] == 1.5
    path = tmp_path / "test.vtt"
    path.write_text(
        "WEBVTT\nKind: captions\nLanguage: it\n\nNOTE hidden\n123\n\nSTYLE\n::cue { color: red; }\n\nidentifier\n00:01.500 --> 00:03.000 align:start\n<v Marco><c.red>Oggi</c> 2 <00:02.000>anni\n"
    )
    assert read_reference(path)["normalized"] == "oggi 2 anni"


@pytest.mark.parametrize(
    "text",
    [
        "1\nbroken\nfoo",
        "1\n00:00:02,000 --> 00:00:01,000\nfoo",
        "1\n00:61:00,000 --> 01:02:00,000\nfoo",
    ],
)
def test_invalid_subtitle_rejected(tmp_path, text):
    p = tmp_path / "bad.srt"
    p.write_text(text)
    with pytest.raises(ResourceError):
        read_reference(p)
