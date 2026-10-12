from importlib import import_module


def test_package_imports():
    for name in (
        "common",
        "media_input",
        "audio",
        "ipa",
        "asr.ita",
        "asr.scn",
        "asr.wfst",
        "evaluation",
        "pipeline",
    ):
        assert import_module(name) is not None
