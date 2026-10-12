from importlib.metadata import version


def package_versions() -> dict:
    return {
        n: version(n)
        for n in (
            "sicilian-asr",
            "torch",
            "torchaudio",
            "k2",
            "kaldilm",
            "kenlm",
            "numpy",
            "lhotse",
            "sentencepiece",
            "soundfile",
            "kaldialign",
            "pyyaml",
        )
    }
