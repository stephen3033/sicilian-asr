"""Explicit monolingual dispatch; language packages never import orchestration."""

from common.errors import ResourceError


def backend(config):
    if config["languages"] == ["ita"]:
        from asr.ita import evaluation, prepare
    elif config["languages"] == ["scn"]:
        from asr.scn import evaluation, prepare
    else:
        raise ResourceError("Bilingual decoding is not implemented")
    return prepare, evaluation.evaluate


def text_policy(config):
    if config["languages"] == ["scn"]:
        from asr.scn.orthography import RULES, normalize
    else:
        from common.text import TEXT_RULES as RULES
        from common.text import normalize
    return normalize, RULES
