"""Compile a checksummed WFST candidate without historical study dependencies."""

import shutil
import sys
from pathlib import Path

from common.artifacts import (
    digest,
    read,
    run_native,
    seal,
    sha,
    space_check,
    verified_files,
)
from common.errors import ResourceError
from common.utils import write_json


def compile_graph(directory, lexicon, arpa, tokens, settings):
    identity = {
        "lexicon": digest(lexicon),
        "arpa": sha(arpa),
        "tokens": tokens,
        "settings": settings,
        "compiler": sha(Path(__file__).with_name("graph.py")),
    }
    receipt = directory / "graph.json"
    if receipt.exists():
        saved = read(receipt)
        if saved["key"] != digest(identity):
            raise ResourceError(f"Graph inputs changed: {directory}")
        verified_files(directory, saved)
        return directory
    directory.mkdir(parents=True, exist_ok=True)
    space_check(directory, required=1024**3)
    shutil.copyfile(arpa, directory / "restricted.arpa")
    write_json(directory / "lexicon_selected.json", lexicon)
    write_json(
        directory / "compile_input.json", {"tokens": tokens, "settings": settings}
    )
    run_native(
        [sys.executable, "-m", "asr.wfst", "--directory", str(directory)],
        directory / "compile.log",
    )
    seal(
        directory,
        "graph.json",
        identity,
        [p.name for p in directory.iterdir() if p.is_file() and p.name != "graph.json"],
    )
    return directory
