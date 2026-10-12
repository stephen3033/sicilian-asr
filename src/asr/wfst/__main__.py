"""Compile a prepared graph in an isolated native worker process."""

import argparse
import json
from pathlib import Path

import torch

from asr.wfst.graph import compile_graph


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    directory = parser.parse_args().directory
    inputs = json.loads((directory / "compile_input.json").read_text())
    lexicon = json.loads((directory / "lexicon_selected.json").read_text())
    torch.set_num_threads(4)
    compile_graph(directory, lexicon, inputs["tokens"], inputs["settings"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
