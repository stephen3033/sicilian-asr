"""Preparation and local-only transcription commands."""

import argparse
import json
from pathlib import Path
import sys

from sicilian_asr.config import load_config
from sicilian_asr.resources import ResourceError, load_manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "transcribe"):
        command = commands.add_parser(name)
        command.add_argument("--config", type=Path, required=True)
        command.add_argument("--root", type=Path, default=Path.cwd())
        if name == "transcribe":
            command.add_argument("--input", type=Path, required=True)
            command.add_argument("--reference", type=Path)
            command.add_argument("--output", type=Path, required=True)
            command.add_argument("--audio-stream", type=int)
            command.add_argument("--export-probabilities", action="store_true")
            command.add_argument("--diagnostics", action="store_true", help="Export probabilities, pronunciation and competing path scores")
    worker = commands.add_parser("_compile", help=argparse.SUPPRESS)
    worker.add_argument("--directory", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "_compile":
            import torch
            from sicilian_asr.decoding.graph import compile_graph
            inputs = json.loads((args.directory/"compile_input.json").read_text())
            torch.set_num_threads(4)
            compile_graph(args.directory, json.loads((args.directory/"lexicon_selected.json").read_text()), inputs["tokens"], inputs["settings"])
            return 0
        root = args.root.resolve()
        manifest = load_manifest(root/"resources/manifest.yaml")
        config = load_config(args.config, root, manifest)
        if args.command == "prepare":
            from sicilian_asr.preparation import prepare
            print(prepare(root, config, manifest))
        else:
            from sicilian_asr.pipeline import transcribe
            if args.audio_stream is not None:
                config["audio"]["stream"] = args.audio_stream
            if args.export_probabilities or args.diagnostics:
                config["exports"]["probabilities"] = True
            if args.diagnostics:
                config["decoder"]["diagnostic_paths"] = 32
            transcribe(root, config, manifest, args.input, args.reference, args.output)
            print(f"Completed: {args.output}")
        return 0
    except (ResourceError, OSError, ValueError, RuntimeError, KeyError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
