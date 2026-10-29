"""Preparation and local-only transcription commands."""

import argparse
import sys
from pathlib import Path

from common.errors import ResourceError
from common.resources import load_resources
from pipeline.config import load_config


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    poc = commands.add_parser(
        "scn-poc", help="Resumable DOS24 Sicilian training and evaluation"
    )
    poc.add_argument("--root", type=Path, default=Path.cwd())
    poc.add_argument("--output", type=Path, default=Path("outputs/scn_poc"))
    poc.add_argument(
        "--stage",
        choices=["all", "acquire", "corpus", "train", "development", "test", "report"],
        default="all",
    )
    from pipeline.scn_experiment import STAGES

    improve = commands.add_parser(
        "scn-improve", help="DOS24 corpus and ZIPA/WFST experiments"
    )
    improve.add_argument("--root", type=Path, default=Path.cwd())
    improve.add_argument(
        "--config", type=Path, default=Path("configs/scn-improve.json")
    )
    improve.add_argument("--output", type=Path)
    improve.add_argument("--stage", choices=["all", *STAGES], default="all")
    for name in ("prepare", "transcribe"):
        command = commands.add_parser(name)
        command.add_argument(
            "--config",
            required=True,
            help="ita, scn, ita_scn, or a YAML configuration path",
        )
        command.add_argument("--root", type=Path, default=Path.cwd())
        if name == "transcribe":
            command.add_argument("--input", type=Path, required=True)
            command.add_argument("--reference", type=Path)
            command.add_argument("--output", type=Path, required=True)
            command.add_argument("--audio-stream", type=int)
            command.add_argument("--export-probabilities", action="store_true")
            command.add_argument(
                "--diagnostics",
                action="store_true",
                help="Export probabilities, pronunciation and competing path scores",
            )
    args = parser.parse_args()
    try:
        root = args.root.resolve()
        if args.command == "scn-improve":
            from pipeline.scn_experiment import main as improve_main

            return improve_main(
                [
                    "--root",
                    str(root),
                    "--config",
                    str(args.config),
                    *(
                        ["--output", str(args.output)]
                        if args.output is not None
                        else []
                    ),
                    "--stage",
                    args.stage,
                ]
            )
        if args.command == "scn-poc":
            from pipeline.scn_poc import run

            run(root, args.output, args.stage)
            return 0
        manifest = load_resources(root)
        config = load_config(args.config, root, manifest)
        if args.command == "prepare":
            from pipeline.languages import backend

            language, _ = backend(config)
            print(language.prepare(root, config, manifest))
        else:
            from pipeline.transcribe import transcribe

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
