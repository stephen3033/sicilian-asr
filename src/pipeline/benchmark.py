"""Run the Italian study with frozen settings and isolated workers."""

import argparse
import json
import subprocess
import sys
import tempfile
from copy import deepcopy
from pathlib import Path

import torch

from asr.ita.benchmark import (
    SCALES,
    VARIANTS,
    initialize,
    posteriors,
    read_rows,
    report,
    result_path,
    run_decode,
    select_scale,
)
from asr.ita.prepare import prepare
from common.errors import ResourceError
from common.resources import load_resources
from common.utils import write_json
from pipeline.config import load_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default="ita")
    parser.add_argument(
        "--baseline",
        type=Path,
        help="Verified unadapted graph; default: prepare one from the selected config",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--stage",
        choices=["all", "acoustic", "variant", "test", "report"],
        default="all",
    )
    parser.add_argument("--variant", choices=list(VARIANTS))
    args = parser.parse_args()
    root = Path.cwd()
    manifest = load_resources(root)
    config = load_config(args.config, root, manifest)
    torch.set_num_threads(config["acoustic"].get("threads", 4))
    if args.baseline is None:
        previous = args.output / "study.json"
        if previous.exists():
            args.baseline = Path(json.loads(previous.read_text())["baseline"])
        else:
            baseline_config = deepcopy(config)
            baseline_config["decoder"].update(
                pronunciation_variants=False, contractions=False
            )
            args.baseline = prepare(root, baseline_config, manifest)
    if args.stage == "all":
        initialize(root, config, manifest, args.output, args.baseline)
        command = [
            sys.executable,
            "-m",
            "pipeline.benchmark",
            "--config",
            str(args.config),
            "--baseline",
            str(args.baseline),
            "--output",
            str(args.output),
        ]
        subprocess.run(command + ["--stage", "acoustic"], check=True)
        for variant in VARIANTS:
            subprocess.run(
                command + ["--stage", "variant", "--variant", variant], check=True
            )
        select_scale(args.output)
        for variant in ("baseline", "combined"):
            subprocess.run(
                command + ["--stage", "test", "--variant", variant], check=True
            )
        report(args.output)
        return
    study = json.loads((args.output / "study.json").read_text())
    # Workers use the frozen config, even if the user edits the default config.
    config = study["config"]
    rows = study["selection"]
    if args.stage == "acoustic":
        posteriors(
            root, config, manifest, root / study["dataset"]["path"], rows, args.output
        )
        return
    if args.stage == "report":
        report(args.output)
        return
    if args.variant is None:
        raise ResourceError("--variant is required for variant/test stages")
    variant = args.variant
    pronunciation, contractions = VARIANTS[variant]
    config["decoder"].update(
        pronunciation_variants=pronunciation, contractions=contractions
    )
    split = "test" if args.stage == "test" else "dev"
    scales = (
        [json.loads((args.output / "selection.json").read_text())["acoustic_scale"]]
        if split == "test" and variant == "combined"
        else SCALES
        if variant == "combined"
        else [1.0]
    )
    if split == "test" and not (args.output / "selection.json").exists():
        raise ResourceError("Freeze development selection before testing")
    if all(
        len(read_rows(result_path(args.output, variant, split, scale)))
        == len(rows[split])
        for scale in scales
    ):
        return
    if variant == "baseline":
        directory = Path(study["baseline"])
        run_decode(
            directory,
            config["decoder"],
            rows[split],
            args.output,
            variant,
            split,
            scales,
        )
    elif variant == "combined":
        directory = prepare(root, config, manifest)
        write_json(
            args.output / "combined_graph.json",
            {"directory": str(directory), "config": config},
        )
        run_decode(
            directory,
            config["decoder"],
            rows[split],
            args.output,
            variant,
            split,
            scales,
        )
    else:
        with tempfile.TemporaryDirectory(
            prefix="ita-ablation-", dir=root / ".cache"
        ) as temp:
            config["decoder"]["graph_path"] = str(Path(temp).relative_to(root))
            directory = prepare(root, config, manifest)
            receipt = json.loads((directory / "prepared.json").read_text())
            write_json(args.output / f"{variant}_graph.json", receipt)
            run_decode(
                directory,
                config["decoder"],
                rows[split],
                args.output,
                variant,
                split,
                scales,
            )


if __name__ == "__main__":
    main()
