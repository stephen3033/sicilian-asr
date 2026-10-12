"""Resume acquisition, DOS24 corpus preparation, training and held-out evaluation."""

import json
import subprocess
from pathlib import Path

from common.errors import ResourceError
from common.settings import SCN_RESOURCES
from common.utils import write_json


def run(root: Path, output: Path, stage: str):
    output = output if output.is_absolute() else root / output
    output.mkdir(parents=True, exist_ok=True)
    recipe = json.loads((root / SCN_RESOURCES / "poc.json").read_text())
    stages = (
        ["acquire", "corpus", "train", "development", "test", "report"]
        if stage == "all"
        else [stage]
    )
    for current in stages:
        if (
            current in {"corpus", "train", "development"}
            and (output / "test.json").exists()
        ):
            raise ResourceError(
                "Final test is frozen; use a new output directory for a new experiment"
            )
        print(f"Sicilian POC: {current}", flush=True)
        if current == "acquire":
            from media_input.youtube import acquire

            failures = {}
            for split, videos in recipe["splits"].items():
                for video in videos:
                    print(f"{split}: {video}", flush=True)
                    try:
                        acquire(
                            video,
                            root / ".cache/scn/youtube" / video,
                            channel_id=recipe["channel_id"],
                        )
                    except (
                        ResourceError,
                        OSError,
                        subprocess.CalledProcessError,
                    ) as error:
                        failures[video] = str(error)
            write_json(
                output / "acquisition.json",
                {"failures": failures, "splits": recipe["splits"]},
            )
            if failures:
                raise ResourceError(f"Acquisition incomplete: {failures}")
        elif current == "corpus":
            from asr.scn.dataset import prepare_corpus

            prepare_corpus(root, output, recipe)
        elif current == "train":
            from asr.scn.training import train_candidates

            train_candidates(root, output, recipe)
        else:
            from pipeline.scn_study import study

            study(root, output, recipe, current)
