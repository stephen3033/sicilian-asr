"""Repository locations and minimal on-disk configuration fixtures."""

import shutil
from pathlib import Path

from common.settings import CONFIG_PATHS, MANIFEST_PATHS, MODEL_CONFIG

ROOT = Path(__file__).resolve().parents[1]


def copy_configuration(destination: Path) -> None:
    for relative in [*MANIFEST_PATHS, MODEL_CONFIG, *CONFIG_PATHS.values()]:
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, target)
