"""Select a fixed neutral test recording before decoding; optionally run acceptance."""

import argparse
import csv
import io
import subprocess
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from common.resources import download_resource, load_resources
from common.utils import command, fingerprints, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--config", default="ita")
    parser.add_argument("--diagnostics", action="store_true")
    parser.add_argument(
        "--data-output",
        type=Path,
        default=Path("data/acceptance/emozionalmente"),
        help="Directory for the selected source recording and independent reference",
    )
    parser.add_argument("--output", type=Path, default=Path("outputs/ita_acceptance"))
    args = parser.parse_args()
    root = Path.cwd()
    spec = load_resources(root)["resources"]["emozionalmente"]
    archive = download_resource(root, spec)
    destination = root / args.data_output
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(root / spec["path"]) as data:
        table = data.read("emozionalmente/metadata/split/test.csv")
        rows = list(csv.DictReader(io.StringIO(table.decode("utf-8"))))
        selected = min(
            (r for r in rows if r["emotion_expressed"] == "neutrality"),
            key=lambda r: r["file_name"],
        )
        wav = destination / selected["file_name"]
        wav.write_bytes(data.read("emozionalmente/audio/" + selected["file_name"]))
        (destination / "test.csv").write_bytes(table)
        (destination / "README.publisher.txt").write_bytes(
            data.read("emozionalmente/README.txt")
        )
    reference = destination / "reference.txt"
    reference.write_text(selected["sentence"] + "\n", encoding="utf-8")
    mp4 = destination / "sample.mp4"
    command(
        [
            "ffmpeg",
            "-nostdin",
            "-v",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "color=c=black:s=160x90:r=25",
            "-i",
            str(wav),
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-b:a",
            "96k",
            "-shortest",
            str(mp4),
        ]
    )
    write_json(
        destination / "selection.json",
        {
            "selected_at": datetime.now(timezone.utc).isoformat(),
            "rule": "Lexically first filename among neutrality recordings in publisher test split; selected before decoding",
            "dataset": spec,
            "archive": archive,
            "selected": selected,
            "files": {
                p.name: fingerprints(p)
                for p in (wav, reference, mp4, destination / "test.csv")
            },
            "reference_policy": "Publisher sentence unchanged, including digit 2; no number verbalization or correction from hypothesis",
        },
    )
    print(f"Selected {selected['file_name']}: {selected['sentence']}", flush=True)
    if args.run:
        return subprocess.call(
            [
                sys.executable,
                "-m",
                "pipeline",
                "transcribe",
                "--config",
                str(args.config),
                "--root",
                str(root),
                "--input",
                str(mp4),
                "--reference",
                str(reference),
                "--output",
                str(args.output),
                *(["--diagnostics"] if args.diagnostics else []),
            ]
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
