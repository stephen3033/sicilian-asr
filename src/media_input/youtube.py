"""Public, uploaded subtitle/audio snapshots, with explicit track provenance."""

import json
import math
import subprocess
import sys
from pathlib import Path

from common.errors import ResourceError
from common.utils import fingerprints, write_json


def parse_json3(data: dict) -> list[dict]:
    """Keep authored repetition; discard only identical events at identical times."""
    cues, seen = [], set()
    for event in data.get("events", []):
        text = "".join(part.get("utf8", "") for part in event.get("segs", [])).strip()
        if not text:
            continue
        start = event.get("tStartMs")
        duration = event.get("dDurationMs")
        if (
            not isinstance(start, (int, float))
            or not isinstance(duration, (int, float))
            or not math.isfinite(start)
            or not math.isfinite(duration)
            or min(start, duration) < 0
        ):
            raise ResourceError("Subtitle event has invalid timing")
        key = (start, duration, text)
        if key not in seen:
            cues.append(
                {
                    "start_seconds": start / 1000,
                    "end_seconds": (start + duration) / 1000,
                    "text": text,
                }
            )
            seen.add(key)
    if not cues:
        raise ResourceError("No timed caption text")
    if any(a["start_seconds"] > b["start_seconds"] for a, b in zip(cues, cues[1:])):
        raise ResourceError("Caption events are out of order")
    return cues


def acquire(
    video_id: str, directory: Path, language: str = "scn", channel_id: str | None = None
) -> dict:
    """Download one explicit uploaded track and original compressed audio."""
    if len(video_id) != 11 or any(
        c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
        for c in video_id
    ):
        raise ResourceError("Invalid YouTube video ID")
    directory.mkdir(parents=True, exist_ok=True)
    receipt_path = directory / "receipt.json"
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text())
        if receipt.get("video_id") != video_id or receipt.get("language") != language:
            raise ResourceError(
                "YouTube snapshot does not match the requested video/track"
            )
        for name, digest in receipt["artifacts"].items():
            if fingerprints(directory / name)["sha256"] != digest:
                raise ResourceError(f"Changed YouTube snapshot: {name}")
        if (
            channel_id
            and json.loads((directory / "source.json").read_text())["channel_id"]
            != channel_id
        ):
            raise ResourceError(f"Unexpected channel for {video_id}")
        return receipt
    url = f"https://www.youtube.com/watch?v={video_id}"
    command = [
        sys.executable,
        "-m",
        "yt_dlp",
        "--no-playlist",
        "--no-progress",
        "--no-cache-dir",
    ]
    metadata = subprocess.run(
        command + ["--skip-download", "--dump-single-json", url],
        capture_output=True,
        text=True,
        check=True,
    )
    info = json.loads(metadata.stdout)
    if channel_id and info.get("channel_id") != channel_id:
        raise ResourceError(f"Unexpected channel for {video_id}")
    if not info.get("subtitles", {}).get(language):
        raise ResourceError(
            f"{video_id}: no uploaded {language} captions; automatic tracks are ineligible"
        )
    # Signed media URLs are deliberately omitted from the durable public metadata.
    selected = {
        k: info.get(k)
        for k in (
            "id",
            "title",
            "channel",
            "channel_id",
            "upload_date",
            "duration",
            "license",
            "description",
            "webpage_url",
        )
    }
    write_json(directory / "source.json", selected)
    for audio_format in ("bestaudio", "bestaudio[ext=m4a]/bestaudio"):
        downloaded = subprocess.run(
            command
            + [
                "-f",
                audio_format,
                "--write-subs",
                "--no-write-auto-subs",
                "--sub-langs",
                language,
                "--sub-format",
                "json3",
                "-o",
                str(directory / "source.%(ext)s"),
                url,
            ]
        )
        if downloaded.returncode == 0:
            break
    else:
        raise ResourceError(
            f"Public audio download failed for {video_id}; incomplete snapshot retained for retry"
        )
    caption = directory / f"source.{language}.json3"
    cues = parse_json3(json.loads(caption.read_text()))
    write_json(directory / "cues.json", cues)
    audio = [
        p
        for p in directory.glob("source.*")
        if p.suffix in {".m4a", ".webm", ".opus", ".ogg", ".mp3"}
    ]
    if len(audio) != 1:
        raise ResourceError(f"Expected one audio artifact for {video_id}")
    receipt = {
        "video_id": video_id,
        "url": url,
        "language": language,
        "track_kind": "uploaded",
        "authorship": "uploaded track; individual caption author not independently verified",
        "audio": audio[0].name,
        "caption": caption.name,
        "yt_dlp_version": __import__(
            "yt_dlp.version", fromlist=["__version__"]
        ).__version__,
        "artifacts": {
            p.name: fingerprints(p)["sha256"]
            for p in directory.iterdir()
            if p.is_file()
        },
    }
    write_json(receipt_path, receipt)
    return receipt
