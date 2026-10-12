import html
import re
from pathlib import Path

from common.errors import ResourceError
from common.text import normalize

TIME = r"(?:\d{2,}:)?\d{2}:\d{2}[.,]\d{3}"


CUE = re.compile(rf"^({TIME})\s+-->\s+({TIME})(?:\s+.*)?$")


def timestamp(value: str) -> float:
    parts = value.replace(",", ".").split(":")
    seconds = float(parts[-1])
    minutes = int(parts[-2])
    hours = int(parts[-3]) if len(parts) == 3 else 0
    if seconds >= 60 or minutes >= 60:
        raise ResourceError(f"Invalid subtitle timestamp: {value}")
    return hours * 3600 + minutes * 60 + seconds


def read_reference(path: Path) -> dict:
    text = (
        path.read_text(encoding="utf-8-sig").replace("\r\n", "\n").replace("\r", "\n")
    )
    if path.suffix.lower() == ".txt":
        return {
            "format": "text",
            "text": text,
            "normalized": normalize(text),
            "cues": [],
        }
    kind = path.suffix.lower().lstrip(".")
    if kind not in {"srt", "vtt"}:
        raise ResourceError("Reference must be SRT, VTT, or UTF-8 .txt")
    blocks = re.split(r"\n\s*\n", text.strip())
    cues, metadata = [], []
    for block in blocks:
        lines = block.splitlines()
        if not lines:
            continue
        if kind == "vtt" and (
            lines[0].startswith("WEBVTT")
            or re.match(r"^(NOTE|STYLE|REGION)(?:\s|$)", lines[0])
        ):
            metadata.append(block)
            continue
        offset = 0 if "-->" in lines[0] else 1
        match = CUE.fullmatch(lines[offset].strip()) if len(lines) > offset else None
        if match is None:
            raise ResourceError(f"Malformed {kind.upper()} cue: {lines[0]}")
        start, end = map(timestamp, match.groups())
        if end < start:
            raise ResourceError("Subtitle cue ends before it starts")
        spoken = " ".join(lines[offset + 1 :])
        spoken = re.sub(r"<[^>]*>|\{\\[^}]*\}", "", spoken)
        spoken = html.unescape(spoken)
        cues.append(
            {
                "id": lines[0] if offset else None,
                "start_seconds": start,
                "end_seconds": end,
                "text": spoken,
            }
        )
    spoken = " ".join(c["text"] for c in cues)
    return {
        "format": kind,
        "text": spoken,
        "normalized": normalize(spoken),
        "cues": cues,
        "metadata": metadata,
    }
