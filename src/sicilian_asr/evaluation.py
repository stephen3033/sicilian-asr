"""Cue-aware reference parsing and reproducible edit counts."""

from collections import Counter
import html
from pathlib import Path
import re

from sicilian_asr.resources import ResourceError
from sicilian_asr.text import TEXT_RULES, normalize
from sicilian_asr.spoken import SPOKEN_RULES, spoken_form

TIME = r"(?:\d{2,}:)?\d{2}:\d{2}[.,]\d{3}"
CUE = re.compile(rf"^({TIME})\s+-->\s+({TIME})(?:\s+.*)?$")


def timestamp(value: str) -> float:
    parts = value.replace(",", ".").split(":")
    seconds = float(parts[-1])
    minutes = int(parts[-2])
    hours = int(parts[-3]) if len(parts) == 3 else 0
    if seconds >= 60 or minutes >= 60:
        raise ResourceError(f"Invalid subtitle timestamp: {value}")
    return hours*3600 + minutes*60 + seconds


def read_reference(path: Path) -> dict:
    text = path.read_text(encoding="utf-8-sig").replace("\r\n", "\n").replace("\r", "\n")
    if path.suffix.lower() == ".txt":
        return {"format": "text", "text": text, "normalized": normalize(text), "cues": []}
    kind = path.suffix.lower().lstrip(".")
    if kind not in {"srt", "vtt"}:
        raise ResourceError("Reference must be SRT, VTT, or UTF-8 .txt")
    blocks = re.split(r"\n\s*\n", text.strip())
    cues, metadata = [], []
    for block in blocks:
        lines = block.splitlines()
        if not lines:
            continue
        if kind == "vtt" and (lines[0].startswith("WEBVTT") or re.match(r"^(NOTE|STYLE|REGION)(?:\s|$)", lines[0])):
            metadata.append(block)
            continue
        offset = 0 if "-->" in lines[0] else 1
        match = CUE.fullmatch(lines[offset].strip()) if len(lines) > offset else None
        if match is None:
            raise ResourceError(f"Malformed {kind.upper()} cue: {lines[0]}")
        start, end = map(timestamp, match.groups())
        if end < start:
            raise ResourceError("Subtitle cue ends before it starts")
        spoken = " ".join(lines[offset+1:])
        spoken = re.sub(r"<[^>]*>|\{\\[^}]*\}", "", spoken)
        spoken = html.unescape(spoken)
        cues.append({"id": lines[0] if offset else None, "start_seconds": start, "end_seconds": end, "text": spoken})
    spoken = " ".join(c["text"] for c in cues)
    return {"format": kind, "text": spoken, "normalized": normalize(spoken), "cues": cues, "metadata": metadata}


def edits(reference: list[str], hypothesis: list[str]) -> tuple[dict, list]:
    import kaldialign

    epsilon = "\0"
    if epsilon in reference or epsilon in hypothesis:
        raise ResourceError("NUL is not a supported reference symbol")
    alignment = kaldialign.align(reference, hypothesis, epsilon)
    counts = Counter(substitutions=0, deletions=0, insertions=0, correct=0)
    rows = []
    for ref, hyp in alignment:
        operation = "insertions" if ref == epsilon else "deletions" if hyp == epsilon else "correct" if ref == hyp else "substitutions"
        counts[operation] += 1
        rows.append({"operation": operation, "reference": None if ref == epsilon else ref, "hypothesis": None if hyp == epsilon else hyp})
    errors = counts["substitutions"] + counts["deletions"] + counts["insertions"]
    return {**counts, "errors": errors, "reference_units": len(reference), "hypothesis_units": len(hypothesis),
            "rate": errors / len(reference) if reference else None, "undefined_reason": None if reference else "empty reference"}, rows


def oov(words: list[str], vocabulary: set[str]) -> dict:
    missing = Counter(w for w in words if w not in vocabulary)
    return {"tokens": sum(missing.values()), "types": len(missing), "reference_tokens": len(words), "words": dict(sorted(missing.items()))}


def evaluate(hypothesis: str, reference: dict, full_lexicon: set[str], selected: set[str]) -> tuple[dict, str]:
    hyp, ref = normalize(hypothesis), reference["normalized"]
    wer, alignment = edits(ref.split(), hyp.split())
    cer, _ = edits(list(ref.replace(" ", "")), list(hyp.replace(" ", "")))
    metrics = {"normalization": TEXT_RULES, "wer": wer, "cer": cer,
               "oov_full_supported_lexicon": oov(ref.split(), full_lexicon), "oov_decoding_vocabulary": oov(ref.split(), selected),
               "per": None, "per_reason": "No IPA reference", "word_alignment": alignment}
    spoken_ref = spoken_form(reference.get("text", reference["normalized"]))
    spoken_hyp = spoken_form(hypothesis)
    spoken_wer, spoken_alignment = edits(spoken_ref["text"].split(), spoken_hyp["text"].split())
    spoken_cer, _ = edits(list(spoken_ref["text"].replace(" ", "")), list(spoken_hyp["text"].replace(" ", "")))
    metrics["spoken_form"] = {"normalization": SPOKEN_RULES, "reference": spoken_ref, "hypothesis": spoken_hyp,
        "wer": spoken_wer, "cer": spoken_cer, "word_alignment": spoken_alignment,
        "oov_full_supported_lexicon": oov(spoken_ref["text"].split(), full_lexicon),
        "oov_decoding_vocabulary": oov(spoken_ref["text"].split(), selected)}
    metrics["oov_causes"] = [{"word": w, "count": count,
        "cause": "absent_from_supported_lexicon" if w not in full_lexicon else "outside_decoding_vocabulary",
        "spoken_form": spoken_form(w)["text"]} for w, count in metrics["oov_decoding_vocabulary"]["words"].items()]
    comparison = f"REFERENCE: {ref}\nHYPOTHESIS: {hyp}\n\noperation\treference\thypothesis\n"
    comparison += "".join(f"{a['operation']}\t{a['reference'] or '∅'}\t{a['hypothesis'] or '∅'}\n" for a in alignment)
    comparison += f"\nSPOKEN-FORM REFERENCE: {spoken_ref['text']}\nSPOKEN-FORM HYPOTHESIS: {spoken_hyp['text']}\n"
    return metrics, comparison
