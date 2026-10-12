"""Reproducible Italian repair ablations with frozen development/test selection.

Run from the repository root. Acoustic posteriors and completed decode rows are
reused on restart. Temporary ablation graphs are removed after their dev run;
the combined graph is kept. Test results never participate in scale selection.
"""

import csv
import hashlib
import io
import json
import tempfile
import time
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import yaml

from asr.ita.evaluation import evaluate
from asr.wfst.cache import verify_cache
from common.errors import ResourceError
from common.resources import verify_file
from common.text import normalize
from common.utils import fingerprints, write_json
from ipa.diagnostics import greedy_ctc
from ipa.types import AcousticOutput
from ipa.zipa import ZipaAcoustic

SCALES = [0.5, 0.75, 1.0, 1.5, 2.0]
VARIANTS = {
    "baseline": (False, False),
    "pronunciation": (True, False),
    "contractions": (False, True),
    "combined": (True, True),
}
EXCLUDED_TEST = "1612969363951.wav"


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def dataset(root, manifest):
    spec = manifest["resources"]["emozionalmente"]
    archive = root / spec["path"]
    verify_file(archive, spec)
    selected = {}
    with zipfile.ZipFile(archive) as z:
        for split in ("dev", "test"):
            rows = csv.DictReader(
                io.StringIO(
                    z.read(f"emozionalmente/metadata/split/{split}.csv").decode()
                )
            )
            selected[split] = sorted(
                [
                    r
                    for r in rows
                    if r["emotion_expressed"] == "neutrality"
                    and not (split == "test" and r["file_name"] == EXCLUDED_TEST)
                ],
                key=lambda r: r["file_name"],
            )
    if {r["actor"] for r in selected["dev"]} & {r["actor"] for r in selected["test"]}:
        raise ResourceError("Development and test actors overlap")
    return archive, selected


def initialize(root, config, manifest, output, baseline):
    archive, rows = dataset(root, manifest)
    sources = {
        str(p.relative_to(root)): fingerprints(p)["sha256"]
        for p in sorted((root / "src").rglob("*.py"))
    }
    sources["src/asr/ita/benchmark.py"] = fingerprints(Path(__file__))["sha256"]
    study = {
        "schema": 1,
        "config": config,
        "baseline": str(baseline.resolve()),
        "baseline_receipt": fingerprints(baseline / "prepared.json")["sha256"],
        "sources": sources,
        "dataset": manifest["resources"]["emozionalmente"],
        "selection": rows,
        "selection_sha256": digest(rows),
        "scales": SCALES,
        "selection_policy": "All publisher neutral dev/test rows; exclude previously inspected acceptance clip from test",
        "reference_policy": "Publisher text unchanged, including E’ spellings; report strict and number-verbalized metrics",
    }
    output.mkdir(parents=True, exist_ok=True)
    path = output / "study.json"
    if path.exists() and json.loads(path.read_text()) != study:
        raise ResourceError("Study inputs/code changed; use a fresh output directory")
    write_json(path, study)
    (output / "frozen-config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    return archive, rows


def posteriors(root, config, manifest, archive, rows, output):
    directory = output / "probabilities"
    directory.mkdir(exist_ok=True)
    all_rows = rows["dev"] + rows["test"]
    pending = [
        r for r in all_rows if not (directory / (r["file_name"] + ".npz")).exists()
    ]
    if not pending:
        return
    model = ZipaAcoustic(root, config["acoustic"], manifest)
    write_json(output / "preprocessing.json", model.preprocessing)
    with zipfile.ZipFile(archive) as z:
        for i, row in enumerate(pending):
            payload = z.read("emozionalmente/audio/" + row["file_name"])
            audio, sr = sf.read(io.BytesIO(payload), dtype="float32")
            if sr != 16000 or audio.ndim != 1:
                raise ResourceError("Benchmark expects publisher 16 kHz mono WAV")
            acoustic = model.infer([torch.from_numpy(audio)])
            path = directory / (row["file_name"] + ".npz")
            with tempfile.NamedTemporaryFile(
                dir=directory, suffix=".npz", delete=False
            ) as temp:
                temporary = Path(temp.name)
            try:
                np.savez_compressed(
                    temporary,
                    log_probs=acoustic.log_probs[0, : int(acoustic.lengths[0])].numpy(),
                    tokens=np.asarray(acoustic.tokens),
                    blank_id=0,
                    frame_length=int(acoustic.lengths[0]),
                    ipa_greedy=greedy_ctc(acoustic)[0],
                    audio_sha256=hashlib.sha256(payload).hexdigest(),
                    duration=len(audio) / sr,
                )
                temporary.replace(path)
            finally:
                temporary.unlink(missing_ok=True)
            if (i + 1) % 10 == 0 or i + 1 == len(pending):
                print(f"Acoustic {i + 1}/{len(pending)}", flush=True)
    write_json(
        output / "probabilities_manifest.json",
        {p.name: fingerprints(p)["sha256"] for p in directory.glob("*.npz")},
    )


def aggregate(rows):
    totals = {
        view: {unit: Counter() for unit in ("wer", "cer")}
        for view in ("strict", "spoken_form")
    }
    oov = {view: Counter() for view in totals}
    for row in rows:
        for view in totals:
            metrics = row["metrics"] if view == "strict" else row["metrics"][view]
            for unit in ("wer", "cer"):
                totals[view][unit].update(
                    {k: v for k, v in metrics[unit].items() if type(v) is int}
                )
            oov[view].update(metrics["oov_decoding_vocabulary"]["words"])
    result = {"recordings": len(rows)}
    for view in totals:
        result[view] = {
            unit: {
                **counts,
                "rate": counts["errors"] / counts["reference_units"]
                if counts["reference_units"]
                else None,
            }
            for unit, counts in totals[view].items()
        }
        result[view]["oov"] = {
            "tokens": sum(oov[view].values()),
            "words": dict(sorted(oov[view].items())),
        }
        denominator = totals[view]["wer"]["reference_units"]
        result[view]["oov"]["rate"] = (
            sum(oov[view].values()) / denominator if denominator else None
        )
    return result


def summarize(rows):
    groups = {}
    for field in ("sentence", "actor"):
        grouped = defaultdict(list)
        for row in rows:
            grouped[row[field]].append(row)
        groups[field] = {
            key: aggregate(values) for key, values in sorted(grouped.items())
        }
    return {
        "aggregate": aggregate(rows),
        "by_sentence": groups["sentence"],
        "by_actor": groups["actor"],
    }


def result_path(output, variant, split, scale):
    return output / "results" / variant / f"{split}-{scale:g}.jsonl"


def read_rows(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def run_decode(directory, settings, rows, output, variant, split, scales):
    import k2

    from asr.wfst.compact import decode

    prepared = json.loads((directory / "prepared.json").read_text())
    verify_cache(directory, prepared["key"])
    pending_scales = [
        s
        for s in scales
        if len(read_rows(result_path(output, variant, split, s))) != len(rows)
    ]
    if not pending_scales:
        return
    graph = k2.Fsa.from_dict(
        torch.load(directory / "TLG.pt", weights_only=False, map_location="cpu")
    )
    graph = k2.create_fsa_vec([graph])
    words = {
        int(i): w
        for w, i in (
            line.rsplit(" ", 1)
            for line in (directory / "words.txt").read_text().splitlines()
        )
    }
    full = set(json.loads((directory / "lexicon_full.json").read_text()))
    selected = set(words.values()) - {"<eps>", "<s>", "</s>", "#0"}
    for scale in pending_scales:
        path = result_path(output, variant, split, scale)
        path.parent.mkdir(parents=True, exist_ok=True)
        completed = read_rows(path)
        expected = [r["file_name"] for r in rows]
        if [r["file_name"] for r in completed] != expected[: len(completed)]:
            raise ResourceError("Unexpected recording order in partial benchmark")
        run_settings = {**settings, "acoustic_scale": scale}
        with path.open("a") as stream:
            for row in rows[len(completed) :]:
                with np.load(
                    output / "probabilities" / (row["file_name"] + ".npz"),
                    allow_pickle=False,
                ) as saved:
                    acoustic = AcousticOutput(
                        saved["tokens"].tolist(),
                        int(saved["blank_id"]),
                        {},
                        torch.from_numpy(saved["log_probs"].copy())[None],
                        torch.tensor([int(saved["frame_length"])]),
                    )
                    ipa, duration = str(saved["ipa_greedy"]), float(saved["duration"])
                start = time.monotonic()
                decoded = decode(
                    graph,
                    acoustic,
                    words,
                    {**run_settings, "diagnostic_paths": 32 if not completed else 0},
                )
                metrics, _ = evaluate(
                    decoded["text"],
                    {"text": row["sentence"], "normalized": normalize(row["sentence"])},
                    full,
                    selected,
                )
                record = {
                    **row,
                    "ipa_greedy": ipa,
                    "duration_seconds": duration,
                    "decode_seconds": time.monotonic() - start,
                    "decoded": decoded,
                    "metrics": metrics,
                    "graph_key": prepared["key"],
                    "settings": run_settings,
                }
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                stream.flush()
                completed.append(record)
                if len(completed) % 10 == 0 or len(completed) == len(rows):
                    print(
                        f"{variant} {split} scale={scale:g}: {len(completed)}/{len(rows)}",
                        flush=True,
                    )
        write_json(path.with_suffix(".summary.json"), summarize(completed))


def select_scale(output):
    baseline = summarize(read_rows(result_path(output, "baseline", "dev", 1.0)))[
        "aggregate"
    ]
    candidates = []
    for scale in SCALES:
        summary = summarize(read_rows(result_path(output, "combined", "dev", scale)))[
            "aggregate"
        ]
        if summary["recordings"] != baseline["recordings"] or not summary["recordings"]:
            raise ResourceError("Development comparison is incomplete")
        candidates.append((scale, summary))
    scale, summary = min(
        candidates,
        key=lambda item: (
            item[1]["spoken_form"]["wer"]["errors"],
            item[1]["spoken_form"]["cer"]["errors"],
            abs(item[0] - 1),
            item[0],
        ),
    )
    result = {
        "selected_variant": "combined",
        "acoustic_scale": scale,
        "criterion": "Development spoken WER, then CER, then distance from scale 1; no test scores",
        "baseline_dev": baseline,
        "selected_dev": summary,
    }
    path = output / "selection.json"
    if path.exists() and json.loads(path.read_text()) != result:
        raise ResourceError("Frozen development selection changed")
    write_json(path, result)
    return result


def report(output):
    selection = json.loads((output / "selection.json").read_text())
    scale = selection["acoustic_scale"]
    baseline = summarize(read_rows(result_path(output, "baseline", "test", 1.0)))[
        "aggregate"
    ]
    candidate = summarize(read_rows(result_path(output, "combined", "test", scale)))[
        "aggregate"
    ]
    if candidate["recordings"] != baseline["recordings"] or not candidate["recordings"]:
        raise ResourceError("Held-out comparison is incomplete")

    def improves(old, new):
        return (
            new["spoken_form"]["wer"]["rate"] < old["spoken_form"]["wer"]["rate"]
            and new["spoken_form"]["cer"]["rate"] <= old["spoken_form"]["cer"]["rate"]
        )

    result = {
        **selection,
        "baseline_test": baseline,
        "selected_test": candidate,
        "adoption_gate_passed": improves(
            selection["baseline_dev"], selection["selected_dev"]
        )
        and improves(baseline, candidate),
        "limitations": [
            "18 acted sentences; not a broad Italian benchmark",
            "Publisher references were not corrected from hypotheses",
            "Number verbalization is separate from strict scores; E’ annotation artifacts remain",
        ],
    }
    write_json(output / "report.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
