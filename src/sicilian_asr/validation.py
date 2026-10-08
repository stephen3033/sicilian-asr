"""Read-only validation of resource files and initialization configurations."""

import csv
from pathlib import Path

from sicilian_asr.config import load_config
from sicilian_asr.resources import ResourceError, load_manifest, local_path, verify_file


def lexicon_statistics(path: Path) -> dict:
    words, pairs = set(), set()
    count = 0
    with path.open(encoding="utf-8", newline="") as stream:
        for row in csv.reader(stream, delimiter="\t"):
            if len(row) != 2 or not row[0] or not row[1].split():
                raise ResourceError(f"Malformed IPA dictionary row {count + 1}")
            count += 1
            words.add(row[0])
            pairs.add(tuple(row))
    return {"rows": count, "vocabulary": len(words), "distinct_word_pronunciation_pairs": len(pairs)}


def validate_resources(root: Path, config_name: str = "ita") -> dict:
    manifest = load_manifest(root / "resources/manifest.yaml")
    configs = {name: load_config(root / f"configs/{name}.yaml", root, manifest) for name in ("ita", "scn", "ita_scn")}
    report = {"config": config_name, "resources": {}, "errors": [], "required_missing": []}
    specs = manifest["resources"]
    for name, spec in specs.items():
        if spec["status"] != "available":
            report["resources"][name] = {"status": spec["status"], "reason": spec["reason"]}
            continue
        path = local_path(root, spec["path"])
        if not path.exists():
            report["resources"][name] = {"status": "missing", "path": spec["path"], "instruction": f"python scripts/download_resources.py {name}"}
            continue
        try:
            actual = verify_file(path, spec)
            result = {"status": "ok", "path": spec["path"], **actual}
            if name == "phonitalia_ipa":
                result["statistics"] = lexicon_statistics(path)
                for key in ("rows", "vocabulary"):
                    if result["statistics"][key] != spec[f"expected_{key}"]:
                        raise ResourceError(f"Unexpected {key}: {result['statistics'][key]}")
            if name == "phonitalia_wordforms":
                with path.open(encoding="utf-8", newline="") as stream:
                    reader = csv.DictReader(stream, delimiter="\t")
                    if not {"word", "Phones", "PhoneSyll"}.issubset(reader.fieldnames or []):
                        raise ResourceError("Original PhonItalia table lacks required columns")
                    words = set()
                    count = 0
                    for row in reader:
                        count += 1
                        words.add(row["word"])
                result["statistics"] = {"rows": count, "vocabulary": len(words)}
                if count != 120000:
                    raise ResourceError(f"Unexpected original PhonItalia row count: {count}")
            report["resources"][name] = result
        except (ResourceError, OSError, UnicodeError) as error:
            report["resources"][name] = {"status": "error", "error": str(error)}
            report["errors"].append(f"{name}: {error}")
    if all(report["resources"][name]["status"] == "ok" for name in ("zipa_tokens", "zipa_tokenizer")):
        try:
            import sentencepiece

            tokens = {}
            for line in local_path(root, specs["zipa_tokens"]["path"]).read_text().splitlines():
                token, identifier = line.rsplit(" ", 1)
                if int(identifier) in tokens:
                    raise ResourceError("Duplicate acoustic token ID")
                tokens[int(identifier)] = token
            tokenizer = sentencepiece.SentencePieceProcessor(model_file=str(local_path(root, specs["zipa_tokenizer"]["path"])))
            if set(tokens) != set(range(127)) or tokens[0] != "<blk>" or tokenizer.get_piece_size() != 127:
                raise ResourceError("Expected a contiguous 127-label inventory and blank ID 0")
            if any(tokenizer.id_to_piece(i) != token for i, token in tokens.items()):
                raise ResourceError("Checkpoint tokens and SentencePiece IDs differ")
            report["acoustic_inventory"] = {"status": "ok", "labels": 127, "blank_id": 0}
        except Exception as error:
            report["acoustic_inventory"] = {"status": "error", "error": str(error)}
            report["errors"].append(f"Acoustic inventory: {error}")
    selected = configs[config_name]
    required = {"zipa_source", "icefall_source"}
    for section in [selected["acoustic"], *selected["resources"].values()]:
        required.update(value for key, value in section.items() if key.endswith("_resource"))
    report["required_missing"] = sorted(name for name in required if report["resources"][name]["status"] != "ok")
    report["resource_ready"] = not report["errors"] and not report["required_missing"]
    report["pipeline_implemented"] = False
    return report
