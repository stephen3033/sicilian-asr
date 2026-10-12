"""Read-only validation of resource files and initialization configurations."""

import csv
from pathlib import Path

from asr.ita.lexicon import lexicon_statistics
from common.errors import ResourceError
from common.resources import load_resources, verify_file
from common.utils import local_path
from pipeline.config import load_config


def validate_resources(root: Path, config_name: str = "ita") -> dict:
    manifest = load_resources(root)
    configs = {
        name: load_config(name, root, manifest) for name in ("ita", "scn", "ita_scn")
    }
    report = {
        "config": config_name,
        "resources": {},
        "errors": [],
        "required_missing": [],
    }
    specs = manifest["resources"]
    for name, spec in specs.items():
        if spec["status"] != "available":
            report["resources"][name] = {
                "status": spec["status"],
                "reason": spec["reason"],
            }
            continue
        path = local_path(root, spec["path"])
        if not path.exists():
            report["resources"][name] = {
                "status": "missing",
                "path": spec["path"],
                "instruction": f"python scripts/download_resources.py {name}",
            }
            continue
        try:
            actual = verify_file(path, spec)
            result = {"status": "ok", "path": spec["path"], **actual}
            if name == "phonitalia_ipa":
                result["statistics"] = lexicon_statistics(path)
                for key in ("rows", "vocabulary"):
                    if result["statistics"][key] != spec[f"expected_{key}"]:
                        raise ResourceError(
                            f"Unexpected {key}: {result['statistics'][key]}"
                        )
            if name == "phonitalia_wordforms":
                with path.open(encoding="utf-8", newline="") as stream:
                    reader = csv.DictReader(stream, delimiter="\t")
                    if not {"word", "Phones", "PhoneSyll"}.issubset(
                        reader.fieldnames or []
                    ):
                        raise ResourceError(
                            "Original PhonItalia table lacks required columns"
                        )
                    words = set()
                    count = 0
                    for row in reader:
                        count += 1
                        words.add(row["word"])
                result["statistics"] = {"rows": count, "vocabulary": len(words)}
                if count != 120000:
                    raise ResourceError(
                        f"Unexpected original PhonItalia row count: {count}"
                    )
            report["resources"][name] = result
        except (ResourceError, OSError, UnicodeError) as error:
            report["resources"][name] = {"status": "error", "error": str(error)}
            report["errors"].append(f"{name}: {error}")
    if all(
        report["resources"][name]["status"] == "ok"
        for name in ("zipa_tokens", "zipa_tokenizer")
    ):
        try:
            import sentencepiece

            tokens = {}
            for line in (
                local_path(root, specs["zipa_tokens"]["path"]).read_text().splitlines()
            ):
                token, identifier = line.rsplit(" ", 1)
                if int(identifier) in tokens:
                    raise ResourceError("Duplicate acoustic token ID")
                tokens[int(identifier)] = token
            tokenizer = sentencepiece.SentencePieceProcessor(
                model_file=str(local_path(root, specs["zipa_tokenizer"]["path"]))
            )
            if (
                set(tokens) != set(range(127))
                or tokens[0] != "<blk>"
                or tokenizer.get_piece_size() != 127
            ):
                raise ResourceError(
                    "Expected a contiguous 127-label inventory and blank ID 0"
                )
            if any(tokenizer.id_to_piece(i) != token for i, token in tokens.items()):
                raise ResourceError("Checkpoint tokens and SentencePiece IDs differ")
            report["acoustic_inventory"] = {
                "status": "ok",
                "labels": 127,
                "blank_id": 0,
            }
        except Exception as error:
            report["acoustic_inventory"] = {"status": "error", "error": str(error)}
            report["errors"].append(f"Acoustic inventory: {error}")
    selected = configs[config_name]
    required = {"zipa_source", "icefall_source"}
    for section in [selected["acoustic"], *selected["resources"].values()]:
        required.update(
            value for key, value in section.items() if key.endswith("_resource")
        )
    report["required_missing"] = sorted(
        name for name in required if report["resources"][name]["status"] != "ok"
    )
    report["resource_ready"] = not report["errors"] and not report["required_missing"]
    report["pipeline_implemented"] = (
        selected["languages"] in (["ita"], ["scn"]) and selected["decoder"]["enabled"]
    )
    if selected["languages"] == ["scn"]:
        from asr.scn.prepare import model_bundle

        try:
            _, model = model_bundle(root, selected)
            report["trained_model"] = {"status": "ok", "key": model["key"]}
        except (ResourceError, OSError, ValueError) as error:
            report["trained_model"] = {
                "status": "missing_or_invalid",
                "error": str(error),
            }
            report["resource_ready"] = False
    report["graph_ready"] = False
    if report["pipeline_implemented"] and report["resource_ready"]:
        from asr.wfst.cache import verify_cache
        from pipeline.languages import backend

        try:
            language, _ = backend(selected)
            key, _ = language.identity(root, selected, manifest)
            directory = local_path(root, selected["decoder"]["graph_path"]) / key
            verify_cache(directory, key)
            report["graph_ready"] = True
            report["graph_directory"] = str(directory)
        except (ResourceError, OSError) as error:
            report["graph_error"] = str(error)
    return report
