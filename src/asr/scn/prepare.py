"""Verify a trained Sicilian bundle and compile its independent Compact CTC graph."""

import json
import shutil
import subprocess
import sys

from asr.scn.dataset import digest
from asr.scn.training import verify_bundle
from asr.wfst.cache import verify_cache
from common.environment import package_versions
from common.errors import ResourceError
from common.resources import download_resource, extract_source, verify_file
from common.utils import fingerprints, local_path, write_json
from ipa.inventory import inventory


def required_resources(config):
    names = {"zipa_source", "icefall_source"}
    for section in (config["acoustic"], config["resources"]["scn"]):
        names.update(v for k, v in section.items() if k.endswith("_resource"))
    return sorted(names)


def model_bundle(root, config):
    path = local_path(root, config["resources"]["scn"]["model_receipt_path"])
    if not path.is_file():
        raise ResourceError(
            "Missing trained Sicilian model; run scn-poc through development"
        )
    data = json.loads(path.read_text())
    return verify_bundle(
        root,
        data["model_receipt"]
        if "model_receipt" in data
        else str(path.relative_to(root)),
    )


def identity(root, config, manifest):
    candidate = config["resources"]["scn"].get("candidate_bundle_path")
    if candidate:
        from common.artifacts import read, sha, verified_files

        directory = local_path(root, candidate)
        receipt = read(directory / "candidate.json")
        verified_files(directory, receipt)
        graph_inputs = read(directory / "graph/compile_input.json")
        settings = graph_inputs["settings"]
        acoustic = config["acoustic"]
        tokens, _ = inventory(
            local_path(root, acoustic["vocabulary_path"]),
            local_path(root, acoustic["tokenizer_path"]),
        )
        if graph_inputs["tokens"] != tokens:
            raise ResourceError(
                "Experimental graph and ZIPA label inventories disagree"
            )
        if settings["lm_scale"] != config["decoder"]["lm_scale"] or settings.get(
            "word_bonus", 0
        ) != config["decoder"].get("word_bonus", 0):
            raise ResourceError("Experimental graph and decoder costs disagree")
        payload = {
            "language": "scn",
            "candidate": sha(directory / "candidate.json"),
            "compiler": sha(root / "src/asr/wfst/graph.py"),
            "settings": settings,
            "resources": {
                name: verify_file(
                    local_path(root, manifest["resources"][name]["path"]),
                    manifest["resources"][name],
                )["sha256"]
                for name in required_resources(config)
            },
            "versions": package_versions(),
        }
        return digest(payload), payload
    directory, model = model_bundle(root, config)
    resources = {}
    for name in required_resources(config):
        spec = manifest["resources"][name]
        resources[name] = verify_file(local_path(root, spec["path"]), spec)["sha256"]
    payload = {
        "language": "scn",
        "model": model["key"],
        "artifacts": model["artifacts"],
        "resources": resources,
        "versions": package_versions(),
        "settings": {
            "lm_order": model["order"],
            "lm_scale": config["decoder"]["lm_scale"],
            "ctc_topology": config["decoder"]["ctc_topology"],
        },
        "sources": {
            p: fingerprints(root / "src" / p)["sha256"]
            for p in (
                "asr/scn/prepare.py",
                "asr/wfst/graph.py",
                "asr/wfst/compact.py",
                "ipa/inventory.py",
                "ipa/types.py",
            )
        },
    }
    return digest(payload), payload


def prepare(root, config, manifest):
    if config["languages"] != ["scn"] or not config["decoder"]["enabled"]:
        raise ResourceError("Expected enabled monolingual Sicilian configuration")
    for name in required_resources(config):
        spec = manifest["resources"][name]
        actual = download_resource(root, spec)
        if spec.get("extract_to"):
            extract_source(root, spec, actual["sha256"])
    candidate = config["resources"]["scn"].get("candidate_bundle_path")
    if candidate:
        from asr.wfst.build import compile_graph
        from common.artifacts import read, seal

        source = local_path(root, candidate) / "graph"
        key, payload = identity(root, config, manifest)
        directory = local_path(root, config["decoder"]["graph_path"]) / key
        if (directory / "prepared.json").exists():
            verify_cache(directory, key)
            return directory
        inputs = read(source / "compile_input.json")
        compile_graph(
            directory,
            read(source / "lexicon_selected.json"),
            source / "restricted.arpa",
            inputs["tokens"],
            inputs["settings"],
        )
        shutil.copyfile(
            directory / "lexicon_selected.json", directory / "lexicon_full.json"
        )
        write_json(
            directory / "vocabulary.json",
            {
                "words": sorted(read(directory / "lexicon_selected.json")),
                "model_key": key,
            },
        )
        receipt = seal(
            directory,
            "prepared.json",
            payload,
            [
                p.name
                for p in directory.iterdir()
                if p.is_file() and p.name != "prepared.json"
            ],
        )
        if receipt["key"] != key:
            raise ResourceError("Experimental preparation identity mismatch")
        return directory
    model_directory, model = model_bundle(root, config)
    key, payload = identity(root, config, manifest)
    directory = local_path(root, config["decoder"]["graph_path"]) / key
    if (directory / "prepared.json").exists():
        verify_cache(directory, key)
        return directory
    directory.mkdir(parents=True, exist_ok=True)
    write_json(directory / "preparation.json", {"status": "running", "key": key})
    try:
        acoustic = config["acoustic"]
        tokens, _ = inventory(
            local_path(root, acoustic["vocabulary_path"]),
            local_path(root, acoustic["tokenizer_path"]),
        )
        lexicon = json.loads((model_directory / "lexicon.json").read_text())
        write_json(directory / "lexicon_full.json", lexicon)
        write_json(
            directory / "vocabulary.json",
            {"words": sorted(lexicon), "model_key": model["key"]},
        )
        lexicon["<unk>"] = [[2]]
        write_json(directory / "lexicon_selected.json", lexicon)
        shutil.copyfile(model_directory / "model.arpa", directory / "restricted.arpa")
        write_json(
            directory / "compile_input.json",
            {
                "tokens": tokens,
                "settings": {**config["decoder"], "lm_order": model["order"]},
            },
        )
        process = subprocess.run(
            [sys.executable, "-m", "asr.wfst", "--directory", str(directory)]
        )
        if process.returncode:
            raise ResourceError(
                f"Graph compiler exited {process.returncode}; see {directory}/compilation.json"
            )
        artifacts = {
            p.name: fingerprints(p)["sha256"]
            for p in directory.iterdir()
            if p.is_file() and p.name not in {"prepared.json", "preparation.json"}
        }
        write_json(
            directory / "prepared.json",
            {
                "status": "complete",
                "key": key,
                "identity": payload,
                "artifacts": artifacts,
            },
        )
        write_json(directory / "preparation.json", {"status": "complete", "key": key})
    except BaseException as error:
        write_json(
            directory / "preparation.json",
            {"status": "failed", "key": key, "error": str(error)},
        )
        raise
    return directory
