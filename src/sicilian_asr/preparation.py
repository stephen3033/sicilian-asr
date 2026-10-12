"""Resource preparation and content-addressed graph cache verification."""

from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import subprocess
import sys

from sicilian_asr.resources import ResourceError, download_resource, extract_members, extract_source, fingerprints, local_path, verify_file, write_json
from sicilian_asr.lexicon.prepare import ADAPTATION_RULES, LEXICON_RULES, adapt, convert, inventory
from sicilian_asr.lm.prepare import VOCAB_RULES, read_unigrams, restrict_arpa, select_vocabulary
from sicilian_asr.text import TEXT_RULES


def package_versions() -> dict:
    return {n: version(n) for n in ("sicilian-asr", "torch", "torchaudio", "k2", "kaldilm", "kenlm", "numpy", "lhotse", "sentencepiece", "soundfile", "kaldialign", "pyyaml")}


def required_resources(config: dict) -> list[str]:
    names = {"zipa_source", "icefall_source"}
    for section in [config["acoustic"], config["resources"]["ita"]]:
        names.update(v for k, v in section.items() if k.endswith("_resource"))
    return sorted(names)


def identity(root: Path, config: dict, manifest: dict) -> tuple[str, dict]:
    from sicilian_asr.decoding.compact import TOPOLOGY_RULES
    from sicilian_asr.decoding.graph import GRAPH_RULES

    resources = {}
    for name in required_resources(config):
        spec = manifest["resources"][name]
        actual = verify_file(local_path(root, spec["path"]), spec)
        resources[name] = {"sha256": actual["sha256"], "revision": spec["revision"]}
    lm_spec = manifest["resources"][config["resources"]["ita"]["lm_resource"]]
    receipt_path = local_path(root, lm_spec["path"] + ".members.json")
    if not receipt_path.is_file():
        raise ResourceError("Missing derived LM receipt; run prepare")
    receipt = json.loads(receipt_path.read_text())
    if receipt["archive_sha256"] != resources[config["resources"]["ita"]["lm_resource"]]["sha256"]:
        raise ResourceError("Stale derived LM archive receipt")
    arpa_path = config["resources"]["ita"]["arpa_path"]
    if arpa_path not in receipt["artifacts"]:
        raise ResourceError("ARPA is not an acquired member of the pinned MLS archive")
    actual = verify_file(local_path(root, arpa_path), {**receipt["artifacts"][arpa_path], "revision": lm_spec["revision"]})
    source = Path(__file__).parent
    payload = {"schema": 1, "resources": resources, "arpa_sha256": actual["sha256"], "versions": package_versions(),
               "normalization": [TEXT_RULES, LEXICON_RULES, ADAPTATION_RULES, VOCAB_RULES], "graph_rules": [TOPOLOGY_RULES, GRAPH_RULES],
               "settings": {**{k: config["decoder"][k] for k in ("vocabulary_size", "ctc_topology", "lm_scale")},
                            **{k: config["decoder"].get(k, False) for k in ("pronunciation_variants", "contractions")}},
               "compiler_sources": {p: fingerprints(source/p)["sha256"] for p in ("lexicon/prepare.py", "lm/prepare.py", "decoding/compact.py", "decoding/graph.py", "text.py")}}
    key = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    return key, payload


def verify_cache(directory: Path, key: str) -> dict:
    path = directory/"prepared.json"
    if not path.is_file():
        raise ResourceError(f"Missing prepared graph: {directory}; run prepare")
    prepared = json.loads(path.read_text())
    if prepared.get("status") != "complete" or prepared.get("key") != key:
        raise ResourceError("Stale or incomplete graph cache; run prepare")
    for name, expected in prepared["artifacts"].items():
        artifact = local_path(directory, name)
        if not artifact.is_file() or fingerprints(artifact)["sha256"] != expected:
            raise ResourceError(f"Prepared artifact checksum mismatch: {artifact}")
    return prepared


def prepare(root: Path, config: dict, manifest: dict) -> Path:
    if config["languages"] != ["ita"] or not config["decoder"]["enabled"]:
        raise ResourceError("Only enabled Italian decoding can be prepared")
    receipts_path = root/".cache/resource-status.json"
    receipts = json.loads(receipts_path.read_text()) if receipts_path.exists() else {}
    for name in required_resources(config):
        print(f"Verifying/acquiring {name}", flush=True)
        spec = manifest["resources"][name]
        actual = download_resource(root, spec, receipts.get(name))
        if spec.get("extract_to"):
            extract_source(root, spec, actual["sha256"])
        if spec.get("members"):
            extract_members(root, spec, actual)
        receipts[name] = {**actual, "revision": spec["revision"], "source_url": spec["url"], "path": spec["path"], "verified_at": datetime.now(timezone.utc).isoformat()}
        write_json(receipts_path, receipts)
    key, payload = identity(root, config, manifest)
    directory = local_path(root, config["decoder"]["graph_path"])/key
    if (directory/"prepared.json").exists():
        verify_cache(directory, key)
        print(f"Verified prepared graph {directory}", flush=True)
        return directory
    directory.mkdir(parents=True, exist_ok=True)
    write_json(directory/"preparation.json", {"status": "preparing", "key": key, "identity": payload})
    try:
        acoustic = config["acoustic"]
        tokens, tokenizer = inventory(local_path(root, acoustic["vocabulary_path"]), local_path(root, acoustic["tokenizer_path"]))
        lexicon, audit = convert(local_path(root, config["resources"]["ita"]["lexicon_path"]), tokens, tokenizer)
        write_json(directory/"lexicon_audit.json", audit)
        arpa = local_path(root, config["resources"]["ita"]["arpa_path"])
        lexicon, adaptation = adapt(lexicon, tokens, set(read_unigrams(arpa)),
            **{k: config["decoder"].get(k, False) for k in ("pronunciation_variants", "contractions")})
        write_json(directory/"lexicon_adaptation.json", adaptation)
        write_json(directory/"lexicon_full.json", lexicon)
        vocabulary, info = select_vocabulary(arpa, lexicon, config["decoder"]["vocabulary_size"])
        write_json(directory/"vocabulary.json", {**info, "words": vocabulary})
        selected = {w: lexicon[w] for w in vocabulary}
        selected["<unk>"] = [[tokens.index("<unk>")]]
        write_json(directory/"lexicon_selected.json", selected)
        write_json(directory/"compile_input.json", {"tokens": tokens, "settings": config["decoder"]})
        print(f"Restricting original ARPA to {len(vocabulary):,} words", flush=True)
        stats = restrict_arpa(arpa, directory/"restricted.arpa", vocabulary)
        write_json(directory/"lm_statistics.json", stats)
        # Native compilation is isolated so aborts leave a failed preparation receipt.
        process = subprocess.run([sys.executable, "-m", "sicilian_asr", "_compile", "--directory", str(directory)])
        if process.returncode:
            raise ResourceError(f"Native graph compilation exited {process.returncode}; see {directory}/compilation.json")
        artifacts = {p.name: fingerprints(p)["sha256"] for p in directory.iterdir() if p.is_file() and p.name not in {"prepared.json", "preparation.json"}}
        write_json(directory/"prepared.json", {"status": "complete", "key": key, "identity": payload, "artifacts": artifacts})
        write_json(directory/"preparation.json", {"status": "complete", "key": key})
    except BaseException as error:
        write_json(directory/"preparation.json", {"status": "failed", "key": key, "error": f"{type(error).__name__}: {error}"})
        raise
    return directory
