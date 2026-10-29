import json
import hashlib

import pytest

from asr.scn import dos24_corpus
from common.artifacts import digest, sha
from common.errors import ResourceError
from common.utils import write_json


def test_corpus_immutable_sources_postconversion_leakage_and_sentence_boundaries(
    tmp_path, monkeypatch
):
    root, bundle, output = tmp_path, tmp_path / "bundle", tmp_path / "corpus"
    source = root / "source.bz2"
    source.write_bytes(b"pinned source bytes")
    write_json(
        root / "resources/scn/sources/catalog.json",
        {
            "sources": {
                "wiki": {
                    "path": "source.bz2",
                    "sha256": sha(source),
                    "license": "CC-BY-SA-4.0",
                }
            }
        },
    )
    write_json(
        root / "resources/scn/poc_reviews.json",
        {
            "replacements": {
                "e": {
                    "replacement": "è",
                    "reason": "context only",
                    "evidence": "old decision",
                }
            }
        },
    )
    write_json(bundle / "recordings.json", [{"reference": "cavaḍḍu cani novu friddu"}])
    write_json(
        bundle / "channel.json",
        {
            "recordings": [{"split": "train", "text": "cani friddu"}],
            "training": [{"id": "train:0", "text": "cani friddu"}],
        },
    )
    converter_source = root / "src/asr/scn/dos24.py"
    converter_source.parent.mkdir(parents=True)
    converter_source.write_text("versioned converter")

    def index(_root, _bundle, directory):
        write_json(directory / "index.json", {"analyses": {}})
        return {}

    monkeypatch.setattr(dos24_corpus, "build_index", index)
    texts = [
        "cavaddu cani novu friddu. Unknownword. cani novu friddu gatta.",
        "cani novu friddu gatta.",
    ]
    records = []
    for i, text in enumerate(texts):
        records.append(
            {
                "id": str(i),
                "source_id": "wiki",
                "source_sha256": sha(source),
                "text": text,
                "original_text": text,
                "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
                "document_id": f"article:{i}",
                "unit_kind": "paragraph",
                "metadata": {
                    "page_id": str(i),
                    "revision_id": "123",
                    "url": "https://example.test",
                },
            }
        )
    records_path = root / "records.jsonl"
    records_path.write_text("".join(json.dumps(r) + "\n" for r in records))
    settings = {"source_id": "wiki", "records": "records.jsonl", "minimum_words": 4}
    summary = dos24_corpus.prepare_corpus(root, bundle, output, settings)
    assert summary["exclusions"]["development_passage_overlap"] == 1
    assert summary["exclusions"]["exact_duplicate"] == 1
    assert summary["counts"]["admitted_sentences"] == 1
    admitted = [
        json.loads(line)
        for line in (output / "admitted.jsonl").read_text().splitlines()
    ]
    assert admitted[0]["converted"] == "cani novu friddu gatta"
    assert dos24_corpus.prepare_corpus(root, bundle, output, settings) == summary
    assert records_path.read_text() == "".join(json.dumps(r) + "\n" for r in records)
    changed = {**records[0], "text": "changed source without updated checksum"}
    records_path.write_text(json.dumps(changed) + "\n")
    with pytest.raises(ResourceError, match="checksum"):
        dos24_corpus.prepare_corpus(root, bundle, root / "changed-output", settings)
