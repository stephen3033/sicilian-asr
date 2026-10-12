import hashlib
import json
from pathlib import Path
from urllib.robotparser import RobotFileParser

import pytest
import yaml

from sicilian_asr.resources import ResourceError, fingerprints, load_manifest, write_json
from sicilian_asr.scn_acquisition import acquire, load_catalog, PoliteHTTP, sync_manifest
from sicilian_asr.scn_corpus import tokens, fluent_records, localization_records, conllu_records, extract, analyze, near_duplicates
from sicilian_asr.scn_lm import ProbeLM, diagnose

ROOT = Path(__file__).resolve().parents[1]


def fixture_catalog(root, *, content="Cchiù ḍḍà.\n", extractor="text_lines", orthography="other_sicilian"):
    path = root / ".cache/scn/acquired/source.txt"
    path.parent.mkdir(parents=True)
    path.write_text(content)
    source = dict(name="fixture", kind="text_corpus", language="scn", status="available",
                  source_url="https://example.org/source", url="https://example.org/source", revision="fixture-v1",
                  version="v1", path=str(path.relative_to(root)), format="TXT", license="CC0-1.0",
                  license_url="https://example.org/license", citation="fixture", checksum_note="test fixture",
                  provenance="http_snapshot", acquired_at="2026-10-08T00:00:00Z", final_url="https://example.org/source",
                  orthography=orthography, orthography_evidence="Explicit fixture classification", data_role="text",
                  acquisition_status="acquired", permissions={"acquisition":"allowed", "local_analysis":"allowed"},
                  extractor=extractor, **fingerprints(path))
    catalog = {"schema_version":1,"sources":{"scn_fixture":source}}
    write_json(root / "resources/scn/sources/catalog.json", catalog)
    return path, catalog


def test_unicode_tokenization_preserves_lexical_distinctions():
    assert tokens("CCHIU\u0300 ḍḍà dâ dû ‘n l’amuri âm’ 2024") == ["cchiù", "ḍḍà", "dâ", "dû", "'n", "l'amuri", "âm'", "2024"]
    assert tokens("--- '''") == []


def test_fluent_ast_handles_variants_and_placeable_boundaries():
    source = """# comment
-brand = Marca
message = Salutamu { $name }!
choice = { $count ->
    [one] Un libbru
   *[other] Tanti libbra
    }
    .accesskey = L
"""
    result = list(fluent_records(source))
    assert [r[1] for r in result] == ["Salutamu", "Un libbru", "Tanti libbra"]
    assert all(r[3] == "localization_fragment" for r in result)
    assert "$name" in result[0][2]
    with pytest.raises(ResourceError, match="Unparsed Fluent"):
        list(fluent_records("bad = {"))


def test_xliff_excludes_source_fallback_and_does_not_bridge_inline_codes():
    source = '<xliff><trans-unit id="1"><source>Hello</source><target>Hello</target></trans-unit><trans-unit id="2"><source>Other</source><target>Salutamu <x id="1"/> tutti</target></trans-unit></xliff>'
    assert [r[1] for r in localization_records("x.xliff", source)] == ["Salutamu", "tutti"]


def test_po_excludes_fuzzy_and_source_text():
    source = '#, fuzzy\nmsgid "Hello"\nmsgstr "Salutamu"\n\nmsgid "Day"\nmsgstr "Jornu"\n'
    assert [r[1] for r in localization_records("x.po",source)] == ["Jornu"]


def test_conllu_counts_surface_sentence_once():
    source = '# newdoc id = book\n# sent_id = 1\n# text = L’amuri.\n1-2\tL’amuri\t_\n1\tL’\t_\n2\tamuri\t_\n'
    rows = list(conllu_records(source))
    assert len(rows) == 1
    assert rows[0][:3] == ("book", "1", "L’amuri.")


def test_google_targets_keep_document_group_and_original_spelling(tmp_path):
    _, catalog = fixture_catalog(tmp_path, content=json.dumps({"id":"doc1","tl":"scn","trgs":["Cchiù ḍḍà.","N'autra frasi."]}), extractor="smol_jsonl")
    extract(tmp_path)
    rows = [json.loads(line) for line in (tmp_path / ".cache/scn/corpus/records.jsonl").read_text().splitlines()]
    assert len(rows) == 2 and len({r['id'] for r in rows}) == 2
    assert len({r['document_id'] for r in rows}) == 1
    assert {r['orthography'] for r in rows} == {"other_sicilian"}
    assert rows[0]['text'] == rows[0]['original_text'] == 'Cchiù ḍḍà.'


def test_checksum_tampering_rejected_offline_and_before_extraction(tmp_path):
    path, _ = fixture_catalog(tmp_path)
    path.write_text("changed")
    assert acquire(tmp_path, offline=True) == ["scn_fixture"]
    with pytest.raises(ResourceError):
        extract(tmp_path)


def test_unlicensed_source_cannot_download(tmp_path, monkeypatch):
    _, catalog = fixture_catalog(tmp_path)
    catalog['sources']['scn_fixture']['permissions']['acquisition'] = 'unclear'
    write_json(tmp_path / "resources/scn/sources/catalog.json", catalog)
    monkeypatch.setattr('sicilian_asr.scn_acquisition.download_resource', lambda *a,**k: pytest.fail("network must not be reached"))
    assert acquire(tmp_path, ['scn_fixture'], record=True) == ['scn_fixture']


def test_catalog_cannot_escape_ignored_directory(tmp_path):
    _, catalog = fixture_catalog(tmp_path)
    catalog['sources']['scn_fixture']['path'] = 'resources/ita/lexicon/overwrite.tsv'
    write_json(tmp_path / "resources/scn/sources/catalog.json", catalog)
    with pytest.raises(ResourceError, match="SCN originals"):
        load_catalog(tmp_path)


def test_robots_agent_specific_disallow_is_respected():
    http = PoliteHTTP(delay=0)
    robot = RobotFileParser()
    robot.parse(['User-agent: GPTBot', 'Disallow: /', '', 'User-agent: *', 'Allow: /'])
    http.robots['https://example.org'] = robot
    with pytest.raises(ResourceError, match="Robots policy"):
        http.check('https://example.org/data')
    with pytest.raises(ResourceError, match="HTTPS"):
        http.check('http://example.org/data')


def test_manifest_sync_preserves_unrelated_bytes_and_is_idempotent(tmp_path):
    fixture_catalog(tmp_path)
    path = tmp_path / 'resources/manifest.yaml'
    original = (ROOT / 'resources/manifest.yaml').read_text()
    path.write_text(original)
    sync_manifest(tmp_path)
    first = path.read_text()
    assert first.startswith(original.rstrip() + '\n')
    sync_manifest(tmp_path)
    assert path.read_text() == first
    manifest = load_manifest(path)
    del manifest['resources']['scn_fixture']['acquired_at']
    path.write_text(yaml.safe_dump(manifest))
    with pytest.raises(ResourceError, match="acquisition receipt"):
        load_manifest(path)


def test_duplicate_documents_share_split_and_are_not_promoted_to_dos24(tmp_path):
    fixture_catalog(tmp_path, content='Na frasi longa cu tanti paroli.\nNa frasi longa cu tanti paroli.\n')
    extract(tmp_path)
    analyze(tmp_path)
    rows = [json.loads(line) for line in (tmp_path/'.cache/scn/corpus/analyzed.jsonl').read_text().splitlines()]
    assert rows[1]['duplicate_of'] == rows[0]['id']
    assert rows[0]['split_group'] == rows[1]['split_group']
    assert rows[0]['split'] == rows[1]['split']
    statistics = json.loads((tmp_path/'resources/scn/corpus/statistics.json').read_text())
    assert statistics['categories']['dos24']['tokens'] == 0


def test_witten_bell_normalizes_and_unseen_histories_back_off():
    lm = ProbeLM([{'tokens':['a','b','a','b']}, {'tokens':['b','a']}])
    for history in [('a','b','a'), ('unseen','history','x')]:
        for n in range(4):
            assert sum(lm.probabilities(history,w)[n] for w in lm.vocabulary) == pytest.approx(1)
    scores = lm.evaluate([{'tokens':['never-seen']}])
    assert scores['oov_rate'] == 1
    assert scores['orders']['4']['perplexity'] > 0
    assert ('b','b') not in lm.counts[1]  # No n-grams across sentence boundaries.


def test_probe_refuses_empty_independent_splits():
    assert diagnose([])['selected_order'] is None


def test_no_italian_manifest_or_config_activation():
    catalog = load_catalog(ROOT)
    manifest = load_manifest(ROOT/'resources/manifest.yaml')['resources']
    for name,source in catalog['sources'].items():
        assert manifest[name]['acquisition_handler'] == 'scn_reviewed'
        assert manifest[name]['revision'] == source['revision']
    for name in ('scn','ita_scn'):
        assert not yaml.safe_load((ROOT/f'configs/{name}.yaml').read_text())['decoder']['enabled']
    assert manifest['scn_lm']['status'] == manifest['scn_lexicon']['status'] == 'planned'
