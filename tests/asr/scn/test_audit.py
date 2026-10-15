import bz2
import copy
import json

import pytest

from asr.scn.acquisition import eligible, load_catalog
from asr.scn.audit import (
    android_unescape,
    apply_review,
    digest,
    liita_forms,
    liita_glossary,
    liita_sentences,
    screen,
    summary,
    tokens,
    validate_catalog,
    wiki_body,
    wikipedia_records,
)
from common.errors import ResourceError
from tests.support import ROOT

RULES = json.loads((ROOT / "resources/scn/audit/rules.json").read_text())


def row(text="Na frasi.", kind="sentence", category="unresolved"):
    return dict(
        id="r1",
        source_id="scn_fixture",
        document_id="doc",
        text=text,
        text_sha256=digest(text),
        source_sha256="a" * 64,
        tokens=tokens(text),
        unit_kind=kind,
        category=category,
        reviewed=False,
        flags=screen(text, RULES),
        language_screen={"screen": "unresolved"},
        metadata={},
    )


def decision(r):
    return dict(
        id=r["id"],
        source_sha256=r["source_sha256"],
        text_sha256=r["text_sha256"],
        category="verified_conforming",
        rules_version=RULES["version"],
        reviewer="test",
        checked_on="2026-10-11",
        explanation="Context reviewed",
        evidence_urls=["https://example.org/source"],
        context="Full sentence",
        checked_rules=["D07", "D08", "D09", "D10", "D11"],
        flag_dispositions={},
    )


def test_android_resource_escapes_do_not_become_lexical_n_tokens():
    assert android_unescape(r"\n\nMannatu di") == "\n\nMannatu di"
    assert android_unescape(r"L\'amuri \u1e0d\u1e0dà") == "L'amuri ḍḍà"
    assert android_unescape(r"\\n") == r"\n"


def test_contextual_exceptions_and_unicode_are_not_errors():
    text = "Aju àju ai ài avi àvi annu ànnu friddu addisijari ‘n casa ḍḍà cchiu\u0300."
    assert screen(text, RULES) == []
    assert tokens(text)[-1] == "cchiù"
    assert {x["rule"] for x in screen("cavaddu cchiu é ḍa", RULES)} == {
        "D01",
        "D03",
        "D04",
        "D06",
    }
    # A foreign name is a candidate that a contextual reviewer can accept.
    assert screen("William", RULES)[0]["rule"] == "D02"


def test_current_wiki_namespace_revision_and_template_boundaries(tmp_path):
    xml = """<mediawiki xmlns="http://www.mediawiki.org/xml/export-0.11/">
      <page><title>Example</title><ns>0</ns><id>12</id><revision><id>45</id><timestamp>2026-10-01</timestamp><text>Na frasi.{{template|hidden}}N'autra frasi. &lt;ref&gt;Italian reference&lt;/ref&gt; [[Catigurìa:Test]]</text></revision></page>
      <page><title>Redirect</title><ns>0</ns><id>13</id><redirect title="Example"/><revision><id>46</id><text>#REDIRECT [[Example]]</text></revision></page>
      <page><title>Template:X</title><ns>10</ns><id>14</id><revision><id>47</id><text>Not body</text></revision></page>
    </mediawiki>"""
    path = tmp_path / "dump.bz2"
    path.write_bytes(bz2.compress(xml.encode()))
    pages = []
    rows = list(wikipedia_records(path, pages))
    assert [r[2] for r in rows] == ["Na frasi.", "N'autra frasi."]
    assert rows[0][4]["revision_id"] == "45"
    assert len(pages) == 3
    assert pages[1]["exclusion"] == "redirect"
    assert pages[2]["exclusion"] == "non_article_namespace"


def test_wiki_nested_markup_keeps_visible_link_text():
    parts, removed = wiki_body(
        "== Titulu ==\n[[Cani|Lu cani]] è cca. <ref><ref>citation</ref></ref>\n\n{| class=wikitable\n| tabella\n|}\n[[File:Test.jpg|caption]]"
    )
    assert parts == ["Lu cani è cca."]
    assert removed["headings"] == 1


def test_liita_extraction_excludes_italian_and_annotation_labels(tmp_path):
    ttl = """@prefix ontolex: <http://www.w3.org/ns/lemon/ontolex#> .
      @prefix lila: <http://lila-erc.eu/ontologies/lila/> .
      @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
      <https://example.org/lemma/1> a lila:Lemma; rdfs:label "cani";
          ontolex:writtenRep "cani"@scn, "cane"@it, "càni"@scn .
      <https://example.org/annotation> rdfs:label "Sentence 1" ."""
    path = tmp_path / "bank.ttl"
    path.write_text(ttl)
    forms = list(liita_forms(path))
    assert [r[1] for r in forms] == ["cani", "càni"]
    assert [r[2]["form_kind"] for r in forms] == ["canonical", "variant"]
    glossary = tmp_path / "glossary.ttl"
    glossary.write_text("""@prefix ontolex: <http://www.w3.org/ns/lemon/ontolex#> .
      <https://example.org/LexicalEntry/siciliano/1> ontolex:canonicalForm <https://example.org/lemma/1> .
      <https://example.org/LexicalEntry/italian/1> ontolex:canonicalForm <https://example.org/lemma/1> .""")
    bank = {"https://example.org/lemma/1": [(r[1], r[2]) for r in forms]}
    assert len(list(liita_glossary(glossary, bank))) == 2


def test_treebank_reads_terminal_chain_not_lexicographic_labels(tmp_path):
    path = tmp_path / "treebank.ttl"
    path.write_text("""@prefix c: <http://lila-erc.eu/ontologies/lila_corpora/> .
      @prefix p: <http://purl.org/powla/powla.owl#> .
      @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
      <urn:s> c:hasRefType "Sentence"; c:first <urn:1>; p:hasChild <urn:1>, <urn:2> .
      <urn:1> p:hasStringValue "Lu"; p:next <urn:2>; rdfs:label "Italian translation" .
      <urn:2> p:hasStringValue "cani" .""")
    assert list(liita_sentences(path))[0][1] == "Lu cani"
    path.write_text(path.read_text() + "\n<urn:2> p:next <urn:1> .")
    with pytest.raises(ResourceError, match="terminal chain"):
        list(liita_sentences(path))


def test_verification_requires_hashes_contextual_rules_and_flag_resolution():
    r = row("William è cca.")
    d = decision(r)
    with pytest.raises(ResourceError, match="unresolved flags"):
        apply_review(r, d, RULES)
    d["flag_dispositions"] = {
        "D02": "Proper name retains foreign spelling per DOS24 p7"
    }
    apply_review(r, d, RULES)
    assert r["reviewed"] and r["category"] == "verified_conforming"
    d["text_sha256"] = digest("Changed")
    with pytest.raises(ResourceError, match="Stale review"):
        apply_review(r, d, RULES)
    d = decision(r)
    d["checked_rules"] = []
    with pytest.raises(ResourceError, match="contextual checklist"):
        apply_review(r, d, RULES)


def test_dedup_is_scoped_and_lexical_forms_are_not_sentences():
    a, b, lex = row(), row(category="verified_conforming"), row(kind="lexical_entry")
    assert summary([a, b])["deduplicated_tokens"] == 2
    assert (
        summary([b])["deduplicated_tokens"] == 2
    )  # unresolved copy cannot erase verified material
    s = summary([a, b, lex])
    assert s["tokens"] == 6 and s["deduplicated_tokens"] == 4
    assert s["lexical_forms"] == 1 and s["sentence_units"] == 2
    assert (
        s["tokens"]
        == s["lexical_tokens"] + s["localization_tokens"] + s["prose_tokens"]
    )


def test_unknown_or_unsatisfied_collection_cannot_download():
    s = {
        "permissions": {"acquisition": "allowed"},
        "url": "https://example.org/export",
        "audit": {"bulk_collection": {"value": "unresolved"}},
    }
    assert not eligible(s)
    s["audit"]["bulk_collection"]["value"] = "conditional"
    assert not eligible(s)
    s["audit"]["bulk_collection"]["conditions_satisfied"] = True
    assert eligible(s)


def test_every_original_entry_has_independent_evidence():
    catalog = load_catalog(ROOT)
    validate_catalog(catalog)
    assert len(catalog["original_source_ids"]) == 83
    broken = copy.deepcopy(catalog)
    broken["sources"]["scn_artesia"]["audit"]["bulk_collection"]["value"] = (
        "not_authorized"
    )
    with pytest.raises(ResourceError, match="collection permission"):
        validate_catalog(broken)
    broken = copy.deepcopy(catalog)
    broken["sources"]["scn_artesia"]["audit"]["dates"]["evidence_urls"] = []
    with pytest.raises(ResourceError, match="Missing evidence"):
        validate_catalog(broken)
