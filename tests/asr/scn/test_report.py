from asr.scn.report import table, write_section


def test_generated_section_preserves_explanation_and_is_repeatable(tmp_path):
    path = tmp_path / "audit.md"
    path.write_text(
        "Explanation\n<!-- BEGIN GENERATED -->\nold\n<!-- END GENERATED -->\nEvidence\n"
    )
    rows = table(["Source", "Finding"], [["example", "line one\npipe | two"]])
    write_section(path, rows)
    first = path.read_text()
    write_section(path, rows)
    assert path.read_text() == first
    assert first.startswith("Explanation\n") and first.endswith("\nEvidence\n")
    assert "line one pipe \\| two" in first
