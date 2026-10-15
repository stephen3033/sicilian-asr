from importlib import import_module


def test_package_imports():
    for name in (
        "common",
        "media_input",
        "audio",
        "ipa",
        "asr.ita",
        "asr.scn",
        "asr.wfst",
        "evaluation",
        "pipeline",
    ):
        assert import_module(name) is not None


def test_source_contains_only_python():
    from tests.support import ROOT

    unexpected = [
        str(p.relative_to(ROOT))
        for p in (ROOT / "src").rglob("*")
        if p.is_file() and p.suffix not in {".py", ".pyc"} and p.name != ".DS_Store"
    ]
    assert not unexpected, "\n".join(unexpected)


def test_documentation_links_resolve():
    import re
    from urllib.parse import unquote, urlsplit

    from tests.support import ROOT

    missing = []
    for path in [ROOT / "README.md", *(ROOT / "docs").rglob("*.md")]:
        if "references" in path.parts:
            continue  # Acquired upstream snapshots retain their original bytes.
        for target in re.findall(r"\]\(([^)]+)\)", path.read_text()):
            url = urlsplit(target.strip("<>"))
            if (
                not url.scheme
                and url.path
                and not (path.parent / unquote(url.path)).exists()
            ):
                missing.append(f"{path.relative_to(ROOT)} -> {target}")
    assert not missing, "\n".join(missing)
