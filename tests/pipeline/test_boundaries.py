"""Protect dependency direction and installed entry points during future changes."""

import ast
import subprocess
import sys
from pathlib import Path

from tests.support import ROOT


def test_layers_do_not_import_orchestration_or_other_languages():
    allowed = {
        "common": {"common"},
        "media_input": {"common", "media_input"},
        "audio": {"common", "audio"},
        "ipa": {"common", "ipa"},
        "evaluation": {"common", "evaluation"},
        "asr.wfst": {"common", "ipa", "asr.wfst"},
        "asr.ita": {"common", "ipa", "evaluation", "asr.wfst", "asr.ita"},
        "asr.scn": {"common", "ipa", "evaluation", "asr.wfst", "asr.scn"},
    }
    local = {p.name for p in (ROOT / "src").iterdir() if p.is_dir()}
    failures = []
    for layer, dependencies in allowed.items():
        for path in (ROOT / "src" / layer.replace(".", "/")).rglob("*.py"):
            for node in ast.walk(ast.parse(path.read_text())):
                names = (
                    [n.name for n in node.names]
                    if isinstance(node, ast.Import)
                    else [node.module]
                    if isinstance(node, ast.ImportFrom)
                    else []
                )
                for name in filter(None, names):
                    if name.split(".")[0] not in local:
                        continue
                    if not any(
                        name == dependency or name.startswith(dependency + ".")
                        for dependency in dependencies
                    ):
                        failures.append(f"{path.relative_to(ROOT)} imports {name}")
    assert not failures, "\n".join(failures)


def test_console_and_module_entry_points_match():
    console = Path(sys.executable).with_name("sicilian-asr")
    for command in ([str(console)], [sys.executable, "-m", "pipeline"]):
        result = subprocess.run(
            [*command, "transcribe", "--help"], capture_output=True, text=True
        )
        assert result.returncode == 0, result.stderr
        assert "--diagnostics" in result.stdout
        assert "ita, scn, ita_scn" in result.stdout
    assert not (ROOT / "src/sicilian_asr").exists()
