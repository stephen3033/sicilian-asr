import subprocess
import sys

from tests.support import ROOT, copy_configuration


def test_downloader_dry_run_does_not_acquire_resources(tmp_path):
    copy_configuration(tmp_path)
    before = sorted(p.relative_to(tmp_path) for p in tmp_path.rglob("*") if p.is_file())
    process = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/download_resources.py"),
            "--root",
            str(tmp_path),
            "--all",
            "--dry-run",
        ],
        capture_output=True,
        text=True,
    )
    assert process.returncode == 0, process.stderr
    assert "WARNING fbk_ita_lm: unavailable" in process.stdout
    assert "zipa_checkpoint:" in process.stdout
    assert not (tmp_path / ".cache").exists()
    assert (
        sorted(p.relative_to(tmp_path) for p in tmp_path.rglob("*") if p.is_file())
        == before
    )
