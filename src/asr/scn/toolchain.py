"""Build pinned KenLM command-line tools locally; no system installation."""

import hashlib
import json
import subprocess
import tarfile
from pathlib import Path
from urllib.request import urlopen

SOURCES = {
    "boost": (
        "https://archives.boost.io/release/1.85.0/source/boost_1_85_0.tar.bz2",
        "7009fe1faa1697476bdc7027703a2badb84e849b7b0baad5086b087b971f8617",
        "boost_1_85_0",
    ),
    "kenlm": (
        "https://files.pythonhosted.org/packages/02/6e/f4e159cef39ea4ef2244b4aa32857d447d60a5d0ee269e1b1f8d3723c03e/kenlm-0.3.0.tar.gz",
        "c4628bb9fb63c8a6f9240035b8b037385cfc404cb72e933cf48878291edac1e8",
        "kenlm-0.3.0",
    ),
}


def main(root: Path | None = None):
    root = (root or Path.cwd()).resolve()
    base = root / ".cache/scn/tools"
    base.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, (url, expected, directory) in SOURCES.items():
        archive = base / url.rsplit("/", 1)[1]
        if not archive.exists():
            print(f"Downloading {name}", flush=True)
            with urlopen(url, timeout=180) as response:
                data = response.read()
            if hashlib.sha256(data).hexdigest() != expected:
                raise RuntimeError(f"Checksum mismatch: {name}")
            archive.write_bytes(data)
        if hashlib.sha256(archive.read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"Checksum mismatch: {archive}")
        paths[name] = base / directory
        if not paths[name].exists():
            with tarfile.open(archive) as stream:
                stream.extractall(base, filter="data")
    boost = paths["boost"]
    prefix = base / "boost-install"
    if not (prefix / "lib/libboost_program_options.a").exists():
        subprocess.run(
            [
                "./bootstrap.sh",
                f"--prefix={prefix}",
                "--with-libraries=program_options,system,thread,test",
            ],
            cwd=boost,
            check=True,
        )
        subprocess.run(
            ["./b2", "-j4", "link=static", "variant=release", "install"],
            cwd=boost,
            check=True,
        )
    build = base / "kenlm-build"
    cmake = root / ".venv/bin/cmake"
    subprocess.run(
        [
            str(cmake),
            "-S",
            str(paths["kenlm"]),
            "-B",
            str(build),
            f"-DBOOST_ROOT={prefix}",
            "-DBoost_NO_SYSTEM_PATHS=ON",
            "-DBoost_USE_STATIC_LIBS=ON",
            "-DCMAKE_POLICY_DEFAULT_CMP0167=OLD",
            "-DCMAKE_BUILD_TYPE=Release",
            "-DCOMPILE_TESTS=OFF",
            "-DENABLE_INTERPOLATE=OFF",
        ],
        check=True,
    )
    subprocess.run(
        [str(cmake), "--build", str(build), "--target", "lmplz", "build_binary", "-j4"],
        check=True,
    )
    receipt = {"sources": SOURCES, "tools": {}}
    for name in ("lmplz", "build_binary"):
        path = build / "bin" / name
        receipt["tools"][name] = {
            "path": str(path.relative_to(root)),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    (base / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    main()
