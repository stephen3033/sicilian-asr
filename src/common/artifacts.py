"""Checksummed experiment artifacts and bounded native-process execution."""

import hashlib
import json
import os
import shutil
import signal
import subprocess
from pathlib import Path

from common.errors import ResourceError
from common.utils import local_path, write_json


def digest(value):
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True).encode()
    ).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verified_files(directory, receipt):
    for name, expected in receipt["artifacts"].items():
        path = local_path(directory, name)
        if not path.is_file() or sha(path) != expected:
            raise ResourceError(f"Artifact checksum mismatch: {path}")


def seal(directory, name, identity, names):
    receipt = {
        "status": "complete",
        "key": digest(identity),
        "identity": identity,
        "artifacts": {n: sha(directory / n) for n in sorted(names)},
    }
    write_json(directory / name, receipt)
    return receipt


def space_check(directory, required=0, reserve=5 * 1024**3):
    free = shutil.disk_usage(directory).free
    if required < 0 or free - required < reserve:
        raise ResourceError(
            f"Insufficient disk space: {free} free, {required} required, {reserve} reserved"
        )
    return {"free_bytes": free, "required_bytes": required, "reserve_bytes": reserve}


def run_native(command, log, *, memory_limit=24 * 1024**3):
    """Serialize native experiments so the process-group cap is also a job cap."""
    import fcntl
    lock_path = Path.cwd() / ".cache/scn-experiments/native-worker.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        _run_native_unlocked(command, log, memory_limit=memory_limit)


def _run_native_unlocked(command, log, *, memory_limit):
    """Bound the worker process group; terminate and reap on errors/interruption."""
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w") as stream:
        proc = subprocess.Popen(
            command,
            stdout=stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            env={**os.environ, "OMP_NUM_THREADS": "4"},
        )
        try:
            while proc.poll() is None:
                listing = subprocess.check_output(
                    ["ps", "-axo", "pgid=,rss="], text=True
                )
                rss = sum(
                    int(parts[1]) * 1024
                    for line in listing.splitlines()
                    if len(parts := line.split()) == 2 and int(parts[0]) == proc.pid
                )
                if rss > memory_limit:
                    raise ResourceError(
                        f"Worker exceeds {memory_limit} byte memory limit; {log}"
                    )
                try:
                    proc.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    pass
            if proc.returncode:
                raise ResourceError(f"Worker exited {proc.returncode}; inspect {log}")
        finally:
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGTERM)
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait()
