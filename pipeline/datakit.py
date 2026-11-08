"""Shared storage, hashing, locking and process-identity helpers for pipeline code.

One implementation of the primitives every stage needs, so durability and
identity rules cannot drift between scripts. Standard library only.
"""
import contextlib
import fcntl
import hashlib
import json
import os
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# The user-designated data root. There is deliberately no fallback location.
DATA = Path('/mnt/f/rwkv-pi-data')
HOLDOUT_MANIFEST = ROOT / 'datasets/eval-holdout-v1/manifest.json'


# -- hashing -----------------------------------------------------------------

def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode())


def sha256_file(path) -> str:
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def fingerprint(value) -> str:
    """Order-independent digest of a JSON value."""
    return sha256_text(json.dumps(value, sort_keys=True))


# -- files -------------------------------------------------------------------

def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    """Durably replace a JSON state file (fsync before rename; unique temp name)."""
    path = Path(path)
    tmp = path.with_name(f'{path.name}.{uuid.uuid4().hex}.tmp')
    with tmp.open('w') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)


def write_text(path, text: str):
    """Atomically replace a text file, creating parent directories."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f'{path.name}.{os.getpid()}.tmp')
    tmp.write_text(text)
    os.replace(tmp, path)


def pretty_json(value) -> str:
    """Human-readable, UTF-8-preserving JSON text used for authored specifications."""
    return json.dumps(value, ensure_ascii=False, indent=2) + '\n'


def require_data_root(path) -> Path:
    """Refuse any output that is not on the mounted F: data root."""
    path = Path(path).resolve()
    if not Path('/mnt/f').is_mount() or not path.is_relative_to(DATA):
        raise ValueError(f'Mounted F: data root required, got {path}')
    return path


# -- locks -------------------------------------------------------------------

def try_lock(path):
    """Non-blocking exclusive lock. The returned file holds it; use it as a context manager."""
    stream = Path(path).open('a+')
    try:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        stream.close()
        raise
    return stream


@contextlib.contextmanager
def locked(path):
    """Blocking exclusive lock for short critical sections."""
    with Path(path).open('a+') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        yield


# -- processes ---------------------------------------------------------------

def boot_id() -> str:
    return Path('/proc/sys/kernel/random/boot_id').read_text().strip()


def process_identity(pid):
    """{pid, startTicks, bootId}; None when the process is gone or a zombie.

    PID reuse cannot impersonate a recorded process: start ticks and boot ID must match too.
    """
    try:
        fields = Path(f'/proc/{pid}/stat').read_text().rsplit(') ', 1)[1].split()
    except (FileNotFoundError, ProcessLookupError):
        return None
    if fields[0] == 'Z':
        return None
    return {'pid': pid, 'startTicks': fields[19], 'bootId': boot_id()}


def is_alive(identity) -> bool:
    return bool(identity and process_identity(identity['pid']) == identity)


# -- policy ------------------------------------------------------------------

def heldout_families() -> set:
    return set(read_json(HOLDOUT_MANIFEST)['families'])
