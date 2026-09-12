"""Capture stdout/stderr for the BlobVision UI console."""
import sys
import threading

_MAX_LINES = 300
_lines = []
_lock = threading.Lock()
_installed = False

# Known noisy lines (offline fp16 park, optional triton, etc.)
_SKIP_FRAGMENTS = (
    "triton not found",
    "flop counting will not work",
    "cannot run with `cpu` device",
    "float16` operations on this device",
    "it is not recommended to move them to `cpu`",
)


def _drop_line(line):
    low = (line or "").lower()
    return any(frag in low for frag in _SKIP_FRAGMENTS)


def append(line):
    line = (line or "").rstrip()
    if not line or _drop_line(line):
        return
    with _lock:
        _lines.append(line)
        if len(_lines) > _MAX_LINES:
            del _lines[: len(_lines) - _MAX_LINES]


def tail(n=80):
    with _lock:
        return "\n".join(_lines[-n:])


class _Tee:
    def __init__(self, stream):
        self._stream = stream

    def write(self, data):
        if not data:
            return
        if any(_drop_line(part) for part in data.splitlines() if part.strip()):
            return
        try:
            self._stream.write(data)
        except OSError:
            return
        if data:
            for part in data.splitlines():
                append(part)

    def flush(self):
        self._stream.flush()

    def isatty(self):
        return getattr(self._stream, "isatty", lambda: False)()


def install():
    global _installed
    if _installed:
        return
    sys.stdout = _Tee(sys.stdout)
    sys.stderr = _Tee(sys.stderr)
    _installed = True
