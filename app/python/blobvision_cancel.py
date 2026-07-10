"""Cooperative cancel flag for long VQGAN optimisation runs."""
import threading

_lock = threading.Lock()
_flag = False


class AbortedError(Exception):
    """Raised when the user aborts an in-progress generation."""


def clear():
    global _flag
    with _lock:
        _flag = False


def request():
    global _flag
    with _lock:
        _flag = True


def is_requested():
    with _lock:
        return _flag
