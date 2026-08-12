"""Tiny single-threaded rlimit launcher used by :mod:`stage_runner`."""

from __future__ import annotations

import argparse
import math
import os
import resource
from collections.abc import Sequence


def _set_limit(kind: int, requested: int) -> None:
    try:
        _soft, hard = resource.getrlimit(kind)
        effective = requested if hard == resource.RLIM_INFINITY else min(requested, hard)
        resource.setrlimit(kind, (effective, effective))
    except (OSError, ValueError):
        # A platform/container may prohibit one limit. The parent still enforces
        # the wall timeout and output cap, while all supported limits remain set.
        return


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--address-space", type=int, required=True)
    parser.add_argument("--file-size", type=int, required=True)
    parser.add_argument("--open-files", type=int, required=True)
    parser.add_argument("--processes", type=int, required=True)
    parser.add_argument("--core-size", type=int, required=True)
    parser.add_argument("--cpu-seconds", type=float, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    options = parser.parse_args(argv)
    command = list(options.command)
    if command and command[0] == "--":
        command.pop(0)
    if not command:
        raise ValueError("launcher requires a command")
    values = (
        options.address_space,
        options.file_size,
        options.open_files,
        options.processes,
        options.core_size,
    )
    if any(value < 0 for value in values) or options.cpu_seconds <= 0:
        raise ValueError("launcher resource limits must be positive or zero")
    _set_limit(resource.RLIMIT_CORE, options.core_size)
    _set_limit(resource.RLIMIT_FSIZE, options.file_size)
    _set_limit(resource.RLIMIT_NOFILE, options.open_files)
    _set_limit(resource.RLIMIT_NPROC, options.processes)
    _set_limit(resource.RLIMIT_AS, options.address_space)
    _set_limit(resource.RLIMIT_CPU, max(1, math.ceil(options.cpu_seconds)))
    os.execvpe(command[0], command, dict(os.environ))


if __name__ == "__main__":  # pragma: no cover - exercised via StageRunner
    main()
