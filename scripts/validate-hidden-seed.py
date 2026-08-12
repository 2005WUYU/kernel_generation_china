#!/usr/bin/env python3
"""Validate the worker/controller hidden seed without printing its value."""

from __future__ import annotations

import os
import sys


def main() -> int:
    raw = os.environ.get("AKG_HIDDEN_SEED", "")
    if not raw or any(character not in "0123456789" for character in raw):
        print(
            "error: AKG_HIDDEN_SEED must be an unsigned base-10 integer",
            file=sys.stderr,
        )
        return 10
    value = int(raw, 10)
    if value <= 0 or value.bit_length() < 128 or value.bit_length() > 256:
        print(
            "error: AKG_HIDDEN_SEED must be a positive 128-to-256-bit integer",
            file=sys.stderr,
        )
        return 10
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
