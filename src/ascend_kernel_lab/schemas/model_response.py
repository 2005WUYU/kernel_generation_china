"""Schema for a complete, independently runnable model candidate."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

MODEL_RESPONSE_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": "https://ascend-kernel-lab.local/schemas/model-response-v1.json",
    "title": "Ascend Kernel Candidate",
    "type": "object",
    "additionalProperties": False,
    "required": [
        "status",
        "round",
        "change_summary",
        "expected_effect",
        "assumptions",
        "code",
    ],
    "properties": {
        "status": {"type": "string", "enum": ["candidate", "no_change"]},
        "round": {"type": "integer", "minimum": 1},
        "change_summary": {
            "type": "array",
            "items": {"type": "string"},
            "maxItems": 32,
        },
        "expected_effect": {
            "type": "array",
            "items": {"type": "string"},
            "maxItems": 32,
        },
        "assumptions": {
            "type": "array",
            "items": {"type": "string"},
            "maxItems": 32,
        },
        "code": {"type": "string", "minLength": 1, "maxLength": 262_144},
    },
}


def model_response_schema() -> dict[str, Any]:
    """Return a defensive copy suitable for handing to an external client."""

    return deepcopy(MODEL_RESPONSE_SCHEMA)
