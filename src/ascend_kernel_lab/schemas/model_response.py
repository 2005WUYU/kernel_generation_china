"""Schema for a complete, independently runnable model candidate."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

MODEL_RESPONSE_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": "https://ascend-kernel-lab.local/schemas/model-response-v2.json",
    "title": "Ascend Kernel Candidate",
    "type": "object",
    "additionalProperties": False,
    "required": [
        "status",
        "round",
        "optimization_summary",
        "expected_effect",
        "assumptions",
        "code",
    ],
    "properties": {
        "status": {"type": "string", "enum": ["candidate", "no_change"]},
        "round": {"type": "integer", "minimum": 1},
        "optimization_summary": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 1,
            "maxItems": 32,
            "description": (
                "The model's contemporaneous summary of the concrete candidate "
                "generation or optimization choices embodied in code. This is "
                "returned by the model, never inferred after evaluation."
            ),
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
