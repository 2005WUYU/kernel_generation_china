"""Schema for a complete, independently runnable model candidate."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

MODEL_RESPONSE_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": "https://ascend-kernel-lab.local/schemas/model-response-v4.json",
    "title": "Ascend Kernel Candidate",
    "type": "object",
    "additionalProperties": True,
    "required": [
        "status",
        "round",
        "code",
    ],
    "properties": {
        "status": {"type": "string", "enum": ["candidate", "no_change"]},
        "round": {"type": "integer", "minimum": 1},
        "changes": {
            "description": (
                "Optional model-authored description of changes embodied in code. "
                "Its shape is advisory and never gates candidate execution."
            ),
        },
        "evidence": {
            "description": "Optional advisory evidence recorded by the model.",
        },
        "hypotheses": {
            "description": "Optional advisory performance hypotheses from the model.",
        },
        "predictions": {
            "description": "Optional advisory predictions from the model.",
        },
        "code": {"type": "string", "minLength": 1, "maxLength": 262_144},
    },
}


def model_response_schema() -> dict[str, Any]:
    """Return a defensive copy suitable for handing to an external client."""

    return deepcopy(MODEL_RESPONSE_SCHEMA)
