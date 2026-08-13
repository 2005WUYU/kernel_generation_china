"""Schema for a complete, independently runnable model candidate."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

MODEL_RESPONSE_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": "https://ascend-kernel-lab.local/schemas/model-response-v3.json",
    "title": "Ascend Kernel Candidate",
    "type": "object",
    "additionalProperties": False,
    "required": [
        "status",
        "round",
        "changes",
        "evidence",
        "hypotheses",
        "predictions",
        "code",
    ],
    "properties": {
        "status": {"type": "string", "enum": ["candidate", "no_change"]},
        "round": {"type": "integer", "minimum": 1},
        "changes": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["target", "before", "after"],
                "properties": {
                    "target": {"type": "string", "minLength": 1},
                    "before": {},
                    "after": {},
                },
            },
            "minItems": 1,
            "maxItems": 32,
            "description": (
                "Concrete model-authored changes embodied in code. These are "
                "returned with the candidate and are never inferred by the evaluator."
            ),
        },
        "evidence": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["fact", "source"],
                "properties": {
                    "fact": {"type": "string", "minLength": 1},
                    "source": {"type": "string", "minLength": 1},
                },
            },
            "minItems": 1,
            "maxItems": 32,
        },
        "hypotheses": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["claim", "confidence", "evidence_refs"],
                "properties": {
                    "claim": {"type": "string", "minLength": 1},
                    "confidence": {
                        "type": "string",
                        "enum": ["low", "medium", "high"],
                    },
                    "evidence_refs": {
                        "type": "array",
                        "items": {"type": "integer", "minimum": 0},
                        "minItems": 1,
                        "maxItems": 32,
                        "uniqueItems": True,
                    },
                },
            },
            "minItems": 1,
            "maxItems": 32,
        },
        "predictions": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["metric", "expected_direction", "reason"],
                "properties": {
                    "metric": {"type": "string", "minLength": 1},
                    "expected_direction": {
                        "type": "string",
                        "enum": ["increase", "decrease", "unchanged"],
                    },
                    "reason": {
                        "type": "string",
                        "pattern": "^hypothesis\\[[0-9]+\\]$",
                    },
                },
            },
            "minItems": 1,
            "maxItems": 32,
        },
        "code": {"type": "string", "minLength": 1, "maxLength": 262_144},
    },
}


def model_response_schema() -> dict[str, Any]:
    """Return a defensive copy suitable for handing to an external client."""

    return deepcopy(MODEL_RESPONSE_SCHEMA)
