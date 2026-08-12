from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any

_TASK_ID = re.compile(r"^k(0[1-9]|10)_[a-z0-9_]+$")
_CASE_KINDS = frozenset({"correctness", "benchmark", "profile"})
_DTYPES = frozenset({"float16", "bfloat16", "float32"})
_DISTRIBUTIONS = frozenset({"normal", "near_zero", "large", "zeros", "repeated"})
_CASE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_PARAMETER = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,63}$")
_TRUSTED_TASK_FILES = (
    "baseline.py",
    "hidden_template.json",
    "input_generator.py",
    "output_validator.py",
    "public_cases.jsonl",
    "reference.py",
    "task.yaml",
)


class TaskSpecError(ValueError):
    """Raised when a task artifact violates the versioned task protocol."""


@dataclass(frozen=True)
class CaseSpec:
    id: str
    kind: str
    dtype: str
    params: Mapping[str, int]
    distribution: str = "normal"
    seed: int = 0
    weight: float = 1.0
    address_offset: int = 0
    noncontiguous: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not _CASE_ID.fullmatch(self.id):
            raise TaskSpecError("case id must be a safe 1-128 character identifier")
        if self.kind not in _CASE_KINDS:
            raise TaskSpecError(f"unsupported case kind: {self.kind}")
        if self.dtype not in _DTYPES:
            raise TaskSpecError(f"unsupported dtype: {self.dtype}")
        if self.distribution not in _DISTRIBUTIONS:
            raise TaskSpecError(f"unsupported input distribution: {self.distribution}")
        if not isinstance(self.params, Mapping) or not self.params:
            raise TaskSpecError("case params must be a non-empty mapping")
        normalized_params: dict[str, int] = {}
        for raw_name, raw_value in self.params.items():
            if not isinstance(raw_name, str) or not _PARAMETER.fullmatch(raw_name):
                raise TaskSpecError("case parameter names must be safe identifiers")
            if isinstance(raw_value, bool) or not isinstance(raw_value, int):
                raise TaskSpecError("case dimensions must be integers")
            if not 0 < raw_value <= 2**31 - 1:
                raise TaskSpecError("case dimensions must be in [1, 2^31-1]")
            normalized_params[raw_name] = raw_value
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise TaskSpecError("case seed must be an integer")
        if not 0 <= self.seed <= 2**63 - 1:
            raise TaskSpecError("case seed must be in [0, 2^63-1]")
        if isinstance(self.weight, bool) or not isinstance(self.weight, (int, float)):
            raise TaskSpecError("case weight must be numeric")
        weight = float(self.weight)
        if not math.isfinite(weight) or weight <= 0:
            raise TaskSpecError("case weight must be finite and positive")
        if isinstance(self.address_offset, bool) or not isinstance(self.address_offset, int):
            raise TaskSpecError("case address_offset must be an integer")
        if not 0 <= self.address_offset <= 4096:
            raise TaskSpecError("case address_offset must be in [0, 4096]")
        if not isinstance(self.noncontiguous, bool):
            raise TaskSpecError("case noncontiguous must be a boolean")
        object.__setattr__(self, "params", MappingProxyType(normalized_params))
        object.__setattr__(self, "weight", weight)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> CaseSpec:
        required = {"id", "kind", "dtype", "params"}
        unknown = set(value) - {
            *required,
            "distribution",
            "seed",
            "weight",
            "address_offset",
            "noncontiguous",
        }
        missing = required - set(value)
        if missing or unknown:
            raise TaskSpecError(f"invalid case keys; missing={sorted(missing)}, unknown={sorted(unknown)}")
        kind = value["kind"]
        dtype = value["dtype"]
        if not isinstance(kind, str) or not isinstance(dtype, str):
            raise TaskSpecError("case kind and dtype must be strings")
        if kind not in _CASE_KINDS:
            raise TaskSpecError(f"unsupported case kind: {kind}")
        if dtype not in _DTYPES:
            raise TaskSpecError(f"unsupported dtype: {dtype}")
        params_raw = value["params"]
        if not isinstance(params_raw, Mapping) or not params_raw:
            raise TaskSpecError("case params must be a non-empty mapping")
        params = dict(params_raw)
        distribution = value.get("distribution", "normal")
        seed = value.get("seed", 0)
        weight = value.get("weight", 1.0)
        address_offset = value.get("address_offset", 0)
        noncontiguous = value.get("noncontiguous", False)
        return cls(
            id=value["id"],
            kind=kind,
            dtype=dtype,
            params=params,
            distribution=distribution,
            seed=seed,
            weight=weight,
            address_offset=address_offset,
            noncontiguous=noncontiguous,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "dtype": self.dtype,
            "params": dict(self.params),
            "distribution": self.distribution,
            "seed": self.seed,
            "weight": self.weight,
            "address_offset": self.address_offset,
            "noncontiguous": self.noncontiguous,
        }


@dataclass(frozen=True)
class TaskSpec:
    id: str
    version: int
    name: str
    description: str
    entry_point: str
    inputs: tuple[Mapping[str, Any], ...]
    outputs: tuple[Mapping[str, Any], ...]
    semantics: Mapping[str, Any]
    correctness: Mapping[str, Any]
    benchmark: Mapping[str, Any]
    restrictions: Mapping[str, Any]
    public_cases: tuple[CaseSpec, ...] = field(default_factory=tuple)
    root: Path | None = None

    @property
    def correctness_cases(self) -> tuple[CaseSpec, ...]:
        return tuple(case for case in self.public_cases if case.kind == "correctness")

    @property
    def benchmark_cases(self) -> tuple[CaseSpec, ...]:
        return tuple(case for case in self.public_cases if case.kind == "benchmark")

    @property
    def profile_cases(self) -> tuple[CaseSpec, ...]:
        explicit = tuple(case for case in self.public_cases if case.kind == "profile")
        return explicit or self.benchmark_cases[:1]

    def public_prompt_view(self) -> dict[str, Any]:
        """Return the only task representation allowed to enter a model prompt."""
        return {
            "task_id": self.id,
            "version": self.version,
            "description": self.description,
            "entry_point": self.entry_point,
            "inputs": list(self.inputs),
            "outputs": list(self.outputs),
            "semantics": dict(self.semantics),
            "public_cases": [case.to_dict() for case in self.public_cases],
            "correctness_policy": dict(self.correctness),
            "benchmark_policy": dict(self.benchmark),
            "restrictions": dict(self.restrictions),
        }

    def digest(self) -> str:
        data = self.public_prompt_view()
        data["name"] = self.name
        encoded = json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
        return hashlib.sha256(encoded).hexdigest()

    def bundle_digest(self) -> str:
        """Hash every trusted task executable/template artifact by path and bytes."""

        if self.root is None:
            raise TaskSpecError("task bundle digest requires a registry-backed root")
        digest = hashlib.sha256(b"ascend-task-bundle-v1\0")
        for name in _TRUSTED_TASK_FILES:
            path = self.root / name
            if path.is_symlink() or not path.is_file():
                raise TaskSpecError(f"trusted task bundle file is missing or unsafe: {name}")
            payload = path.read_bytes()
            digest.update(name.encode("utf-8"))
            digest.update(b"\0")
            digest.update(len(payload).to_bytes(8, "big"))
            digest.update(payload)
        return digest.hexdigest()


def _load_yaml(path: Path) -> Mapping[str, Any]:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - exercised by packaging smoke tests
        raise RuntimeError("PyYAML is required; install the project dependencies") from exc
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise TaskSpecError(f"{path} must contain a mapping")
    return value


def _load_jsonl(path: Path) -> tuple[CaseSpec, ...]:
    cases: list[CaseSpec] = []
    seen: set[str] = set()
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as exc:
            raise TaskSpecError(f"{path}:{line_no}: {exc}") from exc
        if not isinstance(raw, Mapping):
            raise TaskSpecError(f"{path}:{line_no}: case must be an object")
        case = CaseSpec.from_dict(raw)
        if case.id in seen:
            raise TaskSpecError(f"duplicate case id {case.id!r} in {path}")
        seen.add(case.id)
        cases.append(case)
    return tuple(cases)


class TaskRegistry:
    """Loads immutable task specs from a repository checkout."""

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root).resolve()

    def ids(self) -> tuple[str, ...]:
        if not self.root.is_dir():
            return ()
        return tuple(sorted(path.name for path in self.root.iterdir() if (path / "task.yaml").is_file()))

    def load(self, task_id: str) -> TaskSpec:
        if not _TASK_ID.fullmatch(task_id):
            raise TaskSpecError(f"invalid task id: {task_id!r}")
        task_root = (self.root / task_id).resolve()
        if task_root.parent != self.root or not task_root.is_dir():
            raise TaskSpecError(f"unknown task: {task_id}")
        raw = _load_yaml(task_root / "task.yaml")
        allowed = {
            "id", "version", "name", "description", "entry_point", "inputs", "outputs",
            "semantics", "correctness", "benchmark", "restrictions",
        }
        unknown = set(raw) - allowed
        missing = allowed - set(raw)
        if unknown or missing:
            raise TaskSpecError(f"invalid task keys; missing={sorted(missing)}, unknown={sorted(unknown)}")
        if raw["id"] != task_id:
            raise TaskSpecError(f"directory id {task_id} does not match task id {raw['id']}")
        if raw["entry_point"] != "custom_op":
            raise TaskSpecError("entry point must be custom_op")
        cases = _load_jsonl(task_root / "public_cases.jsonl")
        if not 8 <= sum(case.kind == "correctness" for case in cases) <= 12:
            raise TaskSpecError(f"{task_id} must define 8-12 public correctness cases")
        if not 3 <= sum(case.kind == "benchmark" for case in cases) <= 5:
            raise TaskSpecError(f"{task_id} must define 3-5 public benchmark cases")
        return TaskSpec(
            id=task_id,
            version=int(raw["version"]),
            name=str(raw["name"]),
            description=str(raw["description"]),
            entry_point="custom_op",
            inputs=tuple(raw["inputs"]),
            outputs=tuple(raw["outputs"]),
            semantics=dict(raw["semantics"]),
            correctness=dict(raw["correctness"]),
            benchmark=dict(raw["benchmark"]),
            restrictions=dict(raw["restrictions"]),
            public_cases=cases,
            root=task_root,
        )

    def load_many(self, ids: Iterable[str] | None = None) -> tuple[TaskSpec, ...]:
        return tuple(self.load(task_id) for task_id in (ids if ids is not None else self.ids()))
