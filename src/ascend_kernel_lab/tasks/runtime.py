from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .loader import CaseSpec, TaskSpec


@dataclass(frozen=True)
class GeneratedInputs:
    args: tuple[Any, ...]
    metadata: Mapping[str, Any]


def _dtype(torch: Any, name: str) -> Any:
    try:
        return {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[name]
    except KeyError as exc:
        raise ValueError(f"unsupported dtype: {name}") from exc


def _sample(torch: Any, shape: Sequence[int], case: CaseSpec, device: str) -> Any:
    dtype = _dtype(torch, case.dtype)
    generator = torch.Generator(device=device)
    generator.manual_seed(case.seed)
    backing_shape = list(shape)
    backing_shape[-1] += max(0, case.address_offset)
    distribution = case.distribution
    if distribution == "zeros":
        backing = torch.zeros(backing_shape, dtype=dtype, device=device)
    elif distribution == "near_zero":
        backing = torch.randn(backing_shape, dtype=dtype, device=device, generator=generator) * 1e-3
    elif distribution == "large":
        backing = torch.randn(backing_shape, dtype=dtype, device=device, generator=generator) * 8
    elif distribution == "positive":
        backing = torch.rand(backing_shape, dtype=dtype, device=device, generator=generator) + 0.1
    elif distribution == "repeated":
        backing = torch.full(backing_shape, 0.375, dtype=dtype, device=device)
    else:
        backing = torch.randn(backing_shape, dtype=dtype, device=device, generator=generator)
    if case.address_offset:
        backing = backing[..., case.address_offset :]
    if case.noncontiguous and len(shape) >= 2:
        enlarged = torch.empty((*shape[:-1], shape[-1] * 2), dtype=dtype, device=device)
        enlarged[..., ::2].copy_(backing)
        backing = enlarged[..., ::2]
    return backing


def generate_inputs(spec: TaskSpec, case: CaseSpec, torch: Any, device: str = "npu:0") -> GeneratedInputs:
    p = case.params
    args: tuple[Any, ...]
    if spec.id == "k01_vector_add":
        vector_shape = (p["n"],)
        args = (
            _sample(torch, vector_shape, case, device),
            _sample(torch, vector_shape, _with_seed(case, 17), device),
        )
    elif spec.id in {"k02_bias_gelu", "k03_swiglu"}:
        matrix_shape = (p["m"], p["n"])
        first = _sample(torch, matrix_shape, case, device)
        second_shape = (p["n"],) if spec.id == "k02_bias_gelu" else matrix_shape
        args = (first, _sample(torch, second_shape, _with_seed(case, 17), device))
    elif spec.id == "k04_transpose" or spec.id == "k05_row_softmax":
        args = (_sample(torch, (p["m"], p["n"]), case, device),)
    elif spec.id == "k06_rmsnorm":
        rms_shape = (p["m"], p["n"])
        args = (
            _sample(torch, rms_shape, case, device),
            _sample(torch, (p["n"],), _with_seed(case, 17), device),
        )
    elif spec.id == "k07_layernorm":
        layer_shape = (p["m"], p["n"])
        args = (
            _sample(torch, layer_shape, case, device),
            _sample(torch, (p["n"],), _with_seed(case, 17), device),
            _sample(torch, (p["n"],), _with_seed(case, 29), device),
        )
    elif spec.id == "k08_rope":
        rope_shape = (p["seq"], p["heads"], p["dim"])
        x = _sample(torch, rope_shape, case, device)
        positions = torch.arange(p["seq"], device=device, dtype=torch.float32)[:, None]
        inv_freq = 1.0 / (10000 ** (torch.arange(0, p["dim"], 2, device=device, dtype=torch.float32) / p["dim"]))
        angles = positions * inv_freq[None, :]
        args = (x, torch.cos(angles).to(dtype=x.dtype), torch.sin(angles).to(dtype=x.dtype))
    elif spec.id in {"k09_gemm", "k10_gemm_bias_gelu"}:
        a = _sample(torch, (p["m"], p["k"]), case, device)
        b = _sample(torch, (p["k"], p["n"]), _with_seed(case, 17), device)
        gemm_args: tuple[Any, ...] = (a, b)
        if spec.id == "k10_gemm_bias_gelu":
            gemm_args += (
                _sample(torch, (p["n"],), _with_seed(case, 29), device),
            )
        args = gemm_args
    else:  # pragma: no cover - registry constrains built-in ids
        raise ValueError(f"no input generator for {spec.id}")
    return GeneratedInputs(args=args, metadata={"case_id": case.id, "seed": case.seed})


def _with_seed(case: CaseSpec, delta: int) -> CaseSpec:
    return CaseSpec(**{**case.__dict__, "seed": case.seed + delta})


def reference(spec: TaskSpec, args: Sequence[Any], torch: Any) -> Any:
    """Trusted eager reference. It must never be imported by candidate code."""
    if spec.id == "k01_vector_add":
        return args[0] + args[1]
    if spec.id == "k02_bias_gelu":
        return torch.nn.functional.gelu(args[0] + args[1], approximate="tanh")
    if spec.id == "k03_swiglu":
        return torch.nn.functional.silu(args[0]) * args[1]
    if spec.id == "k04_transpose":
        return args[0].transpose(0, 1).contiguous()
    if spec.id == "k05_row_softmax":
        return torch.softmax(args[0].float(), dim=-1).to(args[0].dtype)
    if spec.id == "k06_rmsnorm":
        eps = float(spec.semantics["epsilon"])
        x = args[0].float()
        return (x * torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + eps) * args[1].float()).to(args[0].dtype)
    if spec.id == "k07_layernorm":
        eps = float(spec.semantics["epsilon"])
        return torch.nn.functional.layer_norm(args[0], (args[0].shape[-1],), args[1], args[2], eps)
    if spec.id == "k08_rope":
        x, cos, sin = args
        even, odd = x[..., 0::2], x[..., 1::2]
        cos = cos[:, None, :]
        sin = sin[:, None, :]
        out = torch.empty_like(x)
        out[..., 0::2] = even * cos - odd * sin
        out[..., 1::2] = even * sin + odd * cos
        return out
    if spec.id == "k09_gemm":
        return torch.matmul(args[0].float(), args[1].float()).to(args[0].dtype)
    if spec.id == "k10_gemm_bias_gelu":
        product = torch.matmul(args[0].float(), args[1].float()) + args[2].float()
        return torch.nn.functional.gelu(product, approximate="tanh").to(args[0].dtype)
    raise ValueError(f"no reference for {spec.id}")


def tolerances(spec: TaskSpec, dtype: str) -> tuple[float, float]:
    rtol = float(spec.correctness["rtol"][dtype])
    atol = float(spec.correctness["atol"][dtype])
    return rtol, atol


def validate_output(
    spec: TaskSpec,
    case: CaseSpec,
    actual: Any,
    expected: Any,
    torch: Any,
    *,
    inputs: Sequence[Any] = (),
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "case_id": case.id,
        "passed": False,
        "shape_ok": False,
        "dtype_ok": False,
        "device_ok": False,
        "layout_ok": False,
        "output_alias_ok": False,
        "finite_ok": False,
        "maximum_absolute_error": None,
        "maximum_relative_error": None,
    }
    if not isinstance(actual, torch.Tensor):
        result["error"] = "custom_op did not return a Tensor"
        return result
    result["shape_ok"] = tuple(actual.shape) == tuple(expected.shape)
    result["dtype_ok"] = actual.dtype == expected.dtype
    result["device_ok"] = actual.device == expected.device
    result["layout_ok"] = bool(actual.is_contiguous())
    try:
        actual_storage = int(actual.untyped_storage().data_ptr())
        input_storages = {
            int(value.untyped_storage().data_ptr())
            for value in inputs
            if isinstance(value, torch.Tensor)
        }
        result["output_alias_ok"] = actual_storage not in input_storages
    except (AttributeError, RuntimeError, TypeError, ValueError):
        result["output_alias_ok"] = False
    result["finite_ok"] = bool(torch.isfinite(actual).all().item())
    if not all(
        result[key]
        for key in (
            "shape_ok",
            "dtype_ok",
            "device_ok",
            "layout_ok",
            "output_alias_ok",
            "finite_ok",
        )
    ):
        result["error"] = "output metadata or finiteness check failed"
        return result
    delta = (actual.float() - expected.float()).abs()
    denom = expected.float().abs().clamp_min(1e-12)
    result["maximum_absolute_error"] = float(delta.max().item()) if delta.numel() else 0.0
    result["maximum_relative_error"] = float((delta / denom).max().item()) if delta.numel() else 0.0
    rtol, atol = tolerances(spec, case.dtype)
    result["passed"] = bool(torch.allclose(actual, expected, rtol=rtol, atol=atol, equal_nan=False))
    if not result["passed"]:
        flat = delta.reshape(-1)
        index = int(flat.argmax().item())
        result["maximum_error_flat_index"] = index
        result["actual_at_maximum_error"] = float(actual.reshape(-1)[index].item())
        result["expected_at_maximum_error"] = float(expected.reshape(-1)[index].item())
    return result


def hidden_cases_from_template(
    task_root: Any,
    *,
    secret_seed: int,
    count_correctness: int = 20,
    count_benchmark: int = 6,
) -> tuple[CaseSpec, ...]:
    """Generate hidden cases from ranges without persisting the resulting shapes.

    The caller supplies a deployment secret. Neither that secret nor the returned
    cases may be placed in prompts or candidate working directories.
    """
    import hashlib
    import json
    import random
    from pathlib import Path

    root = Path(task_root)
    template = json.loads((root / "hidden_template.json").read_text(encoding="utf-8"))
    task_id = root.name
    dtypes = template["dtypes"]
    distributions = ("normal", "near_zero", "large", "zeros", "repeated")
    dims: Mapping[str, Sequence[int]] = template["dimensions"]

    def make(rng: random.Random, index: int, kind: str) -> CaseSpec:
        params = {name: rng.choice(tuple(int(v) for v in values)) for name, values in dims.items()}
        return CaseSpec(
            id=f"hidden_{kind}_{index:02d}",
            kind=kind,
            dtype=rng.choice(dtypes),
            params=params,
            distribution=rng.choice(distributions) if kind == "correctness" else "normal",
            seed=rng.randrange(1, 2**31),
            weight=1.0,
            address_offset=rng.choice((0, 1, 3, 7, 17)),
            noncontiguous=bool(template.get("allow_noncontiguous", False) and rng.randrange(4) == 0),
        )

    def suite(kind: str, count: int) -> list[CaseSpec]:
        # Per-kind derivation keeps each suite stable when a durable worker is
        # instructed with only {kind,count}. Benchmark generation therefore
        # does not depend on how many correctness cases another process made.
        derived = hashlib.sha256(
            f"{secret_seed}:{task_id}:hidden-v1:{kind}".encode()
        ).digest()
        rng = random.Random(int.from_bytes(derived[:8], "big"))
        return [make(rng, index, kind) for index in range(count)]

    return tuple(
        suite("correctness", count_correctness)
        + suite("benchmark", count_benchmark)
    )


def hidden_suite_commitment(
    cases: Sequence[CaseSpec],
    *,
    experiment_id: str,
    task_id: str,
    generator: str,
    kind: str,
) -> str:
    """Bind a hidden suite without persisting its seed, shapes, or dtypes.

    Controller and worker compute this independently so a mismatched deployment
    secret fails before candidate execution. Its preimage must be protected by
    a secret with at least 128 bits of entropy.
    """
    import hashlib
    import json

    for name, value in (
        ("experiment_id", experiment_id),
        ("task_id", task_id),
        ("generator", generator),
        ("kind", kind),
    ):
        if not value or not value.strip() or "\x00" in value:
            raise ValueError(f"{name} must be a non-empty, NUL-free string")
    if not cases:
        raise ValueError("hidden suite commitment requires at least one case")
    payload = {
        "domain": "ascend-kernel-lab:hidden-suite-commitment:v1",
        "experiment_id": experiment_id,
        "task_id": task_id,
        "generator": generator,
        "kind": kind,
        "count": len(cases),
        "cases": [case.to_dict() for case in cases],
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_hidden_seed(
    value: str | int,
    *,
    allow_insecure_for_testing: bool = False,
) -> int:
    """Parse a bounded high-entropy deployment seed without echoing it.

    Production seeds must have 128-256 significant bits. The explicit test
    escape hatch exists only for deterministic fake/offline suites.
    """
    if isinstance(value, bool):
        raise ValueError("hidden seed must be a base-10 positive integer")
    if isinstance(value, str):
        if not value or not value.isascii() or not value.isdecimal():
            raise ValueError("hidden seed must be a base-10 positive integer")
        parsed = int(value, 10)
    elif isinstance(value, int):
        parsed = value
    else:
        raise ValueError("hidden seed must be a base-10 positive integer")
    if parsed <= 0:
        raise ValueError("hidden seed must be positive")
    bits = parsed.bit_length()
    if bits > 256:
        raise ValueError("hidden seed must not exceed 256 bits")
    if bits < 128 and not allow_insecure_for_testing:
        raise ValueError("hidden seed must contain at least 128 significant bits")
    return parsed
