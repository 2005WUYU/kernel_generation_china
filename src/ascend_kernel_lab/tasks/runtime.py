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
    else:
        args = _catalog_inputs(_operation_number(spec), case, torch, device)
    return GeneratedInputs(args=args, metadata={"case_id": case.id, "seed": case.seed})


def _with_seed(case: CaseSpec, delta: int) -> CaseSpec:
    return CaseSpec(**{**case.__dict__, "seed": case.seed + delta})


def _operation_number(spec: TaskSpec) -> int:
    return int(str(spec.semantics["op"]).removeprefix("op_"))


def _other_sample(
    torch: Any,
    shape: Sequence[int],
    case: CaseSpec,
    device: str,
    seed_delta: int,
) -> Any:
    return _sample(torch, shape, _with_seed(case, seed_delta), device)


def _positive_sample(
    torch: Any,
    shape: Sequence[int],
    case: CaseSpec,
    device: str,
    seed_delta: int = 0,
) -> Any:
    selected = case if seed_delta == 0 else _with_seed(case, seed_delta)
    return _sample(torch, shape, selected, device).abs() + 0.5


def _boolean_sample(
    torch: Any,
    shape: Sequence[int],
    case: CaseSpec,
    device: str,
    seed_delta: int,
) -> Any:
    generator = torch.Generator(device=device)
    generator.manual_seed(case.seed + seed_delta)
    mask = torch.rand(tuple(shape), device=device, generator=generator) >= 0.35
    mask[..., 0] = True
    return mask


def _indices(
    torch: Any,
    shape: Sequence[int],
    upper: int,
    case: CaseSpec,
    device: str,
    seed_delta: int = 71,
) -> Any:
    generator = torch.Generator(device=device)
    generator.manual_seed(case.seed + seed_delta)
    return torch.randint(0, upper, tuple(shape), dtype=torch.int64, device=device, generator=generator)


def _catalog_inputs(
    operation: int,
    case: CaseSpec,
    torch: Any,
    device: str,
) -> tuple[Any, ...]:
    p = case.params
    if operation <= 16:
        shape = (p["n"],)
        x = _sample(torch, shape, case, device)
        if operation == 6:
            x = _positive_sample(torch, shape, case, device)
        elif operation == 7:
            x = x.clamp(-5, 5)
        elif operation == 8:
            x = _positive_sample(torch, shape, case, device)
        if operation in {1, 2, 3, 4, 16}:
            y = (
                _positive_sample(torch, shape, case, device, 17)
                if operation == 4
                else _other_sample(torch, shape, case, device, 17)
            )
            return x, y
        return (x,)

    if operation <= 24:
        matrix = (p["m"], p["n"])
        x = _sample(torch, matrix, case, device)
        if operation in {17, 18, 19}:
            return x, _other_sample(torch, (p["n"],), case, device, 17)
        if operation == 20:
            return (
                x,
                _other_sample(torch, (p["n"],), case, device, 17),
                _other_sample(torch, (p["n"],), case, device, 29),
            )
        if operation == 21:
            return (
                x,
                _other_sample(torch, matrix, case, device, 17),
                _boolean_sample(torch, matrix, case, device, 29),
            )
        if operation == 22:
            return (
                x,
                _boolean_sample(torch, matrix, case, device, 17),
                _other_sample(torch, (1,), case, device, 29),
            )
        if operation == 23:
            return x, _other_sample(torch, matrix, case, device, 17)
        return (
            x,
            _other_sample(torch, matrix, case, device, 17),
            _other_sample(torch, matrix, case, device, 29),
            _other_sample(torch, (1,), case, device, 43),
        )

    if operation <= 40:
        shape = (p["m"], p["n"])
        x = _sample(torch, shape, case, device)
        if operation in {38, 40}:
            return x, _boolean_sample(torch, shape, case, device, 17)
        return (x,)

    if operation <= 48:
        shape = (p["m"], p["n"])
        x = _sample(torch, shape, case, device)
        vector = (p["n"],)
        if operation == 41:
            return x, _other_sample(torch, vector, case, device, 17)
        if operation in {42, 43}:
            return (
                x,
                _other_sample(torch, vector, case, device, 17),
                _other_sample(torch, vector, case, device, 29),
            )
        if operation == 44:
            return (x,)
        if operation == 45:
            return (
                x,
                _other_sample(torch, vector, case, device, 17),
                _other_sample(torch, vector, case, device, 29),
                _other_sample(torch, vector, case, device, 43),
                _positive_sample(torch, vector, case, device, 59),
            )
        if operation == 46:
            return (
                x,
                _other_sample(torch, shape, case, device, 17),
                _other_sample(torch, vector, case, device, 29),
            )
        if operation == 47:
            return (
                x,
                _other_sample(torch, shape, case, device, 17),
                _other_sample(torch, vector, case, device, 29),
                _other_sample(torch, vector, case, device, 43),
            )
        return (
            x,
            _other_sample(torch, vector, case, device, 17),
            _other_sample(torch, vector, case, device, 29),
        )

    if operation <= 56:
        if operation == 50:
            return (_sample(torch, (p["batch"], p["m"], p["n"]), case, device),)
        if operation == 51:
            backing = _sample(torch, (p["m"], p["n"] * 2), case, device)
            return (backing[..., ::2],)
        if operation == 52:
            base = _sample(torch, (p["m"], p["n"]), case, device)
            return (base.transpose(-2, -1),)
        shape = (p["m"], p["n"])
        x = _sample(torch, shape, case, device)
        if operation == 54:
            return x, _other_sample(torch, shape, case, device, 17)
        return (x,)

    if operation <= 64:
        m, n, count = p["m"], p["n"], p["count"]
        if operation == 57:
            return _sample(torch, (m, n), case, device), _indices(torch, (count,), m, case, device)
        if operation in {58, 59}:
            return (
                _sample(torch, (m, n), case, device),
                _indices(torch, (m, count), n, case, device),
            )
        if operation in {60, 61}:
            return (
                _sample(torch, (m, n), case, device),
                _indices(torch, (m, count), n, case, device),
                _other_sample(torch, (m, count), case, device, 17),
            )
        if operation == 62:
            return (
                _sample(torch, (m, n), case, device),
                _indices(torch, (count,), m, case, device),
                _other_sample(torch, (count, n), case, device, 17),
            )
        return (
            _sample(torch, (p["vocab"], p["dim"]), case, device),
            _indices(torch, (count,), p["vocab"], case, device),
        )

    if operation <= 72:
        return (_sample(torch, (p["m"], p["n"]), case, device),)

    if operation <= 80:
        if operation == 80:
            return (
                _sample(torch, (p["batch"], p["m"], p["k"]), case, device),
                _other_sample(torch, (p["batch"], p["k"], p["n"]), case, device, 17),
            )
        a_shape = (p["k"], p["m"]) if operation in {75, 76} else (p["m"], p["k"])
        b_shape = (p["n"], p["k"]) if operation in {74, 76} else (p["k"], p["n"])
        return (
            _sample(torch, a_shape, case, device),
            _other_sample(torch, b_shape, case, device, 17),
        )

    if operation <= 88:
        a = _sample(torch, (p["m"], p["k"]), case, device)
        b = _other_sample(torch, (p["k"], p["n"]), case, device, 17)
        args = (a, b)
        if operation in {81, 85, 86, 88}:
            args += (_other_sample(torch, (p["n"],), case, device, 29),)
        if operation in {87, 88}:
            args += (_other_sample(torch, (p["m"], p["n"]), case, device, 43),)
        return args

    if operation <= 90:
        shape = (p["m"], p["n"])
        return (
            _sample(torch, shape, case, device),
            _other_sample(torch, shape, case, device, 17),
        )

    if operation <= 96:
        if operation in {91, 92}:
            return (_sample(torch, (p["seq"], p["heads"], p["dim"]), case, device),)
        if operation in {93, 94}:
            return (
                _sample(
                    torch,
                    (p["batch"], p["seq"], 3 * p["heads"] * p["dim"]),
                    case,
                    device,
                ),
            )
        cache = _sample(torch, (p["cache"], p["heads"], p["dim"]), case, device)
        indices = torch.arange(p["seq"], dtype=torch.int64, device=device) % p["cache"]
        if operation == 95:
            value = _other_sample(
                torch, (p["seq"], p["heads"], p["dim"]), case, device, 17
            )
            return cache, value, indices
        return cache, indices

    if operation <= 104:
        q_shape = (p["batch"], p["heads"], p["q"], p["dim"])
        if operation == 101:
            probability = _positive_sample(
                torch, (p["batch"], p["heads"], p["q"], p["kv"]), case, device
            )
            probability = probability / probability.float().sum(dim=-1, keepdim=True).to(probability.dtype)
            value = _other_sample(
                torch, (p["batch"], p["heads"], p["kv"], p["dim"]), case, device, 17
            )
            return probability, value
        q = _sample(torch, q_shape, case, device)
        kv_heads = max(1, p["heads"] // 4) if operation == 104 else p["heads"]
        kv_shape = (p["batch"], kv_heads, p["kv"], p["dim"])
        k = _other_sample(torch, kv_shape, case, device, 17)
        if operation in {97, 98, 100}:
            return q, k
        if operation == 99:
            return (
                q,
                k,
                _other_sample(
                    torch,
                    (p["batch"], p["heads"], p["q"], p["kv"]),
                    case,
                    device,
                    29,
                ),
            )
        return q, k, _other_sample(torch, kv_shape, case, device, 29)

    if operation == 105:
        return (_sample(torch, (p["tokens"], p["experts"]), case, device),)
    if operation in {106, 107}:
        return (_sample(torch, (p["tokens"], p["experts"]), case, device),)
    if operation == 108:
        return (_indices(torch, (p["tokens"], p["topk"]), p["experts"], case, device),)
    if operation == 109:
        return (
            _sample(torch, (p["tokens"], p["hidden"]), case, device),
            _indices(torch, (p["tokens"], p["topk"]), p["experts"], case, device),
        )
    if operation == 110:
        routes = p["tokens"] * p["topk"]
        generator = torch.Generator(device=device)
        generator.manual_seed(case.seed + 71)
        return (
            _sample(torch, (routes, p["hidden"]), case, device),
            torch.randperm(routes, dtype=torch.int64, device=device, generator=generator),
        )
    if operation == 111:
        outputs = _sample(torch, (p["tokens"], p["topk"], p["hidden"]), case, device)
        weights = _positive_sample(torch, (p["tokens"], p["topk"]), case, device, 17)
        weights = weights / weights.float().sum(dim=-1, keepdim=True).to(weights.dtype)
        return outputs, weights
    return (
        _sample(torch, (p["tokens"], p["experts"]), case, device),
        _other_sample(torch, (p["tokens"], p["hidden"]), case, device, 17),
    )


def reference(
    spec: TaskSpec,
    args: Sequence[Any],
    torch: Any,
    *,
    case: CaseSpec | None = None,
) -> Any:
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
    return _catalog_reference(
        _operation_number(spec), spec, args, torch, case=case
    )


def _to_input_dtype(value: Any, source: Any) -> Any:
    return value.to(source.dtype)


def _rms_norm(torch: Any, value: Any, weight: Any) -> Any:
    x = value.float()
    normalized = x * torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + 1e-5)
    return (normalized * weight.float()).to(value.dtype)


def _rotary(torch: Any, value: Any, *, interleaved: bool) -> Any:
    dim = value.shape[-1]
    seq = value.shape[-3]
    positions = torch.arange(seq, device=value.device, dtype=torch.float32)
    frequencies = 1.0 / (
        10000
        ** (torch.arange(0, dim, 2, device=value.device, dtype=torch.float32) / dim)
    )
    angles = positions[:, None] * frequencies[None, :]
    shape = [1] * value.ndim
    shape[-3] = seq
    shape[-1] = dim // 2
    cosine = torch.cos(angles).reshape(shape)
    sine = torch.sin(angles).reshape(shape)
    source = value.float()
    output = torch.empty_like(source)
    if interleaved:
        first, second = source[..., 0::2], source[..., 1::2]
        output[..., 0::2] = first * cosine - second * sine
        output[..., 1::2] = first * sine + second * cosine
    else:
        first, second = source[..., : dim // 2], source[..., dim // 2 :]
        output[..., : dim // 2] = first * cosine - second * sine
        output[..., dim // 2 :] = first * sine + second * cosine
    return output.to(value.dtype)


def _attention_scores(torch: Any, q: Any, k: Any) -> Any:
    return torch.matmul(q.float(), k.float().transpose(-2, -1))


def _causal_mask(torch: Any, rows: int, columns: int, device: Any) -> Any:
    row = torch.arange(rows, device=device)[:, None]
    column = torch.arange(columns, device=device)[None, :]
    return column <= row + max(0, columns - rows)


def _catalog_reference(
    operation: int,
    spec: TaskSpec,
    args: Sequence[Any],
    torch: Any,
    *,
    case: CaseSpec | None,
) -> Any:
    x = args[0]
    if operation == 1:
        return x + args[1]
    if operation == 2:
        return x - args[1]
    if operation == 3:
        return x * args[1]
    if operation == 4:
        return x / args[1]
    if operation == 5:
        return x * x
    if operation == 6:
        return torch.reciprocal(x)
    if operation == 7:
        return torch.exp(x)
    if operation == 8:
        return torch.log(x)
    if operation == 9:
        return torch.relu(x)
    if operation == 10:
        return torch.sigmoid(x)
    if operation == 11:
        return torch.tanh(x)
    if operation == 12:
        return torch.nn.functional.gelu(x, approximate="tanh")
    if operation == 13:
        return torch.nn.functional.silu(x)
    if operation == 14:
        return torch.nn.functional.leaky_relu(x, negative_slope=0.01)
    if operation == 15:
        return torch.clamp(x, -1.0, 1.0)
    if operation == 16:
        return torch.relu(x + args[1])
    if operation == 17:
        return x + args[1]
    if operation == 18:
        return torch.nn.functional.gelu(x + args[1], approximate="tanh")
    if operation == 19:
        return torch.nn.functional.silu(x + args[1])
    if operation == 20:
        return x * args[1] + args[2]
    if operation == 21:
        return torch.where(args[2], x, args[1])
    if operation == 22:
        return torch.where(args[1], args[2][0], x)
    if operation == 23:
        return torch.maximum(x, args[1])
    if operation == 24:
        return x + args[3][0] * args[1] * args[2]
    if operation == 25:
        return x.float().sum(dim=-1).to(x.dtype)
    if operation == 26:
        return x.float().sum(dim=0).to(x.dtype)
    if operation == 27:
        return x.float().mean(dim=-1).to(x.dtype)
    if operation == 28:
        return x.max(dim=-1).values
    if operation == 29:
        return x.min(dim=-1).values
    if operation == 30:
        return x.argmax(dim=-1)
    if operation == 31:
        return x.argmin(dim=-1)
    if operation == 32:
        return torch.linalg.vector_norm(x.float(), dim=-1).to(x.dtype)
    if operation == 33:
        return x.float().var(dim=-1, correction=0).to(x.dtype)
    if operation == 34:
        return torch.logsumexp(x.float(), dim=-1).to(x.dtype)
    if operation == 35:
        return torch.softmax(x.float(), dim=-1).to(x.dtype)
    if operation == 36:
        return torch.log_softmax(x.float(), dim=-1).to(x.dtype)
    if operation == 37:
        return torch.softmax(x.float() / (x.shape[-1] ** 0.5), dim=-1).to(x.dtype)
    if operation == 38:
        masked = x.float().masked_fill(~args[1], float("-inf"))
        return torch.softmax(masked, dim=-1).to(x.dtype)
    if operation == 39:
        mask = _causal_mask(torch, x.shape[-2], x.shape[-1], x.device)
        return torch.softmax(x.float().masked_fill(~mask, float("-inf")), dim=-1).to(x.dtype)
    if operation == 40:
        scaled = x.float() / (x.shape[-1] ** 0.5)
        return torch.softmax(scaled.masked_fill(~args[1], float("-inf")), dim=-1).to(x.dtype)
    if operation == 41:
        return _rms_norm(torch, x, args[1])
    if operation == 42:
        return torch.nn.functional.layer_norm(
            x, (x.shape[-1],), args[1], args[2], 1e-5
        )
    if operation == 43:
        groups = _case_parameter(
            spec, "groups", case=case, m=x.shape[0], n=x.shape[1]
        )
        width = x.shape[-1] // groups
        grouped = x.float().reshape(x.shape[0], groups, width)
        mean = grouped.mean(dim=-1, keepdim=True)
        variance = grouped.var(dim=-1, correction=0, keepdim=True)
        normalized = ((grouped - mean) * torch.rsqrt(variance + 1e-5)).reshape_as(x)
        return (normalized * args[1].float() + args[2].float()).to(x.dtype)
    if operation == 44:
        norm = torch.linalg.vector_norm(x.float(), dim=-1, keepdim=True)
        return (x.float() / norm.clamp_min(1e-12)).to(x.dtype)
    if operation == 45:
        normalized = (x.float() - args[3].float()) * torch.rsqrt(args[4].float() + 1e-5)
        return (normalized * args[1].float() + args[2].float()).to(x.dtype)
    if operation == 46:
        return _rms_norm(torch, x + args[1], args[2])
    if operation == 47:
        residual = x + args[1]
        return torch.nn.functional.layer_norm(
            residual, (residual.shape[-1],), args[2], args[3], 1e-5
        )
    if operation == 48:
        return (_rms_norm(torch, x, args[1]).float() * args[2].float()).to(x.dtype)
    if operation in {49, 50}:
        return x.transpose(-2, -1).contiguous()
    if operation in {51, 52}:
        return x.contiguous()
    if operation == 53:
        return x[:, ::2].contiguous()
    if operation == 54:
        return torch.cat((x, args[1]), dim=-1)
    if operation == 55:
        return torch.nn.functional.pad(x, (1, 1))
    if operation == 56:
        return torch.flip(x, dims=(-1,))
    if operation == 57:
        return torch.index_select(x, 0, args[1])
    if operation in {58, 59}:
        return torch.gather(x, -1, args[1])
    if operation == 60:
        return x.clone().scatter(-1, args[1], args[2])
    if operation == 61:
        return x.clone().scatter_add(-1, args[1], args[2])
    if operation == 62:
        return x.clone().index_add(0, args[1], args[2])
    if operation == 63:
        return torch.index_select(x, 0, args[1])
    if operation == 64:
        bags = _case_parameter(
            spec,
            "bags",
            case=case,
            vocab=x.shape[0],
            dim=x.shape[1],
            count=args[1].numel(),
        )
        chunks = torch.tensor_split(args[1], bags)
        return torch.stack(
            [torch.index_select(x, 0, chunk).float().sum(dim=0) for chunk in chunks]
        ).to(x.dtype)
    if operation == 65:
        return torch.cumsum(x.float(), dim=-1).to(x.dtype)
    if operation == 66:
        return torch.cummax(x, dim=-1).values
    if operation == 67:
        inclusive = torch.cumsum(x.float(), dim=-1)
        return torch.cat((torch.zeros_like(inclusive[..., :1]), inclusive[..., :-1]), dim=-1).to(x.dtype)
    if operation == 68:
        return torch.sort(x, dim=-1).values
    if operation == 69:
        return torch.argsort(x, dim=-1)
    if operation == 70:
        k = _case_parameter(
            spec, "k", case=case, m=x.shape[0], n=x.shape[1]
        )
        return torch.topk(x, k, dim=-1, sorted=True).values
    if operation == 71:
        k = _case_parameter(
            spec, "k", case=case, m=x.shape[0], n=x.shape[1]
        )
        values, indices = torch.topk(x, k, dim=-1, sorted=True)
        return values, indices
    if operation == 72:
        k = _case_parameter(
            spec, "k", case=case, m=x.shape[0], n=x.shape[1]
        )
        values, indices = torch.topk(x, k, dim=-1, sorted=True)
        return torch.softmax(values.float(), dim=-1).to(x.dtype), indices
    if operation == 73 or 77 <= operation <= 79:
        return torch.matmul(x.float(), args[1].float()).to(x.dtype)
    if operation == 74:
        return torch.matmul(x.float(), args[1].float().transpose(-2, -1)).to(x.dtype)
    if operation == 75:
        return torch.matmul(x.float().transpose(-2, -1), args[1].float()).to(x.dtype)
    if operation == 76:
        return torch.matmul(
            x.float().transpose(-2, -1), args[1].float().transpose(-2, -1)
        ).to(x.dtype)
    if operation == 80:
        return torch.matmul(x.float(), args[1].float()).to(x.dtype)
    if 81 <= operation <= 88:
        product = torch.matmul(x.float(), args[1].float())
        cursor = 2
        if operation in {81, 85, 86, 88}:
            product = product + args[cursor].float()
            cursor += 1
        if operation in {87, 88}:
            product = product + args[cursor].float()
        if operation == 82:
            product = torch.relu(product)
        elif operation in {83, 85}:
            product = torch.nn.functional.gelu(product, approximate="tanh")
        elif operation in {84, 86}:
            product = torch.nn.functional.silu(product)
        return product.to(x.dtype)
    if operation == 89:
        return (torch.nn.functional.silu(args[1].float()) * x.float()).to(x.dtype)
    if operation == 90:
        gate = torch.nn.functional.gelu(args[1].float(), approximate="tanh")
        return (gate * x.float()).to(x.dtype)
    if operation == 91:
        return _rotary(torch, x, interleaved=False)
    if operation == 92:
        return _rotary(torch, x, interleaved=True)
    if operation in {93, 94}:
        batch, seq, packed = x.shape
        heads = _case_parameter(
            spec, "heads", case=case, batch=batch, seq=seq
        )
        dim = _case_parameter(
            spec, "dim", case=case, batch=batch, seq=seq, heads=heads
        )
        q, k, v = x.reshape(batch, seq, 3, heads, dim).unbind(dim=2)
        if operation == 94:
            q = _rotary(torch, q, interleaved=True)
            k = _rotary(torch, k, interleaved=True)
        else:
            q, k = q.contiguous(), k.contiguous()
        return q, k, v.contiguous()
    if operation == 95:
        output = x.clone()
        output.index_copy_(0, args[2], args[1])
        return output
    if operation == 96:
        return torch.index_select(x, 0, args[1])
    if operation == 97:
        return _attention_scores(torch, x, args[1]).to(x.dtype)
    if operation == 98:
        scores = _attention_scores(torch, x, args[1]) / (x.shape[-1] ** 0.5)
        return scores.to(x.dtype)
    if operation == 99:
        return (_attention_scores(torch, x, args[1]) + args[2].float()).to(x.dtype)
    if operation == 100:
        scores = _attention_scores(torch, x, args[1])
        mask = _causal_mask(torch, scores.shape[-2], scores.shape[-1], scores.device)
        return scores.masked_fill(~mask, float("-inf")).to(x.dtype)
    if operation == 101:
        return torch.matmul(x.float(), args[1].float()).to(x.dtype)
    if operation in {102, 103, 104}:
        key, value = args[1], args[2]
        if operation == 104:
            repeats = x.shape[1] // key.shape[1]
            key = key.repeat_interleave(repeats, dim=1)
            value = value.repeat_interleave(repeats, dim=1)
        scores = _attention_scores(torch, x, key) / (x.shape[-1] ** 0.5)
        if operation == 103:
            mask = _causal_mask(torch, scores.shape[-2], scores.shape[-1], scores.device)
            scores = scores.masked_fill(~mask, float("-inf"))
        probability = torch.softmax(scores, dim=-1)
        return torch.matmul(probability, value.float()).to(x.dtype)
    if operation == 105:
        return torch.softmax(x.float(), dim=-1).to(x.dtype)
    if operation == 106:
        topk = _case_parameter(
            spec, "topk", case=case, tokens=x.shape[0], experts=x.shape[1]
        )
        return torch.topk(x, topk, dim=-1, sorted=True)
    if operation == 107:
        topk = _case_parameter(
            spec, "topk", case=case, tokens=x.shape[0], experts=x.shape[1]
        )
        values, indices = torch.topk(x, topk, dim=-1, sorted=True)
        return torch.softmax(values.float(), dim=-1).to(x.dtype), indices
    if operation == 108:
        experts = _case_parameter(
            spec, "experts", case=case, tokens=x.shape[0], topk=x.shape[1]
        )
        return torch.bincount(x.reshape(-1), minlength=experts)
    if operation == 109:
        flattened = args[1].reshape(-1)
        order = torch.argsort(flattened, stable=True)
        expanded = x.repeat_interleave(args[1].shape[1], dim=0)
        return torch.index_select(expanded, 0, order)
    if operation == 110:
        return torch.index_select(x, 0, args[1])
    if operation == 111:
        return (x.float() * args[1].float().unsqueeze(-1)).sum(dim=1).to(x.dtype)
    topk = _case_parameter(
        spec, "topk", case=case, tokens=x.shape[0], experts=x.shape[1]
    )
    values, indices = torch.topk(x, topk, dim=-1, sorted=True)
    del values
    flattened = indices.reshape(-1)
    order = torch.argsort(flattened, stable=True)
    expanded = args[1].repeat_interleave(indices.shape[1], dim=0)
    return torch.index_select(expanded, 0, order), torch.index_select(flattened, 0, order)


def _case_parameter(
    spec: TaskSpec,
    target: str,
    *,
    case: CaseSpec | None = None,
    **known: int,
) -> int:
    if case is not None and target in case.params:
        return int(case.params[target])
    for case in spec.public_cases:
        params = case.params
        if target in params and all(params.get(name) == value for name, value in known.items()):
            return int(params[target])
    raise ValueError(f"no declared {spec.id} case matches {known}")


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
    expected_outputs = tuple(expected) if isinstance(expected, (tuple, list)) else (expected,)
    if len(expected_outputs) == 1 and isinstance(actual, torch.Tensor):
        actual_outputs = (actual,)
    elif isinstance(actual, (tuple, list)):
        actual_outputs = tuple(actual)
    else:
        result["error"] = "custom_op returned an invalid output structure"
        return result
    if len(actual_outputs) != len(expected_outputs):
        result["error"] = "custom_op returned the wrong number of outputs"
        return result

    input_storages = {
        int(value.untyped_storage().data_ptr())
        for value in inputs
        if isinstance(value, torch.Tensor)
    }
    rtol, atol = tolerances(spec, case.dtype)
    output_results: list[dict[str, Any]] = []
    for output_index, (actual_output, expected_output) in enumerate(
        zip(actual_outputs, expected_outputs, strict=True)
    ):
        item: dict[str, Any] = {
            "output_index": output_index,
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
        if not isinstance(actual_output, torch.Tensor) or not isinstance(
            expected_output, torch.Tensor
        ):
            item["error"] = "output is not a Tensor"
            output_results.append(item)
            continue
        item["shape_ok"] = tuple(actual_output.shape) == tuple(expected_output.shape)
        item["dtype_ok"] = actual_output.dtype == expected_output.dtype
        item["device_ok"] = actual_output.device == expected_output.device
        item["layout_ok"] = bool(actual_output.is_contiguous())
        item["output_alias_ok"] = (
            int(actual_output.untyped_storage().data_ptr()) not in input_storages
        )
        item["finite_ok"] = bool(
            torch.equal(torch.isfinite(actual_output), torch.isfinite(expected_output))
        )
        metadata_ok = all(
            item[key]
            for key in (
                "shape_ok",
                "dtype_ok",
                "device_ok",
                "layout_ok",
                "output_alias_ok",
                "finite_ok",
            )
        )
        if not metadata_ok:
            item["error"] = "output metadata or finiteness check failed"
            output_results.append(item)
            continue
        finite = torch.isfinite(expected_output)
        actual_finite = actual_output[finite].float()
        expected_finite = expected_output[finite].float()
        delta = (actual_finite - expected_finite).abs()
        denom = expected_finite.abs().clamp_min(1e-12)
        item["maximum_absolute_error"] = float(delta.max().item()) if delta.numel() else 0.0
        item["maximum_relative_error"] = (
            float((delta / denom).max().item()) if delta.numel() else 0.0
        )
        if actual_output.is_floating_point() or actual_output.is_complex():
            item["passed"] = bool(
                torch.allclose(
                    actual_output,
                    expected_output,
                    rtol=rtol,
                    atol=atol,
                    equal_nan=False,
                )
            )
        else:
            item["passed"] = bool(torch.equal(actual_output, expected_output))
        if not item["passed"] and delta.numel():
            index = int(delta.argmax().item())
            item["maximum_error_flat_index"] = index
            item["actual_at_maximum_error"] = float(actual_finite[index].item())
            item["expected_at_maximum_error"] = float(expected_finite[index].item())
        output_results.append(item)

    result["output_results"] = output_results
    for key in (
        "shape_ok",
        "dtype_ok",
        "device_ok",
        "layout_ok",
        "output_alias_ok",
        "finite_ok",
    ):
        result[key] = all(bool(item[key]) for item in output_results)
    absolute = [
        float(item["maximum_absolute_error"])
        for item in output_results
        if item.get("maximum_absolute_error") is not None
    ]
    relative = [
        float(item["maximum_relative_error"])
        for item in output_results
        if item.get("maximum_relative_error") is not None
    ]
    result["maximum_absolute_error"] = max(absolute, default=None)
    result["maximum_relative_error"] = max(relative, default=None)
    result["passed"] = all(bool(item["passed"]) for item in output_results)
    if not result["passed"]:
        result["error"] = "one or more outputs failed validation"
    return result


def hidden_cases_from_template(
    task: TaskSpec | Any,
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

    catalog_profiles: tuple[Mapping[str, int], ...] = ()
    if isinstance(task, TaskSpec):
        if task.root is None:
            raise ValueError("hidden cases require a registry-backed task")
        root = task.root
        task_id = task.id
        template_path = root / "hidden_template.json"
        if template_path.is_file():
            template = json.loads(template_path.read_text(encoding="utf-8"))
        else:
            catalog_profiles = tuple(
                dict(case.params) for case in task.public_cases if case.kind == "correctness"
            )
            template = {
                "dtypes": sorted({case.dtype for case in task.correctness_cases}),
                "allow_noncontiguous": False,
            }
    else:
        root = Path(task)
        task_id = root.name
        template = json.loads((root / "hidden_template.json").read_text(encoding="utf-8"))
    dtypes = template["dtypes"]
    distributions = ("normal", "near_zero", "large", "zeros", "repeated")
    dims: Mapping[str, Sequence[int]] = template.get("dimensions", {})

    def make(rng: random.Random, index: int, kind: str) -> CaseSpec:
        params = (
            dict(rng.choice(catalog_profiles))
            if catalog_profiles
            else {
                name: rng.choice(tuple(int(v) for v in values))
                for name, values in dims.items()
            }
        )
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
