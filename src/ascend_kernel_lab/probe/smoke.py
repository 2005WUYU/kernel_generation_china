from __future__ import annotations

import argparse
import importlib.util
import json
import tempfile
import traceback
from pathlib import Path
from typing import Any

RESULT_PREFIX = "AKG_TRITON_PROBE_RESULT="

_ELEMENTWISE_SOURCE = r'''
import torch
import torch_npu
import triton
import triton.language as tl

@triton.jit
def _add(x, y, out, n: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    tl.store(out + offsets, tl.load(x + offsets, mask=mask) + tl.load(y + offsets, mask=mask), mask=mask)

def run(dtype, n):
    x = torch.randn((n,), device='npu:0', dtype=dtype)
    y = torch.randn((n,), device='npu:0', dtype=dtype)
    out = torch.empty_like(x)
    _add[(triton.cdiv(n, 256),)](x, y, out, n=n, BLOCK=256)
    torch.npu.synchronize()
    return bool(torch.allclose(out, x + y, rtol=0.01, atol=0.01))
'''

_REDUCTION_SOURCE = r'''
import torch
import torch_npu
import triton
import triton.language as tl

@triton.jit
def _reduce(x, out, N: tl.constexpr, USE_MAX_EXP: tl.constexpr):
    offsets = tl.arange(0, N)
    values = tl.load(x + offsets)
    if USE_MAX_EXP:
        maximum = tl.max(values, axis=0)
        result = tl.sum(tl.exp(values - maximum), axis=0)
    else:
        result = tl.sum(values, axis=0)
    tl.store(out, result)

def run(use_max_exp):
    x = torch.randn((128,), device='npu:0', dtype=torch.float32)
    out = torch.empty((1,), device='npu:0', dtype=torch.float32)
    _reduce[(1,)](x, out, N=128, USE_MAX_EXP=use_max_exp)
    torch.npu.synchronize()
    expected = torch.exp(x-x.max()).sum() if use_max_exp else x.sum()
    return bool(torch.allclose(out[0], expected, rtol=0.001, atol=0.001))
'''

_DOT_SOURCE = r'''
import torch
import torch_npu
import triton
import triton.language as tl

@triton.jit
def _dot(a, b, out, BLOCK: tl.constexpr):
    row = tl.arange(0, BLOCK)[:, None]
    col = tl.arange(0, BLOCK)[None, :]
    k = tl.arange(0, BLOCK)
    av = tl.load(a + row * BLOCK + k[None, :])
    bv = tl.load(b + k[:, None] * BLOCK + col)
    tl.store(out + row * BLOCK + col, tl.dot(av, bv))

def run():
    a = torch.randn((16,16), device='npu:0', dtype=torch.float16)
    b = torch.randn((16,16), device='npu:0', dtype=torch.float16)
    out = torch.empty((16,16), device='npu:0', dtype=torch.float16)
    _dot[(1,)](a,b,out,BLOCK=16)
    torch.npu.synchronize()
    return bool(torch.allclose(out, a@b, rtol=0.02, atol=0.02))
'''

_GRID_SOURCE = r'''
import torch
import torch_npu
import triton
import triton.language as tl

@triton.jit
def _copy2d(x, out, M: tl.constexpr, N: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offsets = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N
    tl.store(out + row*N + offsets, tl.load(x + row*N + offsets, mask=mask), mask=mask)

def run():
    x=torch.randn((17,129),device='npu:0',dtype=torch.float16); out=torch.empty_like(x)
    _copy2d[(17,triton.cdiv(129,64))](x,out,M=17,N=129,BLOCK=64)
    torch.npu.synchronize(); return bool(torch.equal(out,x))
'''


def _run_source(source: str, *args: Any) -> Any:
    with tempfile.TemporaryDirectory(prefix="ascend-kernel-probe-") as temporary:
        module_path = Path(temporary) / "feature_smoke.py"
        module_path.write_text(source, encoding="utf-8")
        spec = importlib.util.spec_from_file_location("_ascend_kernel_probe_feature", module_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"failed to load probe module from {module_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.run(*args)


def run_feature(feature: str) -> dict[str, Any]:
    if feature in {"vector_add", "masked_load_store", "fp16", "bfloat16", "fp32", "multiple_kernels"}:
        import torch  # type: ignore[import-not-found]

        dtype = {"bfloat16": torch.bfloat16, "fp32": torch.float32}.get(feature, torch.float16)
        n = 1003 if feature == "masked_load_store" else 1024
        correct = _run_source(_ELEMENTWISE_SOURCE, dtype, n)
        if feature == "multiple_kernels":
            correct = correct and _run_source(_ELEMENTWISE_SOURCE, dtype, n)
    elif feature in {"reduction_sum", "max_exp"}:
        correct = _run_source(_REDUCTION_SOURCE, feature == "max_exp")
    elif feature == "dot":
        correct = _run_source(_DOT_SOURCE)
    elif feature == "grid_2d":
        correct = _run_source(_GRID_SOURCE)
    else:
        raise ValueError(f"unknown feature {feature}")
    return {"feature": feature, "compile": True, "run": True, "correct": bool(correct), "error": None}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature", required=True)
    args = parser.parse_args()
    try:
        result = run_feature(args.feature)
    except Exception as exc:
        result = {
            "feature": args.feature,
            "compile": False,
            "run": False,
            "correct": False,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback_tail": traceback.format_exc().splitlines()[-20:],
        }
    # CANN may write log warnings to stdout without a newline during process
    # teardown.  A sentinel lets the parent recover this JSON from noisy text.
    print(RESULT_PREFIX + json.dumps(result, ensure_ascii=False), flush=True)
    return 0 if result["correct"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
