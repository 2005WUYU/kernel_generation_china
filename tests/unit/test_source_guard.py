from __future__ import annotations

import unittest

from ascend_kernel_lab.evaluation.source_guard import SourceGuard

VALID = """
import torch
import triton
import triton.language as tl

@triton.jit
def generated_kernel(x, out, n: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    tl.store(out + offsets, tl.load(x + offsets, mask=mask), mask=mask)

def custom_op(x: torch.Tensor) -> torch.Tensor:
    output = torch.empty_like(x)
    generated_kernel[(triton.cdiv(x.numel(), 256),)](x, output, x.numel(), n=x.numel(), BLOCK=256)
    return output
"""


class SourceGuardTests(unittest.TestCase):
    def test_accepts_complete_triton_candidate(self) -> None:
        result = SourceGuard().check(VALID)
        self.assertTrue(result.passed, result.findings)
        self.assertTrue(result.entry_point_found)
        self.assertTrue(result.triton_kernel_found)
        self.assertTrue(result.kernel_launch_found)

    def test_accepts_allocator_with_tensor_metadata_and_scalar_grid_math(self) -> None:
        source = VALID.replace(
            "output = torch.empty_like(x)",
            "n = x.shape[0]\n"
            "    output = torch.empty((n,), device=x.device, dtype=x.dtype)",
        ).replace(
            "generated_kernel[(triton.cdiv(x.numel(), 256),)]"
            "(x, output, x.numel(), n=x.numel(), BLOCK=256)",
            "generated_kernel[(triton.cdiv(n, 256),)]"
            "(x, output, n, n=n, BLOCK=256)",
        )
        result = SourceGuard().check(source)
        self.assertTrue(result.passed, result.findings)

    def test_rejects_high_level_torch_via_alias(self) -> None:
        source = VALID.replace("import torch", "import torch as t").replace(
            "torch.empty_like(x)", "t.softmax(x, dim=-1)"
        ).replace("torch.Tensor", "t.Tensor")
        result = SourceGuard().check(source)
        self.assertFalse(result.passed)
        self.assertIn("forbidden_call", {finding.code for finding in result.findings})

    def test_rejects_dynamic_reflection_and_dunder(self) -> None:
        source = VALID.replace(
            "output = torch.empty_like(x)",
            "output = getattr(torch.__dict__, '__getitem__')('empty_like')(x)",
        )
        result = SourceGuard().check(source)
        codes = {finding.code for finding in result.findings}
        self.assertIn("forbidden_builtin", codes)
        self.assertIn("forbidden_attribute", codes)

    def test_rejects_top_level_side_effect(self) -> None:
        result = SourceGuard().check(VALID + "\ncustom_op(None)\n")
        self.assertIn("top_level_side_effect", {finding.code for finding in result.findings})

    def test_rejects_cpu_and_tensor_reduction(self) -> None:
        source = VALID.replace("output = torch.empty_like(x)", "output = x.cpu().sum()")
        result = SourceGuard().check(source)
        attributes = [finding.message for finding in result.findings if finding.code == "forbidden_attribute"]
        self.assertTrue(any("cpu" in message for message in attributes))
        self.assertTrue(any("sum" in message for message in attributes))

    def test_rejects_dummy_launch_with_high_level_tensor_return(self) -> None:
        source = VALID.replace(
            "return output",
            "return x.transpose(0, 0).contiguous()",
        )
        result = SourceGuard().check(source)
        codes = {finding.code for finding in result.findings}
        self.assertFalse(result.passed)
        self.assertIn("forbidden_tensor_method", codes)
        self.assertIn("invalid_host_return", codes)

    def test_rejects_tensor_view_copy_and_indexing_paths(self) -> None:
        replacements = (
            "x.clone()",
            "x.reshape(-1)",
            "x.view(-1)",
            "x[0]",
        )
        for expression in replacements:
            with self.subTest(expression=expression):
                result = SourceGuard().check(
                    VALID.replace("return output", f"return {expression}")
                )
                self.assertFalse(result.passed)
                self.assertTrue(
                    {
                        "forbidden_tensor_method",
                        "forbidden_tensor_indexing",
                    }
                    & {finding.code for finding in result.findings}
                )

    def test_rejects_returned_input_even_after_real_kernel_launch(self) -> None:
        result = SourceGuard().check(VALID.replace("return output", "return x"))
        self.assertFalse(result.passed)
        self.assertIn(
            "invalid_host_return", {finding.code for finding in result.findings}
        )

    def test_rejects_kernel_launch_that_never_receives_returned_output(self) -> None:
        source = VALID.replace(
            "generated_kernel[(triton.cdiv(x.numel(), 256),)]"
            "(x, output, x.numel(), n=x.numel(), BLOCK=256)",
            "generated_kernel[(triton.cdiv(x.numel(), 256),)]"
            "(x, x, x.numel(), n=x.numel(), BLOCK=256)",
        )
        result = SourceGuard().check(source)
        codes = {finding.code for finding in result.findings}
        self.assertIn("kernel_launch_without_output", codes)
        self.assertIn("unlaunched_host_output", codes)

    def test_rejects_broad_kernel_identifier(self) -> None:
        for name in ("k", "_", "kernel"):
            with self.subTest(name=name):
                source = VALID.replace("generated_kernel", name)
                result = SourceGuard().check(source)
                self.assertFalse(result.passed)
                self.assertIn(
                    "unsafe_kernel_name", {item.code for item in result.findings}
                )

    def test_requires_a_return_on_every_entrypoint_path(self) -> None:
        result = SourceGuard().check(VALID.replace("    return output\n", ""))
        self.assertFalse(result.passed)
        self.assertIn("missing_host_return", {item.code for item in result.findings})

    def test_rejects_host_triton_runtime_shortcuts(self) -> None:
        source = VALID.replace(
            "output = torch.empty_like(x)",
            "output = triton.testing.do_bench(lambda: x)",
        )
        result = SourceGuard().check(source)
        self.assertIn(
            "forbidden_triton_runtime_call",
            {finding.code for finding in result.findings},
        )

    def test_accepts_bounded_literal_autotune_configs(self) -> None:
        source = VALID.replace(
            "@triton.jit",
            "@triton.autotune("
            "configs=[triton.Config({}, num_warps=4)], key=['n'])\n"
            "@triton.jit",
        )
        result = SourceGuard().check(source)
        self.assertTrue(result.passed, result.findings)

    def test_rejects_dynamic_or_excessive_autotune_configs(self) -> None:
        dynamic = VALID.replace(
            "@triton.jit",
            "@triton.autotune(configs=make_configs(), key=['n'])\n@triton.jit",
        )
        result = SourceGuard().check(dynamic)
        self.assertIn(
            "dynamic_autotune_configs",
            {finding.code for finding in result.findings},
        )
        configs = ",".join("triton.Config({}, num_warps=1)" for _ in range(33))
        excessive = VALID.replace(
            "@triton.jit",
            f"@triton.autotune(configs=[{configs}], key=['n'])\n@triton.jit",
        )
        result = SourceGuard().check(excessive)
        self.assertIn(
            "too_many_autotune_configs",
            {finding.code for finding in result.findings},
        )


if __name__ == "__main__":
    unittest.main()
