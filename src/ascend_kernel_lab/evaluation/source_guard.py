from __future__ import annotations

import ast
import hashlib
import re
from collections import deque
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class GuardFinding:
    code: str
    message: str
    line: int | None = None
    column: int | None = None


@dataclass(frozen=True)
class SourceGuardResult:
    passed: bool
    syntax_ok: bool
    entry_point_found: bool
    triton_kernel_found: bool
    kernel_launch_found: bool
    source_sha256: str
    source_bytes: int
    findings: tuple[GuardFinding, ...]
    warnings: tuple[GuardFinding, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def is_safe_kernel_identifier(name: str) -> bool:
    """Whether a source-level kernel name can be matched exactly."""

    return len(name) <= 128 and name.isidentifier()


def candidate_kernel_pattern(name: str) -> str:
    """Return the exact profiler identity for one trusted AST kernel name."""

    if not is_safe_kernel_identifier(name):
        raise ValueError("kernel name is not a valid bounded Python identifier")
    return rf"^{re.escape(name)}(?:_[0-9a-fA-F]{{8,64}})?$"


@dataclass
class _FunctionFacts:
    calls: list[ast.Call]
    launches: list[ast.Call]
    returns: set[str]


class _BodyFacts(ast.NodeVisitor):
    """Collect shallow facts while leaving Python authoring style unrestricted."""

    def __init__(self, guard: SourceGuard) -> None:
        self.guard = guard
        self.calls: list[ast.Call] = []
        self.launches: list[ast.Call] = []
        self.returns: set[str] = set()

    def facts(self, function: ast.FunctionDef) -> _FunctionFacts:
        for statement in function.body:
            self.visit(statement)
        return _FunctionFacts(
            self.calls,
            self.launches,
            self.returns,
        )

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        # A nested function is considered only when called by name, not merely
        # because its definition is present in an outer function body.
        del node

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        del node

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        del node

    def visit_Return(self, node: ast.Return) -> None:
        if node.value is not None:
            self.returns.update(_loaded_names(node.value))
            self.visit(node.value)

    def visit_Call(self, node: ast.Call) -> None:
        if isinstance(node.func, ast.Subscript):
            kernel_name = self.guard._qualname(node.func.value)
            if kernel_name in self.guard._kernels:
                self.launches.append(node)
        else:
            self.calls.append(node)
        self.generic_visit(node)


def _loaded_names(node: ast.AST) -> set[str]:
    return {
        child.id
        for child in ast.walk(node)
        if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load)
    }


class SourceGuard(ast.NodeVisitor):
    """Small source policy for unsafe host access and obvious compute bypasses."""

    _SAFE_IMPORT_ROOTS = frozenset(
        {"__future__", "math", "torch", "triton", "typing"}
    )
    _FORBIDDEN_BUILTINS = frozenset(
        {
            "__import__",
            "breakpoint",
            "compile",
            "delattr",
            "dir",
            "eval",
            "exec",
            "exit",
            "getattr",
            "globals",
            "hasattr",
            "help",
            "input",
            "locals",
            "memoryview",
            "open",
            "quit",
            "setattr",
            "vars",
        }
    )
    _REFLECTION_ATTRIBUTES = frozenset(
        {
            "__bases__",
            "__class__",
            "__closure__",
            "__code__",
            "__dict__",
            "__getattribute__",
            "__globals__",
            "__subclasses__",
        }
    )
    _HOST_TENSOR_COMPUTE_METHODS = frozenset(
        {
            "addmm",
            "backward",
            "bmm",
            "clone",
            "contiguous",
            "copy_",
            "cpu",
            "data_ptr",
            "einsum",
            "flatten",
            "item",
            "matmul",
            "mean",
            "mm",
            "numpy",
            "permute",
            "reshape",
            "resize_",
            "scatter_",
            "set_",
            "softmax",
            "storage",
            "sum",
            "tolist",
            "transpose",
            "untyped_storage",
            "view",
        }
    )
    _SAFE_TORCH_ALLOCATORS = frozenset(
        {"torch.empty", "torch.empty_like", "torch.empty_strided"}
    )
    _SAFE_TRITON_HOST_CALLS = frozenset(
        {
            "triton.Config",
            "triton.autotune",
            "triton.cdiv",
            "triton.heuristics",
            "triton.jit",
            "triton.next_power_of_2",
        }
    )

    def __init__(
        self,
        *,
        allowed_import_roots: Sequence[str] = ("torch", "triton"),
        forbidden_import_roots: Sequence[str] = (
            "ctypes",
            "http",
            "multiprocessing",
            "os",
            "pathlib",
            "requests",
            "shutil",
            "socket",
            "subprocess",
            "sys",
            "urllib",
        ),
        forbidden_call_prefixes: Sequence[str] = (
            "torch.matmul",
            "torch.mm",
            "torch.bmm",
            "torch.softmax",
            "torch.sum",
            "torch.mean",
            "torch.nn",
            "torch.nn.functional",
            "torch.compile",
            "torch.ops",
        ),
        maximum_source_bytes: int = 262_144,
        required_entrypoint: str = "custom_op",
        maximum_ast_nodes: int = 50_000,
        maximum_kernels: int = 64,
        maximum_autotune_configs: int = 32,
    ) -> None:
        del allowed_import_roots, maximum_kernels, maximum_autotune_configs
        self.allowed_import_roots = self._SAFE_IMPORT_ROOTS
        self.forbidden_import_roots = frozenset(forbidden_import_roots)
        self.forbidden_call_prefixes = tuple(forbidden_call_prefixes)
        self.maximum_source_bytes = maximum_source_bytes
        self.required_entrypoint = required_entrypoint
        self.maximum_ast_nodes = maximum_ast_nodes
        self._aliases: dict[str, str] = {}
        self._findings: list[GuardFinding] = []
        self._entry: ast.FunctionDef | None = None
        self._functions: dict[str, ast.FunctionDef] = {}
        self._kernels: set[str] = set()
        self._launch = False
        self._kernel_depth = 0

    def check_path(self, path: Path | str) -> SourceGuardResult:
        candidate = Path(path)
        if candidate.is_symlink() or not candidate.is_file():
            return self._result(
                b"",
                False,
                (
                    GuardFinding(
                        "invalid_path",
                        "candidate must be a regular non-symlink file",
                    ),
                ),
            )
        return self.check_bytes(candidate.read_bytes())

    def check(self, source: str) -> SourceGuardResult:
        return self.check_bytes(source.encode("utf-8"))

    def check_bytes(self, source: bytes) -> SourceGuardResult:
        self._reset()
        if len(source) > self.maximum_source_bytes:
            self._add(
                "source_too_large",
                f"source exceeds {self.maximum_source_bytes} bytes",
            )
        try:
            text = source.decode("utf-8")
        except UnicodeDecodeError as exc:
            return self._result(
                source,
                False,
                (GuardFinding("invalid_utf8", str(exc)),),
            )
        if "\x00" in text:
            self._add("nul_byte", "source contains NUL")
        try:
            tree = ast.parse(
                text,
                filename="candidate.py",
                mode="exec",
                type_comments=True,
            )
        except SyntaxError as exc:
            return self._result(
                source,
                False,
                (GuardFinding("syntax_error", exc.msg, exc.lineno, exc.offset),),
            )
        node_count = sum(1 for _ in ast.walk(tree))
        if node_count > self.maximum_ast_nodes:
            self._add(
                "ast_too_large",
                f"AST contains {node_count} nodes (limit {self.maximum_ast_nodes})",
            )

        self._collect_roles(tree)
        self.visit(tree)
        if self._entry is None:
            self._add(
                "missing_entry_point",
                f"required function {self.required_entrypoint!r} was not found",
            )
        if not self._kernels:
            self._add(
                "missing_triton_kernel",
                "no function decorated with @triton.jit was found",
            )
        if self._entry is not None and self._kernels:
            self._check_launch_and_outputs()
        if not self._launch:
            self._add(
                "missing_kernel_launch",
                "custom_op or a reachable helper does not launch a Triton kernel",
            )
        return self._result(source, True, tuple(self._findings))

    def _reset(self) -> None:
        self._aliases = {}
        self._findings = []
        self._entry = None
        self._functions = {}
        self._kernels = set()
        self._launch = False
        self._kernel_depth = 0

    @staticmethod
    def _root(name: str) -> str:
        return name.split(".", 1)[0]

    def _qualname(self, node: ast.AST) -> str | None:
        if isinstance(node, ast.Name):
            return self._aliases.get(node.id, node.id)
        if isinstance(node, ast.Attribute):
            parent = self._qualname(node.value)
            return f"{parent}.{node.attr}" if parent else None
        return None

    def _collect_roles(self, tree: ast.Module) -> None:
        for statement in tree.body:
            if isinstance(statement, ast.Import):
                for alias in statement.names:
                    self._aliases[alias.asname or self._root(alias.name)] = alias.name
            elif isinstance(statement, ast.ImportFrom):
                module = statement.module or ""
                for alias in statement.names:
                    self._aliases[alias.asname or alias.name] = (
                        f"{module}.{alias.name}"
                    )

        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef):
                continue
            is_kernel = any(
                self._qualname(
                    decorator.func if isinstance(decorator, ast.Call) else decorator
                )
                in {"triton.jit", "triton.language.jit"}
                for decorator in node.decorator_list
            )
            if is_kernel:
                self._kernels.add(node.name)
                if not is_safe_kernel_identifier(node.name):
                    self._add(
                        "unsafe_kernel_name",
                        "Triton kernel name must be a bounded Python identifier",
                        node,
                    )
            else:
                self._functions[node.name] = node

        self._entry = next(
            (
                statement
                for statement in reversed(tree.body)
                if isinstance(statement, ast.FunctionDef)
                and statement.name == self.required_entrypoint
            ),
            None,
        )
        if self.required_entrypoint in self._kernels:
            self._add(
                "entrypoint_is_kernel",
                "custom_op must be a host wrapper, not a Triton kernel",
                self._entry,
            )

    def _result(
        self,
        source: bytes,
        syntax_ok: bool,
        findings: tuple[GuardFinding, ...],
    ) -> SourceGuardResult:
        return SourceGuardResult(
            passed=syntax_ok and not findings,
            syntax_ok=syntax_ok,
            entry_point_found=self._entry is not None,
            triton_kernel_found=bool(self._kernels),
            kernel_launch_found=self._launch,
            source_sha256=hashlib.sha256(source).hexdigest(),
            source_bytes=len(source),
            findings=findings,
        )

    def _add(
        self,
        code: str,
        message: str,
        node: ast.AST | None = None,
    ) -> None:
        self._findings.append(
            GuardFinding(
                code,
                message,
                getattr(node, "lineno", None),
                getattr(node, "col_offset", None),
            )
        )

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            root = self._root(alias.name)
            if (
                root in self.forbidden_import_roots
                or root not in self.allowed_import_roots
            ):
                self._add(
                    "forbidden_import",
                    f"import of {alias.name!r} is forbidden",
                    node,
                )
            self._aliases[alias.asname or root] = alias.name

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        module = node.module or ""
        root = self._root(module)
        if (
            node.level
            or root in self.forbidden_import_roots
            or root not in self.allowed_import_roots
        ):
            self._add(
                "forbidden_import",
                f"import from {module!r} is forbidden",
                node,
            )
        for alias in node.names:
            if alias.name == "*":
                self._add("star_import", "star imports are forbidden", node)
            self._aliases[alias.asname or alias.name] = f"{module}.{alias.name}"

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        for decorator in node.decorator_list:
            self.visit(decorator)
        for default in (*node.args.defaults, *node.args.kw_defaults):
            if default is not None:
                self.visit(default)
        arguments = (
            *node.args.posonlyargs,
            *node.args.args,
            *node.args.kwonlyargs,
        )
        for argument in arguments:
            if argument.annotation is not None:
                self.visit(argument.annotation)
        if node.returns is not None:
            self.visit(node.returns)
        is_kernel = node.name in self._kernels
        if is_kernel:
            self._kernel_depth += 1
        try:
            for statement in node.body:
                self.visit(statement)
        finally:
            if is_kernel:
                self._kernel_depth -= 1

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if node.attr.startswith("__") or node.attr in self._REFLECTION_ATTRIBUTES:
            self._add(
                "forbidden_attribute",
                f"attribute {node.attr!r} is forbidden",
                node,
            )
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        name = self._qualname(node.func)
        if name in self._FORBIDDEN_BUILTINS:
            self._add(
                "forbidden_builtin",
                f"call to {name!r} is forbidden",
                node,
            )
        if name is not None:
            root = self._root(name)
            resolved = self._aliases.get(root, root) + name[len(root) :]
            if self._kernel_depth == 0:
                if any(
                    resolved == prefix or resolved.startswith(prefix + ".")
                    for prefix in self.forbidden_call_prefixes
                ):
                    self._add(
                        "forbidden_call",
                        f"call to {resolved!r} is forbidden",
                        node,
                    )
                if (
                    resolved.startswith("torch.")
                    and resolved not in self._SAFE_TORCH_ALLOCATORS
                ):
                    self._add(
                        "forbidden_torch_call",
                        f"host torch call {resolved!r} is not an output allocator",
                        node,
                    )
                if (
                    resolved.startswith("triton.")
                    and not resolved.startswith("triton.language.")
                    and resolved not in self._SAFE_TRITON_HOST_CALLS
                ):
                    self._add(
                        "forbidden_triton_runtime_call",
                        f"host-side Triton call {resolved!r} is forbidden",
                        node,
                    )
        if (
            self._kernel_depth == 0
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in self._HOST_TENSOR_COMPUTE_METHODS
        ):
            self._add(
                "forbidden_tensor_method",
                f"host method {node.func.attr!r} may perform Tensor computation",
                node,
            )
        self.generic_visit(node)

    def _facts(self, function: ast.FunctionDef) -> _FunctionFacts:
        return _BodyFacts(self).facts(function)

    def _check_launch_and_outputs(self) -> None:
        entry = self._entry
        facts = {name: self._facts(function) for name, function in self._functions.items()}
        facts[self.required_entrypoint] = self._facts(entry)

        reachable: set[str] = set()
        queue = deque([self.required_entrypoint])
        while queue:
            name = queue.popleft()
            if name in reachable or name not in facts:
                continue
            reachable.add(name)
            for call in facts[name].calls:
                called = self._qualname(call.func)
                if called in facts and called not in reachable:
                    queue.append(called)

        self._launch = any(facts[name].launches for name in reachable)
        if not self._launch:
            return

        allocated = any(
            self._qualname(call.func) in self._SAFE_TORCH_ALLOCATORS
            for name in reachable
            for call in facts[name].calls
        )
        if not allocated:
            self._add(
                "missing_output_allocator",
                "custom_op or a reachable helper must allocate an output tensor",
                entry,
            )
        if not facts[self.required_entrypoint].returns:
            self._add(
                "invalid_host_return",
                "custom_op must return an allocated output",
                entry,
            )
