from __future__ import annotations

import ast
import hashlib
import re
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


_GENERIC_KERNEL_NAMES = frozenset(
    {"_", "f", "fn", "k", "kernel", "main", "op", "run", "triton"}
)


def is_safe_kernel_identifier(name: str) -> bool:
    """Whether a source-level kernel name is specific enough for attribution."""

    return (
        8 <= len(name) <= 128
        and name.isidentifier()
        and name.lower() not in _GENERIC_KERNEL_NAMES
        and sum(character.isalpha() for character in name) >= 4
        and sum(character.isalnum() for character in name) >= 6
    )


def candidate_kernel_pattern(name: str) -> str:
    """Return a bounded, anchored profiler identity for one trusted AST name.

    Triton/CANN releases commonly preserve the Python function name exactly.
    A compiler-added hexadecimal identity suffix is accepted, but arbitrary
    prefixes/substrings and descriptive suffixes are deliberately rejected.
    """

    if not is_safe_kernel_identifier(name):
        raise ValueError("kernel name is too broad for safe profiler attribution")
    return rf"^{re.escape(name)}(?:_[0-9a-fA-F]{{8,64}})?$"


class SourceGuard(ast.NodeVisitor):
    """Fail-closed AST policy for model-generated Triton candidates.

    This is a strong validation layer, not an OS security boundary. Production
    deployments must also run candidates as a dedicated sandboxed user with no
    network and no credentials.
    """

    _FORBIDDEN_BUILTINS = frozenset({
        "open", "eval", "exec", "compile", "__import__", "input", "breakpoint",
        "globals", "locals", "vars", "dir", "getattr", "setattr", "delattr",
        "hasattr", "memoryview", "help", "exit", "quit",
    })
    _FORBIDDEN_ATTRIBUTES = frozenset({
        "cpu", "numpy", "tolist", "item", "data_ptr", "storage", "untyped_storage",
        "sum", "mean", "softmax", "matmul", "mm", "bmm", "addmm", "einsum",
        "backward", "register_hook", "set_", "resize_", "copy_", "scatter_",
        "clone", "contiguous", "flatten", "permute", "reshape", "transpose", "view",
        "__dict__", "__class__", "__bases__", "__subclasses__", "__globals__",
        "__code__", "__closure__", "__getattribute__",
    })
    _SAFE_TORCH_CALLS = frozenset({
        "torch.empty", "torch.empty_like", "torch.empty_strided",
    })
    _SAFE_TRITON_HOST_CALLS = frozenset({
        "triton.Config", "triton.cdiv", "triton.next_power_of_2",
    })
    _SAFE_TRITON_PREFIXES = ("triton.", "tl.", "triton.language.")

    def __init__(
        self,
        *,
        allowed_import_roots: Sequence[str] = ("torch", "triton"),
        forbidden_import_roots: Sequence[str] = (
            "ctypes", "http", "multiprocessing", "os", "pathlib", "requests",
            "shutil", "socket", "subprocess", "sys", "urllib",
        ),
        forbidden_call_prefixes: Sequence[str] = (
            "torch.matmul", "torch.mm", "torch.bmm", "torch.softmax", "torch.sum",
            "torch.mean", "torch.nn", "torch.nn.functional", "torch.compile", "torch.ops",
        ),
        maximum_source_bytes: int = 262_144,
        required_entrypoint: str = "custom_op",
        maximum_ast_nodes: int = 50_000,
        maximum_kernels: int = 64,
        maximum_autotune_configs: int = 32,
    ) -> None:
        self.allowed_import_roots = frozenset(allowed_import_roots)
        self.forbidden_import_roots = frozenset(forbidden_import_roots)
        self.forbidden_call_prefixes = tuple(forbidden_call_prefixes)
        self.maximum_source_bytes = maximum_source_bytes
        self.required_entrypoint = required_entrypoint
        self.maximum_ast_nodes = maximum_ast_nodes
        self.maximum_kernels = maximum_kernels
        self.maximum_autotune_configs = maximum_autotune_configs
        self._aliases: dict[str, str] = {}
        self._findings: list[GuardFinding] = []
        self._warnings: list[GuardFinding] = []
        self._entry = False
        self._kernels: set[str] = set()
        self._launch = False
        self._function_depth = 0
        self._kernel_depth = 0
        self._autotune_config_count = 0

    def check_path(self, path: Path | str) -> SourceGuardResult:
        candidate = Path(path)
        if candidate.is_symlink() or not candidate.is_file():
            source = b""
            return self._result(source, False, (GuardFinding("invalid_path", "candidate must be a regular non-symlink file"),))
        source = candidate.read_bytes()
        return self.check_bytes(source)

    def check(self, source: str) -> SourceGuardResult:
        return self.check_bytes(source.encode("utf-8"))

    def check_bytes(self, source: bytes) -> SourceGuardResult:
        self._reset()
        if len(source) > self.maximum_source_bytes:
            self._add("source_too_large", f"source exceeds {self.maximum_source_bytes} bytes")
        try:
            text = source.decode("utf-8")
        except UnicodeDecodeError as exc:
            return self._result(source, False, (GuardFinding("invalid_utf8", str(exc)),))
        if "\x00" in text:
            self._add("nul_byte", "source contains NUL")
        try:
            tree = ast.parse(text, filename="candidate.py", mode="exec", type_comments=True)
        except SyntaxError as exc:
            return self._result(
                source,
                False,
                (GuardFinding("syntax_error", exc.msg, exc.lineno, exc.offset),),
            )
        nodes = sum(1 for _ in ast.walk(tree))
        if nodes > self.maximum_ast_nodes:
            self._add("ast_too_large", f"AST contains {nodes} nodes (limit {self.maximum_ast_nodes})")
        self._precollect_function_roles(tree)
        self.visit(tree)
        entry_functions = [
            statement
            for statement in tree.body
            if isinstance(statement, ast.FunctionDef)
            and statement.name == self.required_entrypoint
        ]
        if len(entry_functions) == 1:
            policy = _HostEntrypointPolicy(
                self,
                kernels=self._kernels - {self.required_entrypoint},
                module_constants=self._module_constants(tree),
            )
            self._findings.extend(policy.check(entry_functions[0]))
            self._launch = policy.kernel_launch_found
        else:
            self._launch = False
        if not self._entry:
            self._add("missing_entry_point", f"required function {self.required_entrypoint!r} was not found")
        if not self._kernels:
            self._add("missing_triton_kernel", "no function decorated with @triton.jit was found")
        if len(self._kernels) > self.maximum_kernels:
            self._add("too_many_kernels", f"candidate defines {len(self._kernels)} Triton kernels")
        if self._autotune_config_count > self.maximum_autotune_configs:
            self._add(
                "too_many_autotune_configs",
                f"candidate defines {self._autotune_config_count} autotune configs "
                f"(limit {self.maximum_autotune_configs})",
            )
        if not self._launch:
            self._add("missing_kernel_launch", "custom code does not launch a Triton kernel")
        return self._result(source, True, tuple(self._findings))

    def _reset(self) -> None:
        self._aliases = {}
        self._findings = []
        self._warnings = []
        self._entry = False
        self._kernels = set()
        self._launch = False
        self._function_depth = 0
        self._kernel_depth = 0
        self._autotune_config_count = 0

    def _precollect_function_roles(self, tree: ast.Module) -> None:
        for statement in tree.body:
            if isinstance(statement, ast.Import):
                for alias in statement.names:
                    self._aliases[alias.asname or self._root(alias.name)] = alias.name
            elif isinstance(statement, ast.ImportFrom):
                module = statement.module or ""
                for alias in statement.names:
                    self._aliases[alias.asname or alias.name] = f"{module}.{alias.name}"

        function_counts: dict[str, int] = {}
        entry_count = 0
        for statement in tree.body:
            if not isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            function_counts[statement.name] = function_counts.get(statement.name, 0) + 1
            if statement.name == self.required_entrypoint:
                entry_count += 1
            decorated_as_kernel = any(
                self._qualname(
                    decorator.func if isinstance(decorator, ast.Call) else decorator
                )
                in {"triton.jit", "triton.language.jit"}
                for decorator in statement.decorator_list
            )
            if decorated_as_kernel:
                self._kernels.add(statement.name)
                if not is_safe_kernel_identifier(statement.name):
                    self._add(
                        "unsafe_kernel_name",
                        "Triton kernel name is too broad for profiler attribution",
                        statement,
                    )
                if statement.name == self.required_entrypoint:
                    self._add(
                        "entrypoint_is_kernel",
                        "custom_op must be a host wrapper, not a Triton kernel",
                        statement,
                    )
        if entry_count > 1:
            self._add(
                "duplicate_entry_point",
                f"required function {self.required_entrypoint!r} is defined more than once",
            )
        for name, count in function_counts.items():
            if count > 1:
                self._add(
                    "duplicate_function",
                    f"function {name!r} is defined more than once",
                )

    @staticmethod
    def _module_constants(tree: ast.Module) -> frozenset[str]:
        names: set[str] = set()
        for statement in tree.body:
            if isinstance(statement, ast.Assign) and SourceGuard._is_constant_expression(
                statement.value
            ):
                names.update(
                    target.id for target in statement.targets if isinstance(target, ast.Name)
                )
            elif (
                isinstance(statement, ast.AnnAssign)
                and isinstance(statement.target, ast.Name)
                and statement.value is not None
                and SourceGuard._is_constant_expression(statement.value)
            ):
                names.add(statement.target.id)
        return frozenset(names)

    def _result(
        self,
        source: bytes,
        syntax_ok: bool,
        findings: tuple[GuardFinding, ...],
    ) -> SourceGuardResult:
        return SourceGuardResult(
            passed=syntax_ok and not findings,
            syntax_ok=syntax_ok,
            entry_point_found=self._entry,
            triton_kernel_found=bool(self._kernels),
            kernel_launch_found=self._launch,
            source_sha256=hashlib.sha256(source).hexdigest(),
            source_bytes=len(source),
            findings=findings,
            warnings=tuple(self._warnings),
        )

    def _add(self, code: str, message: str, node: ast.AST | None = None) -> None:
        self._findings.append(GuardFinding(code, message, getattr(node, "lineno", None), getattr(node, "col_offset", None)))

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

    @staticmethod
    def _is_constant_expression(node: ast.AST) -> bool:
        if isinstance(node, ast.Constant):
            return isinstance(node.value, (str, bytes, int, float, bool, type(None)))
        if isinstance(node, (ast.Tuple, ast.List, ast.Set)):
            return all(SourceGuard._is_constant_expression(item) for item in node.elts)
        if isinstance(node, ast.Dict):
            return all(
                key is not None
                and SourceGuard._is_constant_expression(key)
                and SourceGuard._is_constant_expression(value)
                for key, value in zip(node.keys, node.values, strict=True)
            )
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            return SourceGuard._is_constant_expression(node.operand)
        return False

    def visit_Module(self, node: ast.Module) -> None:
        allowed = (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.Assign, ast.AnnAssign)
        for statement in node.body:
            if not isinstance(statement, allowed):
                self._add("top_level_side_effect", f"top-level {type(statement).__name__} is forbidden", statement)
            if isinstance(statement, ast.Assign) and not self._is_constant_expression(statement.value):
                self._add("top_level_dynamic_assignment", "top-level assignments must contain only literals", statement)
            if isinstance(statement, ast.AnnAssign) and statement.value is not None and not self._is_constant_expression(statement.value):
                self._add("top_level_dynamic_assignment", "top-level assignments must contain only literals", statement)
        self.generic_visit(node)

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            root = self._root(alias.name)
            if root in self.forbidden_import_roots or root not in self.allowed_import_roots:
                self._add("forbidden_import", f"import of {alias.name!r} is forbidden", node)
            self._aliases[alias.asname or root] = alias.name

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        module = node.module or ""
        root = self._root(module)
        if node.level or root in self.forbidden_import_roots or root not in self.allowed_import_roots:
            self._add("forbidden_import", f"import from {module!r} is forbidden", node)
        for alias in node.names:
            if alias.name == "*":
                self._add("star_import", "star imports are forbidden", node)
            self._aliases[alias.asname or alias.name] = f"{module}.{alias.name}"

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        if node.name == self.required_entrypoint and self._function_depth == 0:
            self._entry = True
        for decorator in node.decorator_list:
            name = self._qualname(decorator.func if isinstance(decorator, ast.Call) else decorator)
            if name in {"triton.jit", "triton.language.jit"}:
                self._kernels.add(node.name)
            elif name == "triton.autotune":
                configs: ast.AST | None = None
                if isinstance(decorator, ast.Call):
                    configs = next(
                        (
                            keyword.value
                            for keyword in decorator.keywords
                            if keyword.arg == "configs"
                        ),
                        decorator.args[0] if decorator.args else None,
                    )
                if not isinstance(configs, (ast.List, ast.Tuple)):
                    self._add(
                        "dynamic_autotune_configs",
                        "@triton.autotune configs must be a literal list or tuple",
                        decorator,
                    )
                else:
                    self._autotune_config_count += len(configs.elts)
            elif name != "triton.heuristics":
                self._add("forbidden_decorator", f"decorator {name or type(decorator).__name__!r} is forbidden", decorator)

        # Decorators, defaults, and annotations execute or resolve in host
        # Python. Visit them before entering the Triton-kernel context.
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
        if node.args.vararg is not None and node.args.vararg.annotation is not None:
            self.visit(node.args.vararg.annotation)
        if node.args.kwarg is not None and node.args.kwarg.annotation is not None:
            self.visit(node.args.kwarg.annotation)
        if node.returns is not None:
            self.visit(node.returns)

        self._function_depth += 1
        is_kernel = node.name in self._kernels
        if is_kernel:
            self._kernel_depth += 1
        try:
            for statement in node.body:
                self.visit(statement)
        finally:
            if is_kernel:
                self._kernel_depth -= 1
            self._function_depth -= 1

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._add("async_forbidden", "async functions are forbidden", node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._add("class_forbidden", "classes are forbidden", node)

    def visit_Lambda(self, node: ast.Lambda) -> None:
        self._add("lambda_forbidden", "lambda expressions are forbidden", node)

    def visit_Global(self, node: ast.Global) -> None:
        self._add("global_forbidden", "global mutation is forbidden", node)

    def visit_Nonlocal(self, node: ast.Nonlocal) -> None:
        self._add("nonlocal_forbidden", "nonlocal mutation is forbidden", node)

    def visit_Raise(self, node: ast.Raise) -> None:
        self._add("raise_forbidden", "raising arbitrary exceptions is forbidden", node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if node.attr.startswith("__") or (
            self._kernel_depth == 0 and node.attr in self._FORBIDDEN_ATTRIBUTES
        ):
            self._add("forbidden_attribute", f"attribute {node.attr!r} is forbidden", node)
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        name = self._qualname(node.func)
        if name is not None:
            if name in self._FORBIDDEN_BUILTINS:
                self._add("forbidden_builtin", f"call to {name!r} is forbidden", node)
            resolved = self._aliases.get(self._root(name), self._root(name)) + name[len(self._root(name)):]
            if any(resolved == prefix or resolved.startswith(prefix + ".") for prefix in self.forbidden_call_prefixes):
                self._add("forbidden_call", f"call to {resolved!r} is forbidden", node)
            if (
                self._kernel_depth == 0
                and resolved.startswith("torch.")
                and resolved not in self._SAFE_TORCH_CALLS
            ):
                self._add("forbidden_torch_call", f"torch call {resolved!r} is not an allowed allocator", node)
            if (
                self._kernel_depth == 0
                and
                resolved.startswith("triton.")
                and not resolved.startswith("triton.language.")
                and resolved not in self._SAFE_TRITON_HOST_CALLS
                and resolved not in {"triton.jit", "triton.autotune", "triton.heuristics"}
            ):
                self._add(
                    "forbidden_triton_runtime_call",
                    f"host-side Triton call {resolved!r} is forbidden",
                    node,
                )
        self.generic_visit(node)

    def visit_Delete(self, node: ast.Delete) -> None:
        self._add("delete_forbidden", "delete statements are forbidden", node)

    def visit_With(self, node: ast.With) -> None:
        self._add("with_forbidden", "context managers are forbidden", node)

    def visit_AsyncWith(self, node: ast.AsyncWith) -> None:
        self._add("async_forbidden", "async context managers are forbidden", node)

    def visit_Await(self, node: ast.Await) -> None:
        self._add("async_forbidden", "await is forbidden", node)

    def visit_Yield(self, node: ast.Yield) -> None:
        self._add("generator_forbidden", "generators are forbidden", node)

    def visit_YieldFrom(self, node: ast.YieldFrom) -> None:
        self._add("generator_forbidden", "generators are forbidden", node)


@dataclass(frozen=True)
class _HostValue:
    kinds: frozenset[str]
    output_origins: frozenset[str] = frozenset()
    fresh_output: bool = False

    def merge(self, other: _HostValue) -> _HostValue:
        return _HostValue(
            self.kinds | other.kinds,
            self.output_origins | other.output_origins,
            self.fresh_output and other.fresh_output,
        )


_SCALAR = _HostValue(frozenset({"scalar"}))
_INPUT = _HostValue(frozenset({"input"}))
_UNKNOWN = _HostValue(frozenset({"unknown"}))
_FRESH_OUTPUT = _HostValue(frozenset({"output"}), fresh_output=True)


class _HostEntrypointPolicy:
    """Conservative data-flow policy for the untrusted Python host wrapper."""

    _TENSOR_METADATA_PROPERTIES = frozenset(
        {"device", "dtype", "layout", "ndim", "shape"}
    )
    _TENSOR_METADATA_METHODS = frozenset(
        {"dim", "element_size", "is_contiguous", "numel", "size", "stride"}
    )
    _SCALAR_BUILTINS = frozenset(
        {"abs", "bool", "float", "int", "len", "list", "max", "min", "range", "tuple"}
    )
    _TORCH_CONSTANTS = frozenset(
        {
            "bfloat16",
            "bool",
            "contiguous_format",
            "float16",
            "float32",
            "float64",
            "int16",
            "int32",
            "int64",
            "int8",
            "preserve_format",
            "uint8",
        }
    )

    def __init__(
        self,
        guard: SourceGuard,
        *,
        kernels: frozenset[str] | set[str],
        module_constants: frozenset[str],
    ) -> None:
        self.guard = guard
        self.kernels = frozenset(kernels)
        self.module_constants = module_constants
        self.findings: list[GuardFinding] = []
        self.kernel_launch_found = False

    def _add(self, code: str, message: str, node: ast.AST | None = None) -> None:
        self.findings.append(
            GuardFinding(
                code,
                message,
                getattr(node, "lineno", None),
                getattr(node, "col_offset", None),
            )
        )

    @staticmethod
    def _is_tensor(value: _HostValue) -> bool:
        return bool(value.kinds & {"input", "output"})

    def _require_scalar(
        self, value: _HostValue, node: ast.AST, *, context: str
    ) -> _HostValue:
        if self._is_tensor(value):
            self._add(
                "forbidden_tensor_operation",
                f"Tensor values are forbidden in host-side {context}",
                node,
            )
            return _UNKNOWN
        return _SCALAR if value.kinds == {"scalar"} else _UNKNOWN

    def check(self, function: ast.FunctionDef) -> tuple[GuardFinding, ...]:
        if function.args.vararg is not None or function.args.kwarg is not None:
            self._add(
                "variadic_entry_point",
                "custom_op may not use variadic arguments",
                function,
            )
        arguments = (
            *function.args.posonlyargs,
            *function.args.args,
            *function.args.kwonlyargs,
        )
        environment = {argument.arg: _INPUT for argument in arguments}
        environment.update({name: _SCALAR for name in self.module_constants})
        self._block(function.body, environment, frozenset())
        if not self._block_definitely_returns(function.body):
            self._add(
                "missing_host_return",
                "every custom_op control-flow path must return an allocated output",
                function,
            )
        return tuple(self.findings)

    @classmethod
    def _block_definitely_returns(cls, statements: Sequence[ast.stmt]) -> bool:
        for statement in statements:
            if isinstance(statement, ast.Return):
                return True
            if (
                isinstance(statement, ast.If)
                and statement.orelse
                and cls._block_definitely_returns(statement.body)
                and cls._block_definitely_returns(statement.orelse)
            ):
                return True
        return False

    @staticmethod
    def _merge_environments(
        left: dict[str, _HostValue], right: dict[str, _HostValue]
    ) -> dict[str, _HostValue]:
        merged: dict[str, _HostValue] = {}
        for name in left.keys() | right.keys():
            merged[name] = left.get(name, _UNKNOWN).merge(right.get(name, _UNKNOWN))
        return merged

    def _block(
        self,
        statements: Sequence[ast.stmt],
        environment: dict[str, _HostValue],
        launched: frozenset[str],
    ) -> tuple[dict[str, _HostValue], frozenset[str]]:
        current = dict(environment)
        definite_launches = launched
        for statement in statements:
            current, definite_launches = self._statement(
                statement, current, definite_launches
            )
        return current, definite_launches

    def _statement(
        self,
        statement: ast.stmt,
        environment: dict[str, _HostValue],
        launched: frozenset[str],
    ) -> tuple[dict[str, _HostValue], frozenset[str]]:
        if isinstance(statement, ast.Assign):
            value, launches = self._expression(statement.value, environment)
            result = dict(environment)
            for target in statement.targets:
                self._assign(target, value, result)
            return result, launched | launches
        if isinstance(statement, ast.AnnAssign):
            if statement.value is None:
                self._add(
                    "uninitialized_host_value",
                    "custom_op annotations must initialize their value",
                    statement,
                )
                return environment, launched
            value, launches = self._expression(statement.value, environment)
            result = dict(environment)
            self._assign(statement.target, value, result)
            return result, launched | launches
        if isinstance(statement, ast.AugAssign):
            target_value, target_launches = self._expression(
                statement.target, environment
            )
            value, value_launches = self._expression(statement.value, environment)
            merged = self._require_scalar(
                target_value.merge(value), statement, context="augmented arithmetic"
            )
            result = dict(environment)
            self._assign(statement.target, merged, result)
            return result, launched | target_launches | value_launches
        if isinstance(statement, ast.Expr):
            _value, launches = self._expression(statement.value, environment)
            return environment, launched | launches
        if isinstance(statement, ast.Return):
            self._validate_return(statement, environment, launched)
            return environment, launched
        if isinstance(statement, ast.If):
            condition, condition_launches = self._expression(
                statement.test, environment
            )
            self._require_scalar(condition, statement.test, context="branch condition")
            base_launches = launched | condition_launches
            body_environment, body_launches = self._block(
                statement.body, dict(environment), base_launches
            )
            else_environment, else_launches = self._block(
                statement.orelse, dict(environment), base_launches
            )
            return (
                self._merge_environments(body_environment, else_environment),
                body_launches & else_launches,
            )
        if isinstance(statement, (ast.For, ast.While)):
            if isinstance(statement, ast.For):
                iterator, iterator_launches = self._expression(
                    statement.iter, environment
                )
                self._require_scalar(
                    iterator, statement.iter, context="loop iteration"
                )
                loop_environment = dict(environment)
                self._assign(statement.target, _SCALAR, loop_environment)
                test_launches = iterator_launches
            else:
                condition, test_launches = self._expression(
                    statement.test, environment
                )
                self._require_scalar(
                    condition, statement.test, context="loop condition"
                )
                loop_environment = dict(environment)
            body_environment, _body_launches = self._block(
                statement.body,
                loop_environment,
                launched | test_launches,
            )
            else_environment, else_launches = self._block(
                statement.orelse,
                dict(environment),
                launched | test_launches,
            )
            # A loop may execute zero times, so its body cannot establish a
            # definite output launch for a later return.
            return (
                self._merge_environments(
                    environment,
                    self._merge_environments(body_environment, else_environment),
                ),
                launched & else_launches,
            )
        if isinstance(statement, (ast.Pass, ast.Break, ast.Continue)):
            return environment, launched
        self._add(
            "forbidden_host_statement",
            f"custom_op statement {type(statement).__name__!r} is not allowed",
            statement,
        )
        return environment, launched

    def _assign(
        self,
        target: ast.expr,
        value: _HostValue,
        environment: dict[str, _HostValue],
    ) -> None:
        if isinstance(target, ast.Name):
            environment[target.id] = (
                _HostValue(
                    frozenset({"output"}),
                    output_origins=frozenset({target.id}),
                )
                if value.fresh_output
                else value
            )
            return
        if isinstance(target, (ast.Tuple, ast.List)):
            if value.fresh_output:
                self._add(
                    "invalid_output_unpack",
                    "an allocated Tensor may not be unpacked in custom_op",
                    target,
                )
            assigned = _UNKNOWN if self._is_tensor(value) else _SCALAR
            for item in target.elts:
                self._assign(item, assigned, environment)
            return
        self._add(
            "forbidden_host_mutation",
            "custom_op may assign only to local names",
            target,
        )

    def _validate_return(
        self,
        statement: ast.Return,
        environment: dict[str, _HostValue],
        launched: frozenset[str],
    ) -> None:
        if statement.value is None:
            self._add(
                "invalid_host_return",
                "custom_op must return an allocated Tensor output",
                statement,
            )
            return
        values = (
            tuple(statement.value.elts)
            if isinstance(statement.value, (ast.Tuple, ast.List))
            else (statement.value,)
        )
        for value_node in values:
            if not isinstance(value_node, ast.Name):
                # Evaluate it as well so a view/copy/indexing bypass gets a
                # precise finding in addition to the strict return contract.
                self._expression(value_node, environment)
                self._add(
                    "invalid_host_return",
                    "custom_op may return only named allocator outputs",
                    value_node,
                )
                continue
            value = environment.get(value_node.id, _UNKNOWN)
            if value.kinds != {"output"} or not value.output_origins:
                self._add(
                    "invalid_host_return",
                    "custom_op return value must originate from an allowed output allocator",
                    value_node,
                )
                continue
            if not value.output_origins.issubset(launched):
                self._add(
                    "unlaunched_host_output",
                    "every returned output must be passed to a candidate Triton launch",
                    value_node,
                )

    def _expression(
        self, node: ast.AST, environment: dict[str, _HostValue]
    ) -> tuple[_HostValue, frozenset[str]]:
        launches: frozenset[str]
        if isinstance(node, ast.Constant):
            return _SCALAR, frozenset()
        if isinstance(node, ast.Name):
            if node.id in environment:
                return environment[node.id], frozenset()
            self._add(
                "unknown_host_value",
                f"custom_op references unknown host value {node.id!r}",
                node,
            )
            return _UNKNOWN, frozenset()
        if isinstance(node, ast.Attribute):
            qualified = self.guard._qualname(node)
            if qualified is not None and qualified.startswith("torch."):
                if qualified.removeprefix("torch.") in self._TORCH_CONSTANTS:
                    return _SCALAR, frozenset()
                self._add(
                    "forbidden_host_attribute",
                    f"host-side torch attribute {qualified!r} is not allowed",
                    node,
                )
                return _UNKNOWN, frozenset()
            base, launches = self._expression(node.value, environment)
            if self._is_tensor(base):
                if node.attr in self._TENSOR_METADATA_PROPERTIES:
                    return _SCALAR, launches
                self._add(
                    "forbidden_tensor_attribute",
                    f"Tensor attribute {node.attr!r} is not safe host metadata",
                    node,
                )
                return _UNKNOWN, launches
            self._add(
                "forbidden_host_attribute",
                f"host-side attribute {node.attr!r} is not allowed",
                node,
            )
            return _UNKNOWN, launches
        if isinstance(node, ast.Call):
            return self._call(node, environment)
        if isinstance(node, ast.Subscript):
            value, value_launches = self._expression(node.value, environment)
            index, index_launches = self._expression(node.slice, environment)
            if self._is_tensor(value):
                self._add(
                    "forbidden_tensor_indexing",
                    "Tensor indexing is forbidden in custom_op",
                    node,
                )
                return _UNKNOWN, value_launches | index_launches
            self._require_scalar(index, node.slice, context="metadata indexing")
            return _SCALAR, value_launches | index_launches
        if isinstance(node, ast.Slice):
            launches = frozenset()
            for part in (node.lower, node.upper, node.step):
                if part is not None:
                    value, part_launches = self._expression(part, environment)
                    self._require_scalar(value, part, context="slice calculation")
                    launches |= part_launches
            return _SCALAR, launches
        if isinstance(node, (ast.Tuple, ast.List, ast.Set)):
            result = _SCALAR
            launches = frozenset()
            for item in node.elts:
                value, item_launches = self._expression(item, environment)
                result = result.merge(value)
                launches |= item_launches
            return result, launches
        if isinstance(node, ast.Dict):
            result = _SCALAR
            launches = frozenset()
            for key, item in zip(node.keys, node.values, strict=True):
                for child in (key, item):
                    if child is None:
                        continue
                    value, child_launches = self._expression(child, environment)
                    result = result.merge(value)
                    launches |= child_launches
            return result, launches
        if isinstance(node, ast.BinOp):
            left, left_launches = self._expression(node.left, environment)
            right, right_launches = self._expression(node.right, environment)
            return (
                self._require_scalar(
                    left.merge(right), node, context="arithmetic"
                ),
                left_launches | right_launches,
            )
        if isinstance(node, (ast.BoolOp, ast.Compare)):
            children = (
                tuple(node.values)
                if isinstance(node, ast.BoolOp)
                else (node.left, *node.comparators)
            )
            combined = _SCALAR
            launches = frozenset()
            for child in children:
                value, child_launches = self._expression(child, environment)
                combined = combined.merge(value)
                launches |= child_launches
            return (
                self._require_scalar(combined, node, context="comparison"),
                launches,
            )
        if isinstance(node, ast.UnaryOp):
            operand, launches = self._expression(node.operand, environment)
            return (
                self._require_scalar(operand, node, context="unary arithmetic"),
                launches,
            )
        if isinstance(node, ast.IfExp):
            condition, condition_launches = self._expression(node.test, environment)
            body, body_launches = self._expression(node.body, environment)
            otherwise, otherwise_launches = self._expression(node.orelse, environment)
            self._require_scalar(condition, node.test, context="conditional expression")
            return (
                body.merge(otherwise),
                condition_launches | (body_launches & otherwise_launches),
            )
        if isinstance(node, ast.Starred):
            return self._expression(node.value, environment)
        self._add(
            "forbidden_host_expression",
            f"custom_op expression {type(node).__name__!r} is not allowed",
            node,
        )
        return _UNKNOWN, frozenset()

    def _call(
        self, node: ast.Call, environment: dict[str, _HostValue]
    ) -> tuple[_HostValue, frozenset[str]]:
        launches: frozenset[str]
        if isinstance(node.func, ast.Subscript):
            kernel_name = self.guard._qualname(node.func.value)
            if kernel_name not in self.kernels:
                self._add(
                    "forbidden_host_call",
                    "subscripted host calls must target a declared Triton kernel",
                    node,
                )
                return _UNKNOWN, frozenset()
            grid, launches = self._expression(node.func.slice, environment)
            self._require_scalar(grid, node.func.slice, context="Triton launch grid")
            output_origins: set[str] = set()
            for argument in node.args:
                value, argument_launches = self._expression(argument, environment)
                output_origins.update(value.output_origins)
                launches |= argument_launches
            for keyword in node.keywords:
                if keyword.arg is None:
                    self._add(
                        "forbidden_host_call",
                        "Triton launch may not expand arbitrary keyword mappings",
                        keyword,
                    )
                    continue
                value, keyword_launches = self._expression(keyword.value, environment)
                output_origins.update(value.output_origins)
                launches |= keyword_launches
            self.kernel_launch_found = True
            if not output_origins:
                self._add(
                    "kernel_launch_without_output",
                    "candidate Triton launch must receive an allocated output buffer",
                    node,
                )
            return _SCALAR, launches | frozenset(output_origins)

        name = self.guard._qualname(node.func)
        if name in self.guard._SAFE_TORCH_CALLS:
            return self._allocator(node, environment, name)
        if name in self.guard._SAFE_TRITON_HOST_CALLS - {"triton.Config"}:
            launches = frozenset()
            for argument in (*node.args, *(keyword.value for keyword in node.keywords)):
                value, argument_launches = self._expression(argument, environment)
                self._require_scalar(value, argument, context=f"call to {name}")
                launches |= argument_launches
            return _SCALAR, launches
        if isinstance(node.func, ast.Attribute):
            base, launches = self._expression(node.func.value, environment)
            if self._is_tensor(base):
                if node.func.attr not in self._TENSOR_METADATA_METHODS:
                    self._add(
                        "forbidden_tensor_method",
                        f"Tensor method {node.func.attr!r} is not safe host metadata",
                        node,
                    )
                    return _UNKNOWN, launches
                for argument in (*node.args, *(keyword.value for keyword in node.keywords)):
                    value, argument_launches = self._expression(argument, environment)
                    self._require_scalar(
                        value,
                        argument,
                        context=f"Tensor metadata method {node.func.attr}",
                    )
                    launches |= argument_launches
                return _SCALAR, launches
        if isinstance(node.func, ast.Name) and node.func.id in self._SCALAR_BUILTINS:
            launches = frozenset()
            for argument in (*node.args, *(keyword.value for keyword in node.keywords)):
                value, argument_launches = self._expression(argument, environment)
                self._require_scalar(
                    value, argument, context=f"call to {node.func.id}"
                )
                launches |= argument_launches
            return _SCALAR, launches
        self._add(
            "forbidden_host_call",
            f"host call {name or type(node.func).__name__!r} is not allowed",
            node,
        )
        return _UNKNOWN, frozenset()

    def _allocator(
        self,
        node: ast.Call,
        environment: dict[str, _HostValue],
        name: str,
    ) -> tuple[_HostValue, frozenset[str]]:
        launches: frozenset[str] = frozenset()
        if name == "torch.empty_like":
            if not node.args:
                self._add(
                    "invalid_output_allocator",
                    "torch.empty_like requires a Tensor template",
                    node,
                )
            else:
                template, template_launches = self._expression(
                    node.args[0], environment
                )
                launches |= template_launches
                if not self._is_tensor(template):
                    self._add(
                        "invalid_output_allocator",
                        "torch.empty_like template must be an input or allocated Tensor",
                        node.args[0],
                    )
            positional = node.args[1:]
        else:
            positional = node.args
        for argument in (*positional, *(keyword.value for keyword in node.keywords)):
            value, argument_launches = self._expression(argument, environment)
            self._require_scalar(value, argument, context=f"allocator {name}")
            launches |= argument_launches
        return _FRESH_OUTPUT, launches
