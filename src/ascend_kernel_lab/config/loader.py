"""Safe YAML configuration loader with exact-key validation."""

from __future__ import annotations

import dataclasses
import types
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, TypeVar, Union, cast, get_args, get_origin, get_type_hints

import yaml

from .models import ExperimentConfig

T = TypeVar("T")


class ConfigError(ValueError):
    """A configuration is malformed, ambiguous, or unsafe."""


def _convert(value: Any, annotation: Any, path: str) -> Any:
    origin = get_origin(annotation)
    args = get_args(annotation)
    if origin in {Union, types.UnionType}:
        if value is None and type(None) in args:
            return None
        candidates = [item for item in args if item is not type(None)]
        errors: list[str] = []
        for candidate in candidates:
            try:
                return _convert(value, candidate, path)
            except ConfigError as exc:
                errors.append(str(exc))
        raise ConfigError(f"{path}: value does not match expected type ({'; '.join(errors)})")
    if dataclasses.is_dataclass(annotation):
        return _dataclass_from_mapping(cast(type[Any], annotation), value, path)
    if origin in {tuple, list, Sequence}:
        if not isinstance(value, list):
            raise ConfigError(f"{path}: expected a YAML sequence")
        item_type = args[0] if args else Any
        converted = [_convert(item, item_type, f"{path}[{index}]") for index, item in enumerate(value)]
        return tuple(converted) if origin is tuple else converted
    if origin in {dict, Mapping}:
        if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
            raise ConfigError(f"{path}: expected a string-keyed mapping")
        key_type, item_type = args or (str, Any)
        return {
            _convert(key, key_type, f"{path}.<key>"): _convert(item, item_type, f"{path}.{key}")
            for key, item in value.items()
        }
    if annotation is Any:
        return value
    if annotation is bool:
        if type(value) is not bool:
            raise ConfigError(f"{path}: expected a boolean")
        return value
    if annotation is int:
        if type(value) is not int:
            raise ConfigError(f"{path}: expected an integer")
        return value
    if annotation is float:
        if type(value) not in {int, float}:
            raise ConfigError(f"{path}: expected a number")
        return float(value)
    if annotation is str:
        if not isinstance(value, str):
            raise ConfigError(f"{path}: expected a string")
        return value
    if annotation is Path:
        if not isinstance(value, (str, Path)):
            raise ConfigError(f"{path}: expected a filesystem path")
        return Path(value)
    return value


def _dataclass_from_mapping(cls: type[T], value: Any, path: str) -> T:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ConfigError(f"{path}: expected a string-keyed mapping")
    fields = {
        item.name: item
        for item in dataclasses.fields(cast(Any, cls))
        if item.init
    }
    unknown = sorted(set(value) - set(fields))
    if unknown:
        raise ConfigError(f"{path}: unknown field(s): {', '.join(unknown)}")
    hints = get_type_hints(cls)
    required = [
        item.name for item in fields.values()
        if item.default is dataclasses.MISSING and item.default_factory is dataclasses.MISSING
    ]
    missing = sorted(set(required) - set(value))
    if missing:
        raise ConfigError(f"{path}: missing required field(s): {', '.join(missing)}")
    kwargs = {
        key: _convert(item, hints[key], f"{path}.{key}")
        for key, item in value.items()
        if key in hints
    }
    try:
        return cls(**kwargs)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{path}: {exc}") from exc


def _find_project_root(config_path: Path) -> Path:
    for parent in (config_path.parent, *config_path.parents):
        if (parent / "pyproject.toml").is_file():
            return parent
    return config_path.parent


def load_config(path: str | Path, *, project_root: str | Path | None = None) -> ExperimentConfig:
    """Load one experiment YAML without interpolation or object construction."""

    config_path = Path(path).expanduser().resolve()
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise ConfigError(f"cannot load {config_path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigError("configuration root must be a mapping")
    expected_root = {
        "experiment", "model", "worker", "timeouts", "benchmark", "profile", "storage", "security"
    }
    unknown = sorted(set(raw) - expected_root)
    if unknown:
        raise ConfigError(f"configuration: unknown section(s): {', '.join(unknown)}")
    if "experiment" not in raw:
        raise ConfigError("configuration: missing required section: experiment")
    experiment = raw.pop("experiment")
    if not isinstance(experiment, dict):
        raise ConfigError("configuration.experiment: expected a mapping")
    combined = dict(experiment)
    combined.update(raw)
    combined["config_path"] = config_path
    combined["project_root"] = Path(project_root).resolve() if project_root else _find_project_root(config_path)
    return _dataclass_from_mapping(ExperimentConfig, combined, "configuration")
