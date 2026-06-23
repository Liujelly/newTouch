"""工具注册表 (架构文档 5.10)。

工具以 JSON Schema 格式注册，供 LLM tool_use 调用。
各工具实现在 tools/external.py (外部查询) 和 core/tools/self_config.py (自我配置)。
"""
from __future__ import annotations

from typing import Any, Callable

# { name: (schema_dict, async_callable) }
_REGISTRY: dict[str, tuple[dict, Callable]] = {}


def register(schema: dict, fn: Callable) -> None:
    _REGISTRY[schema["name"]] = (schema, fn)


def get_schemas() -> list[dict]:
    return [s for s, _ in _REGISTRY.values()]


async def call(name: str, **kwargs: Any) -> Any:
    if name not in _REGISTRY:
        raise KeyError(f"工具 '{name}' 未注册")
    _, fn = _REGISTRY[name]
    return await fn(**kwargs)
