"""记忆检索工具 (架构文档 5.10 / 8.x)。

把长期记忆检索做成 LLM 可自主调用的工具 memory_search：
  - 反应路径：LLM 觉得 system 里自动注入的 [相关记忆] 不够时，可主动再查一次
    （用自己想的 query，比"用 user_text 原话"质量高）。
  - 与"反应路径自动 recall 注入"（memory.reactive_auto_recall）是两套独立机制，
    各有开关，可叠加（兜底+增强）也可互替（纯工具 / 纯自动注入）。

注册：main.py 启动时调 register_memory_tools(memory_store, config)，
把工具连同 store 闭包注册进 tools.registry，供 tool_use 调用。
memory_store=None / 记忆未启用 / tools.memory_search=false 时不注册（空操作）。

工具内部固定 limit=3（和自动注入一致），不暴露给 LLM 防乱刷检索。
v2.58 开关从 memory.tool_enabled 迁移到 tools.memory_search（旧值保留作兼容回退）。
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from ..config import Config
from ..logger import get_logger
from .catalog import is_tool_enabled
from .registry import register

if TYPE_CHECKING:
    from ..memory.store import MemoryStore

log = get_logger("memory_tool")


_SCHEMA = {
    "name": "memory_search",
    "description": (
        "检索关于用户的长久记忆。当你需要回忆用户的偏好/习惯/过往经历/约定/计划，"
        "而当前对话上下文里没有足够信息时调用。query 用你想查的角度（如"
        "“忌口”“最近安排”“宠物”“工作”），不必照搬用户原话。"
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "检索关键词或角度，如“忌口”“最近安排”"},
        },
        "required": ["query"],
    },
}


def register_memory_tools(memory_store: "MemoryStore | None", config: Config | None = None) -> None:
    """把 memory_search 工具注册进 registry。

    memory_store 为 None（记忆禁用）/ store 未启用 / tools.memory_search=false
    时不注册。开关启动时读一次（改后需重启，与 LLM/STT 等“需重启”字段一致——工具增删低频）。
    """
    if memory_store is None:
        return
    if config is not None and not is_tool_enabled(config, "memory_search"):
        log.info("tools.memory_search=false，不注册 memory_search 工具")
        return
    if not getattr(memory_store, "_enabled", False):
        log.info("记忆未启用，不注册 memory_search 工具")
        return

    async def _memory_search(query: str) -> str:
        """检索关于用户的长久记忆，返回命中的记忆条目。

        当你需要回忆用户的偏好/习惯/过往/约定/计划，而当前对话上下文里没有
        相关信息时调用。query 用你想查的角度（如“忌口”“最近安排”“宠物”），不必照搬用户原话。
        """
        if not query or not query.strip():
            return "请提供要检索的关键词"
        try:
            hits = memory_store.recall(query.strip(), limit=3)
        except Exception as e:  # noqa: BLE001
            log.warning("memory_search 执行失败: %s", e)
            return "记忆检索失败"
        if not hits:
            return "没有找到相关记忆"
        return "找到以下相关记忆：\n" + "\n".join(f"- {h}" for h in hits)

    register(_SCHEMA, _memory_search)
    log.info("已注册 memory_search 工具")
