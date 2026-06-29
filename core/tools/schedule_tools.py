"""日程管理工具（架构文档 14.5）。

ta 通过这组白名单工具管理日程（需 ai_permissions.allow_manage_schedules=true）：
  - add_schedule     新建提醒（"你说要早睡，我帮你设个 11 点提醒"）
  - list_schedules   查看待办（可在主动循环里"回忆今天有什么事"）
  - mark_done        完成后标记（不删，保留）
  - update_schedule  改时间/内容
  - delete_schedule  删除

与查天气同一套 tool_use 机制（注册进 registry，反应路径 LLM 直接 call）。
失败返回错误串不抛异常，让 LLM 知道被拒绝/出错能换说法。

注册：main.py 启动时调 register_schedule_tools(store, config)，开关 tools.*
（默认 true，与 self_config 工具一致——注册开关控制是否暴露，权限控制能否调用）。
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import TYPE_CHECKING

from ..config import Config
from ..logger import get_logger
from .catalog import is_tool_enabled
from .registry import register

if TYPE_CHECKING:
    from ..perception.schedule import ScheduleStore

log = get_logger("schedule_tool")


_SCHEMAS = {
    "add_schedule": {
        "name": "add_schedule",
        "description": (
            "新建一条提醒/待办，到点你会主动提醒用户。需用户已授权管理日程。"
            "content=提醒内容；trigger_at=触发时间，用 ISO（2026-06-29T19:00:00）"
            "或自然语言（明天10点/晚上7点/后天9点）；repeat=none(单次)/daily(每天)。"
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "content": {"type": "string", "description": "提醒内容，如「该吃药了」「10点开会」"},
                "trigger_at": {"type": "string", "description": "触发时间，ISO 或自然语言"},
                "repeat": {"type": "string", "enum": ["none", "daily"], "description": "none=单次 / daily=每天"},
            },
            "required": ["content", "trigger_at"],
        },
    },
    "list_schedules": {
        "name": "list_schedules",
        "description": "查看待办日程。返回未完成的日程列表（内容/触发时间/重复）。可选 date 筛某天。",
        "input_schema": {
            "type": "object",
            "properties": {
                "date": {"type": "string", "description": "可选，筛选某天（ISO 日期如 2026-06-29），不传则全部未完成"},
            },
            "required": [],
        },
    },
    "mark_done": {
        "name": "mark_done",
        "description": "标记一条日程为已完成（不删除，保留记录）。",
        "input_schema": {
            "type": "object",
            "properties": {"id": {"type": "string", "description": "日程 id"}},
            "required": ["id"],
        },
    },
    "update_schedule": {
        "name": "update_schedule",
        "description": "修改一条日程的内容/触发时间/重复方式。",
        "input_schema": {
            "type": "object",
            "properties": {
                "id": {"type": "string"},
                "content": {"type": "string", "description": "新内容（可选）"},
                "trigger_at": {"type": "string", "description": "新触发时间（可选）"},
                "repeat": {"type": "string", "enum": ["none", "daily"], "description": "新重复方式（可选）"},
            },
            "required": ["id"],
        },
    },
    "delete_schedule": {
        "name": "delete_schedule",
        "description": "删除一条日程。",
        "input_schema": {
            "type": "object",
            "properties": {"id": {"type": "string"}},
            "required": ["id"],
        },
    },
}


def register_schedule_tools(store: "ScheduleStore", config: "Config | None" = None) -> None:
    """注册日程管理工具。每个工具各自读 tools.<name> 开关（默认 true）。"""
    if store is None:
        return

    audit_path = Path(str(config.project_root)) / "data" / "audit.log" if config is not None else None

    def _audit(tool: str, args: dict, result: str) -> None:
        if audit_path is None:
            return
        entry = {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"), "tool": tool, "args": args, "result": result}
        try:
            audit_path.parent.mkdir(parents=True, exist_ok=True)
            with open(audit_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception:  # noqa: BLE001
            pass

    def _has_perm() -> bool:
        return bool(config.get("ai_permissions.allow_manage_schedules", False)) if config is not None else False

    # add_schedule
    if config is None or is_tool_enabled(config, "add_schedule"):
        async def _add_schedule(content: str, trigger_at: str, repeat: str = "none") -> str:
            args = {"content": content, "trigger_at": trigger_at, "repeat": repeat}
            if not _has_perm():
                msg = "拒绝：未获授权管理日程（allow_manage_schedules=false）"
                _audit("add_schedule", args, msg)
                return msg
            item = store.add(content, trigger_at, repeat=repeat, context=content)
            if item is None:
                msg = f"失败：无法解析时间「{trigger_at}」，请用 ISO（2026-06-29T19:00:00）或「明天10点」「晚上7点」"
                _audit("add_schedule", args, msg)
                return msg
            _audit("add_schedule", args, "OK")
            return f"OK：已设提醒「{content}」，时间 {item.trigger_at}（{'每天' if repeat == 'daily' else '单次'}）"
        register(_SCHEMAS["add_schedule"], _add_schedule)

    # list_schedules
    if config is None or is_tool_enabled(config, "list_schedules"):
        async def _list_schedules(date: str = "") -> str:
            args = {"date": date}
            if not _has_perm():
                msg = "拒绝：未获授权管理日程"
                _audit("list_schedules", args, msg)
                return msg
            items = store.list_all(include_done=False)
            if date:
                items = [i for i in items if i.trigger_at.startswith(date[:10])]
            if not items:
                return "当前没有待办日程"
            lines = [f"共 {len(items)} 条待办："]
            for i in items:
                rep = "每天" if i.repeat == "daily" else "单次"
                lines.append(f"- [{i.id[:8]}] {i.content} @ {i.trigger_at} ({rep})")
            return "\n".join(lines)
        register(_SCHEMAS["list_schedules"], _list_schedules)

    # mark_done
    if config is None or is_tool_enabled(config, "mark_done"):
        async def _mark_done(id: str) -> str:
            args = {"id": id}
            if not _has_perm():
                msg = "拒绝：未获授权管理日程"
                _audit("mark_done", args, msg)
                return msg
            if store.mark_done(id):
                _audit("mark_done", args, "OK")
                return f"OK：日程 {id[:8]} 已标记完成"
            return f"失败：找不到日程 {id[:8]}"
        register(_SCHEMAS["mark_done"], _mark_done)

    # update_schedule
    if config is None or is_tool_enabled(config, "update_schedule"):
        async def _update_schedule(id: str, content: str | None = None,
                                    trigger_at: str | None = None, repeat: str | None = None) -> str:
            args = {"id": id, "content": content, "trigger_at": trigger_at, "repeat": repeat}
            if not _has_perm():
                msg = "拒绝：未获授权管理日程"
                _audit("update_schedule", args, msg)
                return msg
            if trigger_at:
                # 预检时间格式
                from ..perception.schedule import parse_trigger_at
                if not parse_trigger_at(trigger_at):
                    return f"失败：无法解析时间「{trigger_at}」"
            if store.update(id, content, trigger_at, repeat):
                _audit("update_schedule", args, "OK")
                return f"OK：日程 {id[:8]} 已更新"
            return f"失败：找不到日程 {id[:8]}"
        register(_SCHEMAS["update_schedule"], _update_schedule)

    # delete_schedule
    if config is None or is_tool_enabled(config, "delete_schedule"):
        async def _delete_schedule(id: str) -> str:
            args = {"id": id}
            if not _has_perm():
                msg = "拒绝：未获授权管理日程"
                _audit("delete_schedule", args, msg)
                return msg
            if store.delete(id):
                _audit("delete_schedule", args, "OK")
                return f"OK：日程 {id[:8]} 已删除"
            return f"失败：找不到日程 {id[:8]}"
        register(_SCHEMAS["delete_schedule"], _delete_schedule)

    log.info("已注册日程管理工具")
