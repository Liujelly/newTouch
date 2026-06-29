"""工具元数据表（catalog）—— 工具页的数据源。

registry.get_schemas() 只返回**已注册**的工具（启动时按开关注册的），关掉的工具
不在其中。但工具页要展示**所有**工具（含关闭的）让用户能重新打开，所以需要一个
独立于注册状态的元数据表，列出每个工具的名字/说明/默认开关/来源。

各 register_* 函数统一读 config 的 ``tools.<name>`` 决定是否注册（默认值取自此表）。
旧开关 memory.tool_enabled / web_search.enabled 保留作向后兼容回退（tools.<name>
未显式设置时回退旧值），避免已配用户静默失效。
"""
from __future__ import annotations

# name: 工具名（registry key）
# label: 给 UI 展示的中文短说明（区别于给 LLM 的 schema description）
# default: tools.<name> 未设置时的默认开关
# source: 来源分类，工具页分组展示用
TOOL_CATALOG: list[dict] = [
    {"name": "get_weather", "label": "查询天气", "default": True, "source": "外部查询"},
    {"name": "memory_search", "label": "检索长久记忆", "default": True, "source": "记忆"},
    {"name": "web_search", "label": "联网搜索实时信息", "default": False, "source": "联网搜索"},
    {"name": "look", "label": "看一眼摄像头画面", "default": True, "source": "视觉"},
    {"name": "add_schedule", "label": "新建提醒/待办", "default": True, "source": "日程"},
    {"name": "list_schedules", "label": "查看待办日程", "default": True, "source": "日程"},
    {"name": "mark_done", "label": "标记日程完成", "default": True, "source": "日程"},
    {"name": "update_schedule", "label": "修改日程", "default": True, "source": "日程"},
    {"name": "delete_schedule", "label": "删除日程", "default": True, "source": "日程"},
    {"name": "set_speaking_frequency", "label": "调整主动说话频率", "default": True, "source": "自我配置"},
    {"name": "toggle_vision", "label": "开关视觉感知", "default": True, "source": "自我配置"},
    {"name": "switch_preset", "label": "切换预设配置", "default": True, "source": "自我配置"},
    {"name": "get_my_status", "label": "查看自己状态", "default": True, "source": "自我配置"},
]

# 旧开关 → 新 tools.<name> 的回退映射（向后兼容）
_LEGACY_FALLBACK = {
    "memory_search": "memory.tool_enabled",
    "web_search": "web_search.enabled",
}


def get_tool_default(name: str) -> bool:
    """工具默认开关（catalog 里查；未知工具默认 True）。"""
    for t in TOOL_CATALOG:
        if t["name"] == name:
            return t["default"]
    return True


def is_tool_enabled(config, name: str) -> bool:
    """读 tools.<name> 决定工具是否注册。

    tools.<name> 未显式设置（None）时：catalog 默认值；若有旧开关回退映射则用旧值
    （向后兼容已配 memory.tool_enabled / web_search.enabled 的用户）。
    """
    val = config.get(f"tools.{name}")
    if val is not None:
        return bool(val)
    legacy = _LEGACY_FALLBACK.get(name)
    if legacy is not None:
        legacy_val = config.get(legacy)
        if legacy_val is not None:
            return bool(legacy_val)
    return get_tool_default(name)


def catalog_summary() -> list[dict]:
    """返回 catalog 的浅拷贝列表（供 API 序列化）。"""
    return [dict(t) for t in TOOL_CATALOG]
