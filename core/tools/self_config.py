"""AI 白名单自我配置工具 (架构文档 8.3)。

ta（AI）只能通过这一组预定义工具改一个**受限子集**：
  - set_speaking_frequency  调主动说话频率档位（枚举，不能直接改秒数）
  - toggle_vision           开关视觉感知（需用户授权）
  - switch_preset           切换预设（需用户授权 + 文件必须存在）
  - get_my_status           只读：自己的情绪/权限

  注：select_voice（选音色）已移除——音色=切 GPT-SoVITS 模型，与「按情绪选参考音频」
  是两回事，待 GPT-SoVITS 多模型切换做好后再加。情绪参考音频由 speak 按 <emo:> 标签
  自动选（见 action/speak.py），不走 AI 工具。

安全边界（与用户全权的管理平台分开）：
  - 绝不触碰 API key / 安全约束 / 权限授权 / shell。
  - 校验失败**返回错误信息字符串**而不是抛异常——让 LLM 知道被拒绝了，能换个说法。
  - 所有操作写 data/audit.log（{timestamp, tool, args, result}）。
  - 改动只落到白名单子集，持久化到 data/self_config.json（绝不回写含 ${ENV} 的 config.yaml）。

注册：main.py 启动时调 register_self_config_tools(config, state, refreshers)，
把工具连同 config/state 闭包注册进 tools.registry，供反应路径 tool_use 调用。
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable

from ..config import Config
from ..state import EmotionState
from ..logger import get_logger
from .registry import register

log = get_logger("self_config")

# 频率档位（枚举 → 具体参数），ta 只能选档位，碰不到秒数本身
FREQUENCY_PRESETS: dict[str, dict[str, int]] = {
    "quiet": {"min_interval_seconds": 1800, "hourly_cap": 2},
    "normal": {"min_interval_seconds": 600, "hourly_cap": 5},
    "chatty": {"min_interval_seconds": 300, "hourly_cap": 10},
}

# 允许 AI 持久化的白名单点路径（防止任何其他键被写进 overrides 文件）
_PERSIST_WHITELIST = {
    "proactive.min_interval_seconds",
    "proactive.hourly_cap",
    "perception.vision.enabled",
    "character.current_preset",
}


class SelfConfig:
    """绑定 config/state，提供白名单操作 + 审计 + 持久化。"""

    def __init__(
        self,
        config: Config,
        state: EmotionState,
        refreshers: dict[str, Callable] | None = None,
    ):
        self._cfg = config
        self._state = state
        # 运行时刷新钩子：改配置后让活着的模块实例同步（gatekeeper/vision）
        # refreshers = {"gatekeeper": fn, "vision": fn}，fn 接收新 config
        self._refreshers = refreshers or {}
        root = config.project_root
        self._audit_path = config.char_data_dir() / "audit.log"
        self._overrides_path = root / "data" / "self_config.json"
        self._presets_dir = root / "data" / "presets"

    # ── 审计 + 持久化 ──────────────────────────────────────────
    def _audit(self, tool: str, args: dict, result: str) -> None:
        entry = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "tool": tool,
            "args": args,
            "result": result,
        }
        self._audit_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._audit_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def _persist(self, dotted: str) -> None:
        """把当前内存值落盘到 self_config.json（仅白名单键）。"""
        if dotted not in _PERSIST_WHITELIST:
            return
        data: dict[str, Any] = {}
        if self._overrides_path.exists():
            try:
                data = json.loads(self._overrides_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, ValueError):
                data = {}
        data[dotted] = self._cfg.get(dotted)
        self._overrides_path.parent.mkdir(parents=True, exist_ok=True)
        self._overrides_path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def _check_perm(self, key: str) -> bool:
        return bool(self._cfg.get(f"ai_permissions.{key}", False))

    def _refresh(self, name: str) -> None:
        fn = self._refreshers.get(name)
        if fn:
            try:
                fn(self._cfg)
            except Exception as e:  # noqa: BLE001
                log.warning("刷新 %s 失败: %s", name, e)

    # ── 工具实现（失败返回错误串，不抛异常）────────────────────
    async def set_speaking_frequency(self, level: str) -> str:
        """调主动说话频率档位（quiet/normal/chatty）。需授权 allow_adjust_frequency。"""
        args = {"level": level}
        if not self._check_perm("allow_adjust_frequency"):
            msg = "拒绝：未获授权调整说话频率（allow_adjust_frequency=false）"
            self._audit("set_speaking_frequency", args, msg)
            return msg
        if level not in FREQUENCY_PRESETS:
            msg = f"拒绝：档位须为 {list(FREQUENCY_PRESETS)} 之一"
            self._audit("set_speaking_frequency", args, msg)
            return msg
        preset = FREQUENCY_PRESETS[level]
        for k, v in preset.items():
            self._cfg.set(f"proactive.{k}", v)
            self._persist(f"proactive.{k}")
        self._refresh("gatekeeper")
        self._audit("set_speaking_frequency", args, "OK")
        return f"OK：说话频率已设为 {level}（间隔{preset['min_interval_seconds']}s，每小时上限{preset['hourly_cap']}）"

    async def toggle_vision(self, enabled: bool) -> str:
        """开关视觉感知。需用户已授权 allow_toggle_vision。"""
        args = {"enabled": enabled}
        if not self._check_perm("allow_toggle_vision"):
            msg = "拒绝：未获授权开关视觉（allow_toggle_vision=false）"
            self._audit("toggle_vision", args, msg)
            return msg
        self._cfg.set("perception.vision.enabled", bool(enabled))
        self._persist("perception.vision.enabled")
        self._refresh("vision")
        self._audit("toggle_vision", args, "OK")
        return f"OK：视觉感知已{'开启' if enabled else '关闭'}"

    async def switch_preset(self, name: str) -> str:
        """切换预设。需授权 allow_switch_preset，且 name 必须在 data/presets/ 存在。"""
        args = {"name": name}
        if not self._check_perm("allow_switch_preset"):
            msg = "拒绝：未获授权切换预设（allow_switch_preset=false）"
            self._audit("switch_preset", args, msg)
            return msg
        if "/" in name or "\\" in name or ".." in name:
            msg = "拒绝：预设名不能包含路径分隔符"
            self._audit("switch_preset", args, msg)
            return msg
        if not (self._presets_dir / f"{name}.json").exists():
            msg = f"拒绝：预设 '{name}' 不存在于 data/presets/"
            self._audit("switch_preset", args, msg)
            return msg
        self._cfg.set("character.current_preset", name)
        self._persist("character.current_preset")
        self._audit("switch_preset", args, "OK")
        return f"OK：已切换预设为 {name}"

    async def get_my_status(self) -> str:
        """只读：自己当前情绪状态 + 权限（无需权限检查）。"""
        status = {
            "emotion": self._state.snapshot(),
            "permissions": self._cfg.get("ai_permissions", {}),
            "vision_enabled": self._cfg.get("perception.vision.enabled", False),
            "speaking_interval_s": self._cfg.get("proactive.min_interval_seconds"),
        }
        return json.dumps(status, ensure_ascii=False)


# ── 工具 JSON Schema ────────────────────────────────────────────
_SCHEMAS = {
    "set_speaking_frequency": {
        "name": "set_speaking_frequency",
        "description": "调整自己主动说话的频率档位。需用户授权。",
        "input_schema": {
            "type": "object",
            "properties": {"level": {"type": "string", "enum": ["quiet", "normal", "chatty"],
                                     "description": "quiet=话少 normal=适中 chatty=话多"}},
            "required": ["level"],
        },
    },
    "toggle_vision": {
        "name": "toggle_vision",
        "description": "开启或关闭自己的视觉感知（摄像头）。需用户授权。",
        "input_schema": {
            "type": "object",
            "properties": {"enabled": {"type": "boolean", "description": "true 开启 / false 关闭"}},
            "required": ["enabled"],
        },
    },
    "switch_preset": {
        "name": "switch_preset",
        "description": "切换到另一个预设配置。需用户授权，且预设须已存在。",
        "input_schema": {
            "type": "object",
            "properties": {"name": {"type": "string", "description": "预设名（不含 .json）"}},
            "required": ["name"],
        },
    },
    "get_my_status": {
        "name": "get_my_status",
        "description": "查看自己当前的情绪状态、权限和音色设置（只读）。",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
}


def register_self_config_tools(
    config: Config,
    state: EmotionState,
    refreshers: dict[str, Callable] | None = None,
) -> SelfConfig:
    """构造 SelfConfig 并把白名单工具注册进 registry。返回实例供测试/复用。

    每个工具各自读 tools.<name> 开关（默认 true，restart）。
    注意：注册开关（tools.<name>）控制工具是否暴露给 LLM；ai_permissions 运行时
    权限控制调用时是否放行——两层独立，可叠加（注册了但权限关 = LLM 看得到调不动）。
    """
    from .catalog import is_tool_enabled
    sc = SelfConfig(config, state, refreshers)
    _to_register = [
        ("set_speaking_frequency", sc.set_speaking_frequency),
        ("toggle_vision", sc.toggle_vision),
        ("switch_preset", sc.switch_preset),
        ("get_my_status", sc.get_my_status),
    ]
    for name, fn in _to_register:
        if is_tool_enabled(config, name):
            register(_SCHEMAS[name], fn)
    return sc
