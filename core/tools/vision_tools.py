"""视觉查看工具 look（v2.59：把"看一眼"做成 LLM 自主调用的工具）。

此前反应路径靠 _maybe_look 关键词预筛（"看看"/"看一下"...）+ 主动路径意图判断决定
要不要抓帧，问题：关键词太宽，"看天气"/"看新闻"这种信息查询被误判成"用摄像头看"，
劫持到视觉路径，本该调 get_weather/web_search 的查询走偏。

改成工具后：反应路径不再关键词拦截，直接走 react_stream，LLM 拿到 look 工具自主
决定调不调——问"看看我"→ 调 look 抓帧；问"看天气"→ 调 get_weather，互不干扰。
和 web_search/memory_search 同范式（LLM 自主查信息）。

注册：main.py 启动时调 register_vision_tools(vision, config)，开关 tools.look（默认 true）。
vision=None / 视觉未开（perception.vision.enabled=false）时工具仍可注册但调用返回
提示（让 LLM 知道"看不了"而非默默失败）。look_now 自带 min_look_interval 节流。

不额外要 ai_permissions 授权：look 是只读抓一帧（不像 toggle_vision 改持续运行状态），
视觉总开关 perception.vision.enabled 已把关（关了 look 调了也是空）。
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from ..config import Config
from ..logger import get_logger
from .catalog import is_tool_enabled
from .registry import register

if TYPE_CHECKING:
    from ..perception.vision import Vision

log = get_logger("vision_tool")

_SCHEMA = {
    "name": "look",
    "description": (
        "看一眼摄像头当前的画面。当用户要求你看看 ta、看看周围环境、确认眼前发生了什么"
        "（如“看看我”“你看我这边”“打开摄像头看看”）时调用，返回对当前画面的描述。"
        "注意：查天气/查新闻/查资料等是查信息，应调 get_weather 或 web_search，不是 look。"
    ),
    "input_schema": {
        "type": "object",
        "properties": {},
        "required": [],
    },
}


def register_vision_tools(vision: "Vision | None", config: "Config | None" = None) -> None:
    """把 look 工具注册进 registry。开关 tools.look（默认 true，restart）。

    vision=None（视觉模块未构造）时不注册。视觉未开（perception.vision.enabled=false）
    时仍注册，调用返回"看不了"提示——让 LLM 知道当前无视觉，别反复试。
    """
    if vision is None:
        return
    if config is not None and not is_tool_enabled(config, "look"):
        return

    async def _look() -> str:
        """看一眼摄像头当前画面，返回画面描述。

        当用户要求你看看 ta、看看周围、确认眼前发生了什么时调用。
        查天气/查新闻/查资料请用 get_weather 或 web_search，不是本工具。
        """
        if not vision._enabled:
            return "当前视觉未开启，看不了画面。如需开启，请让用户在管理平台打开视觉或授权你开关视觉。"
        try:
            vc = await vision.look_now()
        except Exception as e:  # noqa: BLE001
            log.warning("look 工具调用失败: %s", e)
            return f"看一眼失败: {type(e).__name__}"
        if not vc or not vc.caption:
            return "看了一眼，但没识别到画面内容"
        return f"你看到了：{vc.caption}"

    register(_SCHEMA, _look)
    log.info("已注册 look 工具")
