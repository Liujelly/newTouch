"""测试反应路径工具使用引导（v2.59）。

验证：
1. build_reactive_prompt 的 system 含工具引导（"工具使用"段），且动态反映已注册工具
2. build_proactive_prompt 的 system 不含工具引导（主动路径走 action 机制，不需要）
3. 无工具注册时引导段不出现（不污染 prompt）

不触网、不调 LLM，只验 prompt 字符串。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8")

from core.character import CharacterCard, build_reactive_prompt, build_proactive_prompt
from core.tools import registry


def _cleanup():
    """测试用的临时工具用完即删，避免污染其他测试。"""
    for name in ("_test_tool_a", "_test_tool_b"):
        if name in registry._REGISTRY:
            del registry._REGISTRY[name]


def _card():
    return CharacterCard(name="测试角色")


def test_reactive_has_tool_guidance():
    """反应路径 system 含工具引导段，且含已注册工具名。"""
    async def _a(): return "ok"
    registry.register({"name": "_test_tool_a", "description": "测试工具A",
                       "input_schema": {"type": "object", "properties": {}, "required": []}}, _a)
    try:
        sys_prompt, _ = build_reactive_prompt(
            _card(), "用户", "看看我", chat_history=[])
        assert "工具使用" in sys_prompt, "反应路径 system 应含工具使用引导"
        assert "_test_tool_a" in sys_prompt, "引导应含已注册工具名"
        assert "不要凭记忆编" in sys_prompt or "过去" in sys_prompt, \
            "应明确告诫历史信息是过去的"
        print("✅ 反应路径 system 含工具引导 + 工具名 + 反编告诫")
    finally:
        _cleanup()


def test_reactive_look_strong_guidance():
    """注册 look 时引导含「必须调 look」强措辞 + 提及自动抓取画面。"""
    async def _look(): return "ok"
    registry.register({"name": "look", "description": "看一眼画面",
                       "input_schema": {"type": "object", "properties": {}, "required": []}}, _look)
    try:
        sys_prompt, _ = build_reactive_prompt(
            _card(), "用户", "看看我", chat_history=[])
        assert "必须调 look" in sys_prompt, "注册 look 时应强措辞要求调 look"
        assert "自动抓取" in sys_prompt, "应提及历史里的自动抓取画面可能过时"
        print("✅ 注册 look 时引导含「必须调 look」+ 提及自动抓取画面")
    finally:
        _cleanup()


def test_proactive_no_tool_guidance():
    """主动路径 system 不含工具引导。"""
    async def _b(): return "ok"
    registry.register({"name": "_test_tool_b", "description": "测试工具B",
                       "input_schema": {"type": "object", "properties": {}, "required": []}}, _b)
    try:
        sys_prompt, _ = build_proactive_prompt(
            _card(), "用户", trigger_reason="心跳", emotion_summary="平静",
            chat_history=[], elapsed_desc="刚刚", can_look=False, reply_lang="zh")
        assert "工具使用" not in sys_prompt, "主动路径不应含工具引导（走 action 机制）"
        assert "_test_tool_b" not in sys_prompt, "主动路径不应列工具名"
        print("✅ 主动路径 system 不含工具引导")
    finally:
        _cleanup()


def test_no_tools_no_guidance():
    """无工具注册时引导段不出现。"""
    # 清空 registry（记下原内容事后还原）
    saved = dict(registry._REGISTRY)
    try:
        registry._REGISTRY.clear()
        sys_prompt, _ = build_reactive_prompt(
            _card(), "用户", "你好", chat_history=[])
        assert "工具使用" not in sys_prompt, "无工具时不应有引导段"
        print("✅ 无工具注册时不污染 prompt")
    finally:
        registry._REGISTRY.clear()
        registry._REGISTRY.update(saved)


if __name__ == "__main__":
    print("=== 测试反应路径工具使用引导 ===\n")
    test_reactive_has_tool_guidance()
    test_reactive_look_strong_guidance()
    test_proactive_no_tool_guidance()
    test_no_tools_no_guidance()
    _cleanup()
    print("\n=== 全部测试通过 ===")
