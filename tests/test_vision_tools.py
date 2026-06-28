"""测试视觉查看工具 look（v2.59）。

验证（不触网、不抓真摄像头）：
1. register_vision_tools：vision=None 不注册；tools.look=false 不注册；正常注册
2. look 工具：视觉未开 → 返回"看不了"提示；look_now 抛异常 → 友好失败串；正常 → "你看到了："
3. 用 fake vision 模拟 look_now 返回值，不依赖真摄像头/VLM
"""
import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8")

from core.config import load_config
from core.tools import registry
from core.tools.vision_tools import register_vision_tools


def _cleanup():
    if "look" in registry._REGISTRY:
        del registry._REGISTRY["look"]


class _FakeVision:
    """假 vision：_enabled 可控，look_now 返回预设 caption 或抛异常。"""
    def __init__(self, enabled=True, caption="一只橘猫趴在键盘上", exc=None):
        self._enabled = enabled
        self._caption = caption
        self._exc = exc

    async def look_now(self):
        if self._exc:
            raise self._exc
        if self._caption is None:
            return None
        return SimpleNamespace(caption=self._caption)


def test_not_registered_when_none():
    """vision=None 不注册。"""
    cfg = load_config()
    cfg.set("tools.look", True)
    register_vision_tools(None, cfg)
    assert "look" not in {s["name"] for s in registry.get_schemas()}
    print("✅ vision=None 时不注册")


def test_not_registered_when_disabled():
    """tools.look=false 不注册。"""
    cfg = load_config()
    cfg.set("tools.look", False)
    register_vision_tools(_FakeVision(), cfg)
    assert "look" not in {s["name"] for s in registry.get_schemas()}
    print("✅ tools.look=false 时不注册")


def test_registered_and_call():
    """注册成功，look 工具返回 caption。"""
    cfg = load_config()
    cfg.set("tools.look", True)
    v = _FakeVision(enabled=True, caption="一只橘猫趴在键盘上")
    register_vision_tools(v, cfg)
    assert "look" in {s["name"] for s in registry.get_schemas()}
    out = asyncio.run(registry.call("look"))
    assert "你看到了" in out and "橘猫" in out, f"应返回 caption，实际：{out}"
    print("✅ look 工具注册并返回 caption")
    _cleanup()


def test_vision_disabled():
    """视觉未开 → 返回"看不了"提示，不调 look_now。"""
    cfg = load_config()
    cfg.set("tools.look", True)
    called = {"n": 0}
    class V(_FakeVision):
        async def look_now(self):
            called["n"] += 1
            return SimpleNamespace(caption="不该走到这")
    register_vision_tools(V(enabled=False), cfg)
    out = asyncio.run(registry.call("look"))
    assert "未开启" in out or "看不了" in out, f"应提示未开，实际：{out}"
    assert called["n"] == 0, "视觉关时不应调 look_now"
    print("✅ 视觉未开返回提示、不抓帧")
    _cleanup()


def test_look_now_exception():
    """look_now 抛异常 → 返回友好失败串，不崩。"""
    cfg = load_config()
    cfg.set("tools.look", True)
    register_vision_tools(_FakeVision(enabled=True, exc=RuntimeError("摄像头被占")), cfg)
    out = asyncio.run(registry.call("look"))
    assert "失败" in out, f"异常应返回失败串，实际：{out}"
    print("✅ look_now 异常返回友好失败串")
    _cleanup()


def test_look_now_none():
    """look_now 返回 None（没识别到）→ 提示。"""
    cfg = load_config()
    cfg.set("tools.look", True)
    register_vision_tools(_FakeVision(enabled=True, caption=None), cfg)
    out = asyncio.run(registry.call("look"))
    assert "没识别到" in out, f"None 应提示没识别到，实际：{out}"
    print("✅ look_now 返回 None 提示没识别到")
    _cleanup()


if __name__ == "__main__":
    print("=== 测试视觉查看工具 look ===\n")
    test_not_registered_when_none()
    test_not_registered_when_disabled()
    test_registered_and_call()
    test_vision_disabled()
    test_look_now_exception()
    test_look_now_none()
    _cleanup()
    print("\n=== 全部测试通过 ===")
