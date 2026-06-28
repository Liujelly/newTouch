"""测试工具统一开关 catalog（v2.58）。

验证 is_tool_enabled 的优先级与向后兼容回退：
  tools.<name> 显式设置 > 旧开关回退（memory.tool_enabled / web_search.enabled）> catalog 默认值

不触网、不依赖 LLM。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8")

from core.config import load_config
from core.tools.catalog import (
    TOOL_CATALOG,
    catalog_summary,
    get_tool_default,
    is_tool_enabled,
)


def test_catalog_completeness():
    """catalog 覆盖所有已知工具，字段齐全。"""
    names = {t["name"] for t in TOOL_CATALOG}
    expected = {"get_weather", "memory_search", "web_search", "look",
                "set_speaking_frequency", "toggle_vision", "switch_preset", "get_my_status"}
    assert names == expected, f"catalog 工具集不符：缺 {expected - names}，多 {names - expected}"
    for t in TOOL_CATALOG:
        assert "label" in t and "default" in t and "source" in t
    # catalog_summary 返回浅拷贝
    s = catalog_summary()
    assert s[0] is not TOOL_CATALOG[0], "summary 应是浅拷贝"
    print("✅ catalog 覆盖 8 个工具、字段齐全、summary 浅拷贝")


def test_default_values():
    """catalog 默认值：web_search 默认 false，其余默认 true。"""
    assert get_tool_default("web_search") is False
    assert get_tool_default("get_weather") is True
    assert get_tool_default("memory_search") is True
    assert get_tool_default("get_my_status") is True
    assert get_tool_default("unknown_tool") is True  # 未知工具默认 True
    print("✅ 默认值正确（web_search=false，其余=true，未知=true）")


def test_priority_tools_over_legacy():
    """tools.<name> 显式设置优先于旧开关。"""
    cfg = load_config()
    cfg.set("tools.memory_search", False)
    cfg.set("memory.tool_enabled", True)  # 旧开关 true，但新开关 false 应优先
    assert is_tool_enabled(cfg, "memory_search") is False
    print("✅ tools.<name> 优先于旧开关")

    cfg.set("tools.web_search", True)
    cfg.set("web_search.enabled", False)  # 旧开关 false，新开关 true 优先
    assert is_tool_enabled(cfg, "web_search") is True
    print("✅ tools.web_search=true 优先于 web_search.enabled=false")


def test_legacy_fallback_when_tools_unset():
    """tools.<name> 未设时回退旧开关。"""
    cfg = load_config()
    cfg.set("tools.memory_search", None)
    cfg.set("memory.tool_enabled", False)
    assert is_tool_enabled(cfg, "memory_search") is False
    print("✅ tools 未设时回退 memory.tool_enabled=false")

    cfg.set("tools.web_search", None)
    cfg.set("web_search.enabled", True)
    assert is_tool_enabled(cfg, "web_search") is True
    print("✅ tools 未设时回退 web_search.enabled=true")


def test_catalog_default_when_both_unset():
    """tools 和旧开关都没设时用 catalog 默认值。"""
    cfg = load_config()
    cfg.set("tools.web_search", None)
    cfg.set("web_search.enabled", None)
    assert is_tool_enabled(cfg, "web_search") is False  # catalog 默认 false
    cfg.set("tools.get_weather", None)
    assert is_tool_enabled(cfg, "get_weather") is True  # catalog 默认 true
    print("✅ 都没设时用 catalog 默认值")


if __name__ == "__main__":
    print("=== 测试工具统一开关 catalog ===\n")
    test_catalog_completeness()
    test_default_values()
    test_priority_tools_over_legacy()
    test_legacy_fallback_when_tools_unset()
    test_catalog_default_when_both_unset()
    print("\n=== 全部测试通过 ===")
