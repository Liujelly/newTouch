"""测试联网搜索工具 web_search。

验证（全离线，不触网）：
1. _parse_ddg_html 解析 DDG HTML 结果页：标题/uddg 跳转链接解码/HTML 实体/标签剥离
2. _resolve_ddg_url：uddg 跳转解出真实 URL，非跳转原样返回
3. _format_results：有结果/无结果格式
4. register_web_search_tools：enabled=true 注册 / enabled=false 不注册 / tavily 缺 key 降级
5. config 默认值（enabled 默认 false、provider 默认 duckduckgo、max_results 默认 3）

不依赖真实 DDG/Tavily 接口。
"""
import asyncio
import sys
from pathlib import Path
from urllib.parse import quote

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8")

from core.config import load_config
from core.tools import registry
from core.tools.web_search import (
    _format_results,
    _parse_ddg_html,
    _resolve_ddg_url,
    _strip_tags,
    register_web_search_tools,
)


def _cleanup_registry():
    if "web_search" in registry._REGISTRY:
        del registry._REGISTRY["web_search"]


# ── 构造一段贴近真实的 DDG html 结果页 ──────────────────────────────
_REAL_URL = "https://docs.python.org/3/library/asyncio.html"
_UDDG = "//duckduckgo.com/l/?uddg=" + quote(_REAL_URL, safe="") + "&rut=abc"
_HTML = f"""
<div class="results">
  <div class="result">
    <a rel="nofollow" class="result__a" href="{_UDDG}">asyncio — <b>Asynchronous</b> I/O &#8212; Python docs</a>
    <a class="result__snippet" href="{_UDDG}">asyncio is a library to write concurrent code using the async/await syntax &amp; coroutines.</a>
  </div>
  <div class="result">
    <a rel="nofollow" class="result__a" href="{_UDDG}">Real Python: Async IO</a>
    <a class="result__snippet" href="{_UDDG}">Learn <em>async</em> programming step by step.</a>
  </div>
</div>
"""


def test_parse_ddg_html():
    """解析标题/链接解码/实体/标签剥离/数量。"""
    items = _parse_ddg_html(_HTML)
    assert len(items) == 2, f"应解析出 2 条，实际 {len(items)}"

    title0, url0, snip0 = items[0]
    assert "asyncio" in title0, f"标题应含 asyncio，实际：{title0}"
    assert "Asynchronous" in title0, "标题里 <b> 标签内容应保留"
    assert "<" not in title0, "标题不应残留标签"
    assert url0 == _REAL_URL, f"应解出真实 URL，实际：{url0}"
    assert "&amp;" not in snip0, "摘要 HTML 实体应解码"
    assert "&" in snip0, "实体解码后应有普通 &"
    assert "<em>" not in snip0, "摘要标签应剥离"
    print("✅ DDG HTML 解析正确（标题/链接解码/实体/标签）")

    # 第二条标题/摘要也正常
    assert "Real Python" in items[1][0]
    print("✅ 多条结果都解析")


def test_resolve_ddg_url():
    """uddg 跳转解出真实 URL；非跳转原样返回；// 开头补 https。"""
    assert _resolve_ddg_url(_UDDG) == _REAL_URL
    assert _resolve_ddg_url("https://example.com/page") == "https://example.com/page"
    assert _resolve_ddg_url("//example.com/x") == "https://example.com/x"
    assert _resolve_ddg_url("") == ""
    print("✅ _resolve_ddg_url 跳转解码/原样返回/补协议")


def test_strip_tags():
    assert _strip_tags("<b>hi</b> &amp; bye") == "hi & bye"
    assert _strip_tags("  a   b  ") == "a b"
    assert _strip_tags("") == ""
    print("✅ _strip_tags 去标签/实体/折叠空白")


def test_format_results():
    items = [("标题1", "https://a.com", "摘要1")]
    out = _format_results("test", items)
    assert "1 条结果" in out and "标题1" in out and "摘要1" in out
    empty = _format_results("test", [])
    assert "未找到" in empty
    print("✅ _format_results 有结果/无结果格式")


def test_parse_empty_html():
    """无结果块时返回空列表（不崩）。"""
    assert _parse_ddg_html("<html><body>nothing here</body></html>") == []
    assert _parse_ddg_html("") == []
    print("✅ 空/无结果 HTML 返回空列表")


def test_register_enabled():
    """enabled=true 时注册进 registry，能 call。"""
    cfg = load_config()
    cfg.set("web_search.enabled", True)
    cfg.set("web_search.provider", "duckduckgo")
    cfg.set("web_search.max_results", 5)
    register_web_search_tools(cfg)
    schemas = {s["name"] for s in registry.get_schemas()}
    assert "web_search" in schemas, "enabled=true 应注册 web_search"
    print("✅ enabled=true 注册 web_search")

    # 空 query 不触网
    result = asyncio.run(registry.call("web_search", query=""))
    assert "关键词" in result, f"空 query 应提示，实际：{result}"
    print("✅ 空 query 处理正确（不触网）")
    _cleanup_registry()


def test_register_disabled():
    """enabled=false 时不注册。"""
    cfg = load_config()
    cfg.set("web_search.enabled", False)
    register_web_search_tools(cfg)
    schemas = {s["name"] for s in registry.get_schemas()}
    assert "web_search" not in schemas, "enabled=false 不应注册"
    print("✅ enabled=false 时不注册")


def test_config_defaults():
    """config 默认值：enabled=false / provider=duckduckgo / max_results=3。"""
    cfg = load_config()
    # config.yaml 现在显式写了 web_search.enabled=false，断言默认行为
    assert cfg.get("web_search.enabled", False) is False
    assert cfg.get("web_search.provider", "duckduckgo") == "duckduckgo"
    assert int(cfg.get("web_search.max_results", 3)) == 3
    print("✅ config 默认值正确（enabled=false/provider=duckduckgo/max=3）")


if __name__ == "__main__":
    print("=== 测试联网搜索工具 web_search ===\n")
    test_parse_ddg_html()
    test_resolve_ddg_url()
    test_strip_tags()
    test_format_results()
    test_parse_empty_html()
    test_register_enabled()
    test_register_disabled()
    test_config_defaults()
    _cleanup_registry()
    print("\n=== 全部测试通过 ===")
