"""联网搜索工具 web_search（架构文档早有设想，首次实现）。

把联网搜索做成 LLM 可自主调用的工具 web_search：
  - 用户问及实时信息（近期新闻/事件/最新版本号/实时数据/你不确定的客观事实）而
    模型训练数据不覆盖或可能过时时，LLM 自主决定查一下再答。
  - 与 get_weather（wttr.in 免费天气专用接口）互补：天气走 get_weather，其余实时
    查询走 web_search。

后端（web_search.provider）：
  - duckduckgo（默认，免费无需 key）：抓 html.duckduckgo.com/html/ 结果页，正则解析
    标题/链接/摘要。无额外依赖（复用 httpx，与 get_weather 同款）。
  - tavily（可选，需 web_search.api_key）：结果更干净，有免费额度。配了 key 即用。

注册：main.py 启动时调 register_web_search_tools(config)，开关 tools.web_search
（默认 false，改后需重启——工具增删低频，与 memory.tool_enabled 一致）。
provider/api_key/max_results 同为启动时读（restart）。
v2.58 开关从 web_search.enabled 迁移到 tools.web_search（旧值保留作兼容回退）。
"""
from __future__ import annotations

import html as _html
import re
from urllib.parse import parse_qs, unquote, urlparse

import httpx

from ..config import Config
from ..logger import get_logger
from .catalog import is_tool_enabled
from .registry import register

log = get_logger("web_search")

_SCHEMA = {
    "name": "web_search",
    "description": (
        "联网搜索实时信息。当用户问到可能超出你训练数据或会过时的事实"
        "（近期新闻/事件/最新版本/实时数据/你不确定的客观事实）时调用，"
        "返回若干条结果的标题、链接与摘要，据此回答。query 用搜索关键词。"
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "搜索关键词"},
        },
        "required": ["query"],
    },
}

_UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
       "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")


# ── DuckDuckGo HTML 结果页解析（纯函数，便于离线测试） ───────────────
_RESULT_RE = re.compile(
    r'class="result__a"[^>]*href="(?P<href>[^"]+)"[^>]*>(?P<title>.*?)</a>',
    re.DOTALL,
)
_SNIPPET_RE = re.compile(
    r'class="result__snippet"[^>]*>(?P<snippet>.*?)</a>',
    re.DOTALL,
)
_TAG_RE = re.compile(r"<[^>]+>")


def _strip_tags(s: str) -> str:
    """去标签 + HTML 实体解码 + 折叠空白。"""
    s = _TAG_RE.sub("", s)
    s = _html.unescape(s)
    return " ".join(s.split()).strip()


def _resolve_ddg_url(href: str) -> str:
    """DDG 结果链接是跳转 uddg=<真实URL>，解出来；非跳转原样返回。"""
    if not href:
        return ""
    if href.startswith("//"):
        href = "https:" + href
    if "uddg=" in href:
        qs = parse_qs(urlparse(href).query)
        u = qs.get("uddg", [""])[0]
        return unquote(u) if u else href
    return href


def _parse_ddg_html(text: str) -> list[tuple[str, str, str]]:
    """解析 DuckDuckGo html 结果页，返回 [(title, url, snippet), ...]。

    结果块顺序：先 result__a（标题+链接），其后跟着 result__snippet（摘要）。
    用 snippet 正则从标题匹配结束位置向后找最近一条配对。
    """
    results: list[tuple[str, str, str]] = []
    for m in _RESULT_RE.finditer(text):
        url = _resolve_ddg_url(m.group("href"))
        title = _strip_tags(m.group("title"))
        if not title and not url:
            continue
        # 从标题匹配结束位置向后找最近的摘要
        tail = text[m.end():]
        sm = _SNIPPET_RE.search(tail)
        snippet = _strip_tags(sm.group("snippet")) if sm else ""
        results.append((title, url, snippet))
    return results


async def _ddg_search(query: str, max_results: int) -> list[tuple[str, str, str]]:
    """DuckDuckGo HTML 端点搜索（免费无需 key）。"""
    url = "https://html.duckduckgo.com/html/"
    for attempt in range(3):
        try:
            async with httpx.AsyncClient(
                timeout=10.0, follow_redirects=True, headers={"User-Agent": _UA}
            ) as client:
                resp = await client.post(url, data={"q": query, "kl": "us-en"})
                resp.raise_for_status()
            items = _parse_ddg_html(resp.text)
            if items:
                return items[:max_results]
            return []  # 解析成功但无结果，不重试
        except httpx.TimeoutException:
            if attempt < 2:
                continue
            raise RuntimeError(f"搜索超时（已重试{attempt+1}次）")
        except httpx.HTTPStatusError as e:
            raise RuntimeError(f"搜索失败: HTTP {e.response.status_code}")
        except Exception as e:
            if attempt < 2:
                continue
            raise RuntimeError(f"搜索失败: {type(e).__name__}: {e}")
    return []


async def _tavily_search(query: str, max_results: int, api_key: str) -> list[tuple[str, str, str]]:
    """Tavily 搜索（需 api_key，结果更干净）。"""
    url = "https://api.tavily.com/search"
    async with httpx.AsyncClient(timeout=15.0, headers={"User-Agent": _UA}) as client:
        resp = await client.post(url, json={
            "api_key": api_key,
            "query": query,
            "max_results": max_results,
            "search_depth": "basic",
        })
        resp.raise_for_status()
        data = resp.json()
    out = []
    for r in data.get("results", []):
        out.append((
            str(r.get("title", "")).strip(),
            str(r.get("url", "")).strip(),
            str(r.get("content", "")).strip(),
        ))
    return out


def _format_results(query: str, items: list[tuple[str, str, str]]) -> str:
    if not items:
        return f"未找到与「{query}」相关的搜索结果"
    lines = [f"搜索「{query}」得到 {len(items)} 条结果："]
    for i, (title, url, snippet) in enumerate(items, 1):
        lines.append(f"\n[{i}] {title}")
        if url:
            lines.append(f"    链接: {url}")
        if snippet:
            lines.append(f"    摘要: {snippet}")
    return "\n".join(lines)


def register_web_search_tools(config: "Config | None" = None) -> None:
    """把 web_search 工具注册进 registry。

    tools.web_search=false（默认）时不注册。provider/api_key/max_results 启动时
    读一次（改后需重启，与 LLM/STT/memory.tool_enabled 等 restart 字段一致）。
    tavily 必须配 api_key，否则降级回 duckduckgo 并告警。
    """
    if config is None or not is_tool_enabled(config, "web_search"):
        return

    provider = config.get("web_search.provider", "duckduckgo") or "duckduckgo"
    max_results = int(config.get("web_search.max_results", 3) or 3)
    max_results = max(1, min(max_results, 10))
    api_key = config.get("web_search.api_key", "") or ""

    if provider == "tavily":
        if not api_key:
            log.warning("web_search.provider=tavily 但未配 api_key，降级用 duckduckgo")
            provider = "duckduckgo"

    async def _web_search(query: str) -> str:
        """联网搜索实时信息，返回若干条结果的标题/链接/摘要。

        当用户问到可能超出你训练数据或会过时的事实（近期新闻/事件/最新版本/
        实时数据/你不确定的客观事实）时调用。query 用搜索关键词。
        """
        if not query or not query.strip():
            return "请提供要搜索的关键词"
        q = query.strip()
        try:
            if provider == "tavily":
                items = await _tavily_search(q, max_results, api_key)
            else:
                items = await _ddg_search(q, max_results)
        except RuntimeError as e:
            log.warning("web_search 执行失败: %s", e)
            return f"搜索失败: {e}"
        except Exception as e:  # noqa: BLE001
            log.warning("web_search 执行失败: %s", e)
            return f"搜索失败: {type(e).__name__}"
        return _format_results(q, items)

    register(_SCHEMA, _web_search)
    log.info("已注册 web_search 工具（provider=%s, max=%d）", provider, max_results)
