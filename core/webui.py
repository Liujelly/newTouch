"""管理平台启动辅助。"""
from __future__ import annotations

import asyncio
import webbrowser
from collections.abc import Callable


def browser_url(host: str, port: int) -> str:
    """把监听地址转换为本机浏览器可访问的管理平台 URL。"""
    browser_host = (host or "127.0.0.1").strip()
    if browser_host in {"0.0.0.0", "::", "[::]"}:
        browser_host = "127.0.0.1"
    elif ":" in browser_host and not browser_host.startswith("["):
        browser_host = f"[{browser_host}]"
    return f"http://{browser_host}:{int(port)}/"


async def open_when_started(
    server,
    url: str,
    opener: Callable[[str], object] = webbrowser.open,
    timeout_seconds: float = 15.0,
) -> bool:
    """等待 uvicorn 真正开始监听后，再交给系统默认浏览器打开。"""
    deadline = asyncio.get_running_loop().time() + timeout_seconds
    while not getattr(server, "started", False):
        if getattr(server, "should_exit", False) or asyncio.get_running_loop().time() >= deadline:
            return False
        await asyncio.sleep(0.1)
    return bool(await asyncio.to_thread(opener, url))
