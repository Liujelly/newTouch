"""测试管理平台启动后自动打开浏览器。"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.webui import browser_url, open_when_started


def test_browser_url():
    assert browser_url("127.0.0.1", 8080) == "http://127.0.0.1:8080/"
    assert browser_url("0.0.0.0", 9000) == "http://127.0.0.1:9000/"
    assert browser_url("::", 8080) == "http://127.0.0.1:8080/"
    assert browser_url("::1", 8080) == "http://[::1]:8080/"


def test_open_when_started():
    class Server:
        started = False
        should_exit = False

    server = Server()
    opened = []

    def opener(url):
        opened.append(url)
        return True

    async def run():
        task = asyncio.create_task(
            open_when_started(server, "http://127.0.0.1:8080/", opener, 1)
        )
        await asyncio.sleep(0.05)
        assert opened == []
        server.started = True
        assert await task is True

    asyncio.run(run())
    assert opened == ["http://127.0.0.1:8080/"]


if __name__ == "__main__":
    test_browser_url()
    test_open_when_started()
    print("管理平台自动打开测试通过")
