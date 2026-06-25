"""face 广播器：主程序 TCP server，把当前立绘表情推给立绘浮窗进程。

浮窗是独立进程（PyQt6），通过本地 TCP socket 订阅 face 变化。主程序是 server，
浮窗是 client。浮窗后启动/断线重连都能接上。

设计要点：
- 可选功能：start() 失败（端口被占等）静默降级，不阻塞主程序。
- push() 无 client 时直接丢弃（不积压，浮窗没连就不显示，符合"可选"定位）。
- 只推 face（立绘用）；语音情绪 _cur_emotion 留在主程序内驱动 TTS，不跨进程。
"""
from __future__ import annotations

import asyncio
import json
from typing import Iterable

from ..logger import get_logger

log = get_logger("sprite")


class FaceBroadcaster:
    """TCP server，向所有连上的浮窗 client 推送 face 变化。"""

    def __init__(self, host: str = "127.0.0.1", port: int = 17621) -> None:
        self._host = host
        self._port = port
        self._clients: set[asyncio.StreamWriter] = set()
        self._server: asyncio.AbstractServer | None = None
        self._started = False

    async def start(self) -> None:
        """启动 TCP server。失败静默降级（浮窗功能可选）。"""
        if self._started:
            return
        try:
            self._server = await asyncio.start_server(
                self._handle_client, self._host, self._port
            )
            self._started = True
            log.info("立绘广播器就绪 %s:%s", self._host, self._port)
        except OSError as e:
            # 端口被占/不可用：浮窗功能降级，主程序照常跑
            log.warning("立绘广播器启动失败（浮窗功能禁用）: %s", e)
            self._started = False

    async def _handle_client(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        """每个浮窗 client 连接：加入列表，保持连接直到断开。"""
        self._clients.add(writer)
        peer = writer.get_extra_info("peername")
        log.info("浮窗已连接: %s", peer)
        try:
            # 不读 client 数据（单向推送），但保持连接存活：读到 EOF 说明 client 断开
            while not reader.at_eof():
                try:
                    await reader.read(1024)
                except (ConnectionError, OSError):
                    break
        except (ConnectionError, OSError):
            pass
        finally:
            self._clients.discard(writer)
            try:
                writer.close()
                await writer.wait_closed()
            except (ConnectionError, OSError, OSError):
                pass
            log.info("浮窗已断开: %s", peer)

    async def _broadcast(self, obj: dict) -> None:
        """向所有连上的浮窗推送一行 JSON。无 client 时丢弃。"""
        if not self._started or not self._clients:
            return
        line = json.dumps(obj, ensure_ascii=False) + "\n"
        data = line.encode("utf-8")
        dead: list[asyncio.StreamWriter] = []
        for writer in self._clients:
            try:
                writer.write(data)
            except (ConnectionError, RuntimeError):
                dead.append(writer)
        for w in dead:
            self._clients.discard(w)
        # 尽力 drain（不阻塞太久，单个失败不影响其他）
        for writer in list(self._clients):
            try:
                await writer.drain()
            except (ConnectionError, OSError):
                self._clients.discard(writer)

    async def push(self, face: str, character: str) -> None:
        """推送立绘表情变化。"""
        await self._broadcast({"face": face, "character": character})

    async def push_text(self, text: str, character: str) -> None:
        """推送一段增量台词（浮窗气泡逐 chunk 流式追加）。"""
        if not text:
            return
        await self._broadcast({"text": text, "character": character})

    async def push_text_end(self, character: str) -> None:
        """标记一段回复结束（浮窗据此启动气泡淡出）。"""
        await self._broadcast({"text_end": True, "character": character})


    def has_clients(self) -> bool:
        return bool(self._clients)

    async def stop(self) -> None:
        if self._server is not None:
            self._server.close()
            try:
                await self._server.wait_closed()
            except (OSError, RuntimeError):
                pass
        for w in list(self._clients):
            try:
                w.close()
            except (ConnectionError, RuntimeError):
                pass
        self._clients.clear()
        self._started = False
