"""主动心跳 (架构文档 4.3 / 12 阶段2)。

定时往事件队列投递 HEARTBEAT 事件，驱动主动思考。
间隔在配置区间内随机，避免机械感。
"""
from __future__ import annotations

import asyncio
import random

from .config import Config
from .events import Event, EventType, EventPriority


class Heartbeat:
    def __init__(self, config: Config, enqueue):
        self._cfg = config
        self._enqueue = enqueue
        self._running = False

    async def start(self) -> None:
        """持久化循环：每轮现读 config，主动开关 / 间隔区间改动即时生效，无需重启。

        proactive.enabled 关闭时轻量轮询等待，开启时按区间随机间隔投递心跳事件。
        """
        self._running = True
        while self._running:
            if not self._cfg.get("proactive.enabled", False):
                await asyncio.sleep(2)  # 关闭状态：轮询等待开启
                continue
            rng = self._cfg.get("proactive.heartbeat_interval_range", [60, 180])
            lo, hi = rng[0], rng[1]
            # 注意: 用 random 模块（非 Date/Math.random），这是普通 Python 进程
            delay = random.uniform(lo, hi)
            await asyncio.sleep(delay)
            if not self._running or not self._cfg.get("proactive.enabled", False):
                continue
            await self._enqueue(Event(
                priority=EventPriority.LOW,
                type=EventType.HEARTBEAT,
                payload={"reason": "心跳"},
            ))

    def stop(self) -> None:
        self._running = False
