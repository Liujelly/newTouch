"""日程与待办（架构文档第 14 章）。

核心理念：提醒是一个"念头"，不是一个"播报"。日程到点不直接播固定话，而是往认知
循环投一个 SCHEDULE 事件，走主动路径进内心独白——提醒的措辞/时机/要不要稍后再说，
全由 ta 结合情绪+人设+当前视觉自己决定。

本模块只管数据 + 调度，不管演绎（演绎在 orchestrator 主动路径）：
  - ScheduleItem：日程数据结构
  - ScheduleStore：读写 data/characters/{角色}/schedules.json（按角色隔离），CRUD
  - parse_trigger_at：ISO 主格式 + 宽松解析兜底（"明天10点""晚上7点"等）
  - Scheduler：后台 asyncio 任务，到点 enqueue(SCHEDULE)，勿扰延后，单次标 done/daily 每天

存储：data/characters/{角色}/schedules.json，轻量 JSON 列表。ta 通过白名单工具读写
（需 ai_permissions.allow_manage_schedules），用户在管理平台「日程」Tab 也可管。
"""
from __future__ import annotations

import asyncio
import json
import re
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta, time as dtime
from pathlib import Path
from typing import Any

from ..config import Config
from ..logger import get_logger

log = get_logger("schedule")


@dataclass
class ScheduleItem:
    """单条日程。"""
    id: str
    content: str                 # "提醒我吃药" / "明天10点开会"
    trigger_at: str              # ISO 字符串（解析后的标准格式）
    repeat: str = "none"         # "none"(单次) / "daily"(每天)
    context: str = ""            # 用户当时说的原话（帮 ta 措辞更自然）
    created_at: str = ""
    done: bool = False
    last_triggered: str = ""     # 上次触发的 ISO 时间（防 daily 同一天重复触发）

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ScheduleItem":
        return cls(
            id=d.get("id", str(uuid.uuid4())),
            content=d.get("content", ""),
            trigger_at=d.get("trigger_at", ""),
            repeat=d.get("repeat", "none"),
            context=d.get("context", ""),
            created_at=d.get("created_at", ""),
            done=bool(d.get("done", False)),
            last_triggered=d.get("last_triggered", ""),
        )

    def trigger_dt(self) -> datetime | None:
        try:
            return datetime.fromisoformat(self.trigger_at)
        except (ValueError, TypeError):
            return None


# ── 时间解析 ──────────────────────────────────────────────────
def parse_trigger_at(text: str, now: datetime | None = None) -> str | None:
    """把 LLM/用户传的时间描述解析成 ISO 字符串。

    主格式：ISO（2026-06-29T19:00:00 / 2026-06-29 19:00）
    宽松兜底：自然语言
      - "明天10点" / "明天10:00" / "明天上午10点"
      - "晚上7点" / "今晚8点半" / "下午3点"
      - "19点" / "7点"（默认当天的下一个该时刻）
      - "后天9点"
      - "每天7点"（返回今天下一个7点，repeat 由调用方另设）
    解析不了返回 None（让调用方提示重传）。
    """
    if not text or not str(text).strip():
        return None
    s = str(text).strip()
    now = now or datetime.now()

    # 1. ISO 标准格式
    try:
        dt = datetime.fromisoformat(s.replace("T", " ") if "T" not in s and " " in s and re.match(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}", s) else s)
        return dt.strftime("%Y-%m-%dT%H:%M:%S")
    except ValueError:
        pass

    # 2. 自然语言
    return _parse_natural(s, now)


_TIME_RE = re.compile(r"(\d{1,2})(?::(\d{1,2}))?")  # 7 / 7:30 / 19:00


def _to_24h(hour: int, minute: int, period: str | None) -> tuple[int, int]:
    """12小时制 + 上下午提示 → 24小时。"""
    if period in ("上午", "早上", "早晨", "早", "今早", "明早"):
        if hour == 12:
            hour = 0
    elif period in ("下午", "傍晚", "晚上", "晚", "夜里", "夜晚", "今晚", "明晚"):
        if hour != 12:
            hour += 12
    return hour, minute


def _parse_natural(s: str, now: datetime) -> str | None:
    # 提取日期偏移
    day_offset = 0
    if "后天" in s:
        day_offset = 2
    elif "明天" in s or "明早" in s or "明晚" in s:
        day_offset = 1
    elif "今天" in s or "今晚" in s or "今早" in s:
        day_offset = 0

    # 提取时段（上下午）
    period = None
    for p in ("上午", "下午", "早上", "早晨", "傍晚", "晚上", "夜晚", "夜里",
              "今早", "今晚", "明早", "明晚"):
        if p in s:
            period = p
            break

    # 提取时分
    m = _TIME_RE.search(s)
    if not m:
        return None
    hour = int(m.group(1))
    minute = int(m.group(2)) if m.group(2) else 0
    if "半" in s:
        minute = 30
    hour, minute = _to_24h(hour, minute, period)
    if not (0 <= hour <= 23 and 0 <= minute <= 59):
        return None

    base = now + timedelta(days=day_offset)
    target = base.replace(hour=hour, minute=minute, second=0, microsecond=0)

    # 没指明日期偏移（今天/明天/后天都没出现）→ 取下一个该时刻（已过则明天）
    if day_offset == 0 and not any(k in s for k in ("今天", "今晚", "今早")):
        if target <= now:
            target = target + timedelta(days=1)
    return target.strftime("%Y-%m-%dT%H:%M:%S")


# ── 存储 ──────────────────────────────────────────────────────
class ScheduleStore:
    """读写 schedules.json（按角色隔离）。CRUD。"""

    def __init__(self, char_dir: Path):
        self._path = char_dir / "schedules.json"
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def _load(self) -> list[ScheduleItem]:
        if not self._path.exists():
            return []
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, ValueError):
            return []
        return [ScheduleItem.from_dict(d) for d in (data or []) if isinstance(d, dict)]

    def _save(self, items: list[ScheduleItem]) -> None:
        self._path.write_text(
            json.dumps([i.to_dict() for i in items], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def list_all(self, include_done: bool = True) -> list[ScheduleItem]:
        items = self._load()
        if not include_done:
            items = [i for i in items if not i.done]
        return sorted(items, key=lambda i: i.trigger_at)

    def get(self, item_id: str) -> ScheduleItem | None:
        for i in self._load():
            if i.id == item_id:
                return i
        return None

    def add(self, content: str, trigger_at: str, repeat: str = "none",
            context: str = "") -> ScheduleItem | None:
        """新建日程。trigger_at 已是 ISO 字符串（parse_trigger_at 的产物）。"""
        iso = parse_trigger_at(trigger_at)
        if not iso:
            return None
        repeat = repeat if repeat in ("none", "daily") else "none"
        item = ScheduleItem(
            id=str(uuid.uuid4()),
            content=content,
            trigger_at=iso,
            repeat=repeat,
            context=context,
            created_at=datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        )
        items = self._load()
        items.append(item)
        self._save(items)
        log.info("新增日程: %s @ %s (%s)", content[:30], iso, repeat)
        return item

    def update(self, item_id: str, content: str | None = None,
               trigger_at: str | None = None, repeat: str | None = None) -> bool:
        items = self._load()
        for i in items:
            if i.id == item_id:
                if content is not None:
                    i.content = content
                if trigger_at is not None:
                    iso = parse_trigger_at(trigger_at)
                    if iso:
                        i.trigger_at = iso
                if repeat is not None and repeat in ("none", "daily"):
                    i.repeat = repeat
                self._save(items)
                return True
        return False

    def mark_done(self, item_id: str) -> bool:
        items = self._load()
        for i in items:
            if i.id == item_id:
                i.done = True
                self._save(items)
                return True
        return False

    def delete(self, item_id: str) -> bool:
        items = self._load()
        new = [i for i in items if i.id != item_id]
        if len(new) == len(items):
            return False
        self._save(new)
        return True

    def mark_triggered(self, item_id: str) -> None:
        """触发后：单次标 done；daily 只更新 last_triggered（防当天重复触发）。"""
        items = self._load()
        for i in items:
            if i.id == item_id:
                i.last_triggered = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
                if i.repeat == "none":
                    i.done = True
                self._save(items)
                return


# ── 调度器 ────────────────────────────────────────────────────
class Scheduler:
    """后台任务：周期扫描到期日程，到点投 SCHEDULE 事件。

    单次日程触发后自动标 done；daily 每天到点触发一次（用 last_triggered 防当天重复）。
    勿扰时段的日程延到 quiet_end 后再触发（架构 14.3：深夜提醒自动延后）。
    """

    def __init__(self, store: ScheduleStore, enqueue, config: Config):
        self._store = store
        self._enqueue = enqueue
        self._cfg = config
        self._task: asyncio.Task | None = None

    def start(self) -> None:
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    def _in_quiet_hours(self, now: datetime) -> tuple[bool, datetime | None]:
        """返回 (是否勿扰, 勿扰结束时刻)。跨午夜勿扰段（如23:00-07:00）。"""
        qs = self._cfg.get("proactive.quiet_start", "23:00")
        qe = self._cfg.get("proactive.quiet_end", "07:00")
        try:
            sh, sm = map(int, qs.split(":"))
            eh, em = map(int, qe.split(":"))
        except (ValueError, AttributeError):
            return False, None
        cur = now.time()
        start, end = dtime(sh, sm), dtime(eh, em)
        in_quiet = cur >= start or cur < end if start > end else start <= cur < end
        if not in_quiet:
            return False, None
        # 算勿扰结束时刻（今天或明天）
        end_dt = now.replace(hour=eh, minute=em, second=0, microsecond=0)
        if end_dt <= now:
            end_dt = end_dt + timedelta(days=1)
        return True, end_dt

    async def _run(self) -> None:
        from ..events import Event, EventType, EventPriority
        interval = int(self._cfg.get("schedule.check_interval_seconds", 30))
        interval = max(5, interval)
        log.info("日程调度器已启动（扫描间隔 %ss）", interval)
        while True:
            try:
                now = datetime.now()
                in_quiet, quiet_end = self._in_quiet_hours(now)
                for item in self._store.list_all(include_done=False):
                    self._maybe_trigger(item, now, in_quiet, quiet_end, EventType, Event, EventPriority)
            except Exception as e:  # noqa: BLE001
                log.warning("日程扫描异常: %s", e)
            await asyncio.sleep(interval)

    def _maybe_trigger(self, item: ScheduleItem, now: datetime,
                        in_quiet: bool, quiet_end: datetime | None,
                        EventType, Event, EventPriority) -> None:
        trig = item.trigger_dt()
        if not trig:
            return
        # daily：今天已触发过则跳过
        if item.repeat == "daily" and item.last_triggered:
            try:
                lt = datetime.fromisoformat(item.last_triggered)
                if lt.date() == now.date():
                    return
            except ValueError:
                pass

        if trig > now:
            return  # 还没到点

        # 到点了。勿扰时段 → 延到 quiet_end 后
        if in_quiet and quiet_end:
            log.info("日程「%s」到期但处勿扰时段，延到 %s", item.content[:20], quiet_end)
            # 不触发，等勿扰结束后自然到点（trig 已 < now，下轮 quiet 结束就会触发）
            return

        # 触发
        log.info("日程到期触发: %s", item.content[:30])
        self._store.mark_triggered(item.id)
        try:
            self._enqueue(Event(
                priority=EventPriority.NORMAL,
                type=EventType.SCHEDULE,
                payload={"item": item.to_dict()},
            ))
        except Exception as e:  # noqa: BLE001
            log.warning("日程事件入队失败: %s", e)
