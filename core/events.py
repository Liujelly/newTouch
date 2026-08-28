"""事件定义: 所有触发源汇入同一队列的消息协议 (架构文档 5.1)。"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, IntEnum
from typing import Any


class EventType(Enum):
    HEARTBEAT = "heartbeat"          # 心跳tick, 驱动主动思考 (阶段2)
    VISION_CHANGE = "vision_change"  # 视觉场景变化 (阶段3)
    VISION_SIGNIFICANT = "vision_significant"  # 显著视觉变化 -> LLM智能判断是否开口
    USER_SPEECH = "user_speech"      # 对ta说的话 -> 反应路径
    EAVESDROP = "eavesdrop"          # 旁听到的话 -> 旁听路径 (阶段3)
    SCHEDULE = "schedule"            # 定时/闹钟触发 (阶段14)
    EXTERNAL = "external"            # 外部告警/webhook


class EventPriority(IntEnum):
    URGENT = 0   # 你说话、打断
    NORMAL = 5   # 视觉变化、旁听
    LOW = 10     # 心跳、定时


# 单调递增的序号, 保证同优先级下 FIFO, 且避免 PriorityQueue 比较 Event 时报错
_seq = itertools.count()


@dataclass(order=True)
class Event:
    # 用于 PriorityQueue 排序的字段必须排在前面; payload/type 不参与比较
    priority: EventPriority
    _seq: int = field(init=False)
    type: EventType = field(compare=False)
    payload: dict[str, Any] = field(default_factory=dict, compare=False)
    timestamp: datetime = field(default_factory=datetime.now, compare=False)

    def __post_init__(self):
        self._seq = next(_seq)


def user_speech(
    text: str,
    speaker_id: str | None = None,
    *,
    source: str = "text",
    uncertain_audio: bool = False,
) -> Event:
    return Event(
        priority=EventPriority.URGENT,
        type=EventType.USER_SPEECH,
        payload={
            "text": text,
            "speaker_id": speaker_id,
            "source": source,
            "uncertain_audio": uncertain_audio,
        },
    )
