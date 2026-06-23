"""意识流日志 (架构文档 8.2 观测面板)。

把每次内心独白 (thought/action/text) append 到 data/consciousness.jsonl，
供管理平台"意识流时间线"展示，也是调沉默闸门参数的核心反馈。
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path


class ConsciousnessLog:
    def __init__(self, path: Path):
        self._path = path
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def record(
        self,
        trigger: str,
        action: str,         # speak | silent | look | tool
        thought: str = "",   # 内心独白：ta 此刻在想什么
        text: str = "",
        emotion: dict | None = None,
        gate: str = "",      # 闸门判定原因
    ) -> None:
        entry = {
            "ts": datetime.now().isoformat(timespec="seconds"),
            "trigger": trigger,
            "action": action,
            "thought": thought,
            "text": text,
            "emotion": emotion or {},
            "gate": gate,
        }
        with self._path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        # 终端回显，方便实时观察
        if action == "silent":
            if thought:
                print(f"\n  〔{trigger}·想：{thought} → 沉默〕")
            else:
                print(f"\n  〔{trigger} → 沉默：{gate or 'ta 选择不说'}〕")
