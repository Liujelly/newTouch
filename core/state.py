"""情绪状态 (架构文档 5.3 / 7.2)。

五维情绪向量，随时间衰减，支持 LLM 增量更新和持久化。
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class EmotionState:
    valence: float = 0.0     # [-1, 1] 愉悦度, baseline 0.0
    arousal: float = 0.2     # [0, 1]  唤醒度, baseline 0.2
    attachment: float = 0.0  # [0, 1]  依恋度, 慢变量 (tau≈7天)
    worry: float = 0.0       # [0, 1]  担心,   快变量
    loneliness: float = 0.0  # [0, 1]  孤独感, 随沉默线性增长
    consecutive_ignored: int = 0  # 连续主动发言未被回应次数（驱动"越叫越担心/失落"）

    last_interaction: float = field(default_factory=time.time)
    last_tick: float = field(default_factory=time.time)

    # baseline 用于衰减目标
    _BASELINE = {"valence": 0.0, "arousal": 0.2, "worry": 0.0}
    _TAU_FAST = 30 * 60.0   # 30 分钟半衰期（快变量）
    _TAU_SLOW = 7 * 86400.0 # 7 天半衰期（attachment）

    # ---------- 对外 API ----------

    def apply_delta(self, delta: dict[str, float]) -> None:
        """应用 LLM 输出的情绪增量，范围裁剪。"""
        _RANGES = {"valence": (-1, 1), "arousal": (0, 1),
                   "attachment": (0, 1), "worry": (0, 1), "loneliness": (0, 1)}
        for k, v in delta.items():
            if k in _RANGES:
                lo, hi = _RANGES[k]
                setattr(self, k, max(lo, min(hi, getattr(self, k) + v)))

    def tick(self) -> None:
        """时间驱动的衰减，每次认知循环前调用。"""
        now = time.time()
        dt = now - self.last_tick
        self.last_tick = now
        if dt <= 0:
            return

        # 快变量: 指数回归到 baseline
        for key, base in self._BASELINE.items():
            cur = getattr(self, key)
            k = math.exp(-dt / self._TAU_FAST)
            setattr(self, key, base + (cur - base) * k)

        # 慢变量 attachment: 极慢衰减
        self.attachment *= math.exp(-dt / self._TAU_SLOW)

        # loneliness: 随沉默线性增长 (2h 到 1.0), 上限 1.0
        silent = now - self.last_interaction
        self.loneliness = min(1.0, silent / 7200.0)

    def on_interaction(self, positive: bool = True) -> None:
        """用户互动时调用：重置孤独感，累积依恋度，清零未回应计数。"""
        self.last_interaction = time.time()
        self.loneliness = 0.0
        self.consecutive_ignored = 0  # 用户回应了，重置"被忽略"递进
        delta = 0.02 if positive else -0.01
        self.attachment = max(0.0, min(1.0, self.attachment + delta))

    def on_proactive_spoke(self) -> None:
        """ta 主动开口后调用（区别于 on_interaction）：只小幅累积依恋，
        **不**清零孤独感、**不**清零未回应计数。

        理由：对着没人回应的对话框说话不等于得到了陪伴，孤独感不该因为"自己说了话"
        而消失，否则会出现"对空气说完就不孤独了"的反直觉行为，也使"越叫越担心"无从递进。
        是否真的被回应，由下一次 on_interaction（用户来聊）或 on_ignored（仍无回应）裁决。
        """
        self.attachment = max(0.0, min(1.0, self.attachment + 0.01))

    def on_ignored(self) -> None:
        """主动发言后一直没等到用户回应时调用：推高担心、压低愉悦、累加未回应次数。

        次数越多，担心增量越大（前几次轻、之后变重），模拟"叫了好几次都不理→越来越不安"。
        worry 是快变量（30min 半衰期），若用户长时间不回会被多次 on_ignored 持续顶高；
        一旦用户回应，on_interaction 清零计数、worry 也随时间衰减回落。
        """
        self.consecutive_ignored += 1
        n = self.consecutive_ignored
        worry_inc = min(0.5, 0.12 * n)        # 第1次+0.12，封顶0.5
        self.worry = max(0.0, min(1.0, self.worry + worry_inc))
        self.valence = max(-1.0, min(1.0, self.valence - 0.08))

    def snapshot(self) -> dict:
        """对话结束时冻结快照，写入记忆标签。"""
        return {k: round(getattr(self, k), 3)
                for k in ("valence", "arousal", "attachment", "worry", "loneliness")}

    def summary(self) -> str:
        """给 prompt 用的简短文字描述。consecutive_ignored>0 时附加未回应次数。"""
        s = (f"valence={self.valence:.2f} arousal={self.arousal:.2f} "
             f"attachment={self.attachment:.2f} worry={self.worry:.2f} "
             f"loneliness={self.loneliness:.2f}")
        if self.consecutive_ignored > 0:
            s += f" (已主动找过{self.consecutive_ignored}次但对方暂无回应)"
        return s

    # ---------- 持久化 ----------

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        data = asdict(self)
        path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "EmotionState":
        if not path.exists():
            return cls()
        data = json.loads(path.read_text(encoding="utf-8"))
        fields = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in fields})
