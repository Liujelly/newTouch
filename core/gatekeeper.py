"""硬闸门 GateKeeper (架构文档 5.4 / 4.3 第1层)。

基于规则的主动发言控制器：勿扰时段、最小间隔、每小时上限、backoff。
这是 LLM 之外的硬性兜底——情绪可以调低阈值，但勿扰/上限不可被 LLM 绕过。
"""
from __future__ import annotations

import time
from collections import deque
from datetime import datetime, time as dtime

from .config import Config
from .state import EmotionState


def _parse_hhmm(s: str, default: dtime) -> dtime:
    try:
        h, m = s.split(":")
        return dtime(int(h), int(m))
    except Exception:  # noqa: BLE001
        return default


class GateKeeper:
    def __init__(self, config: Config):
        p = "proactive."
        self._quiet_start = _parse_hhmm(config.get(p + "quiet_start", "23:00"), dtime(23, 0))
        self._quiet_end = _parse_hhmm(config.get(p + "quiet_end", "07:00"), dtime(7, 0))
        self._min_interval = config.get(p + "min_interval_seconds", 900)
        self._hourly_cap = config.get(p + "hourly_cap", 5)
        self._max_backoff = config.get(p + "max_backoff_minutes", 240) * 60
        # 黏人度 clinginess ∈ [0,1]，决定"被冷落时该催还是该退"的性格取向。
        # 0.5=中性(等于历史行为)；高→worry 更催、backoff 退避更弱；低→反之。
        # 全局默认在此读，角色卡 extensions.clinginess 由 orchestrator 切角色时覆盖。
        self._clinginess = self._clamp01(config.get(p + "clinginess", 0.5))

        self._last_spoke: float = 0.0
        self._backoff: float = 0.0           # 被忽略时指数增长的额外冷却
        self._spoke_times: deque[float] = deque()  # 近一小时发言时间戳

    # ---------- 闸门判定 ----------

    def check(self, state: EmotionState) -> tuple[bool, str]:
        """返回 (是否允许主动发言, 原因)。"""
        now = time.time()

        # ① 勿扰时段（硬规则，情绪不可绕过）
        if self._in_quiet_hours():
            return False, "勿扰时段"

        # ② 最小间隔 + backoff，受情绪 + 性格(clinginess)双重调节
        #    attachment 高 → 间隔缩短(越亲近越想说)
        #    worry 高 → 间隔缩短，缩短力度由 clinginess 决定(黏人的角色越担心越催)
        #    backoff(被冷落退避) → 退避力度由 clinginess 反向决定(黏人的角色不太退、独立的角色识趣退)
        #    clinginess=0.5 时 worry 系数=0.3、backoff 缩放=1.0，严格等于历史行为。
        c = self._clinginess
        worry_w = 0.3 * (2 * c)          # c=0→0, c=0.5→0.3, c=1→0.6
        backoff_w = 2.0 * (1 - c)        # c=0→2.0(很退), c=0.5→1.0, c=1→0(几乎不退)
        factor = 1.0 - 0.5 * state.attachment - worry_w * state.worry
        factor = max(0.2, factor)  # 最多缩到 20%
        effective_interval = self._min_interval * factor + self._backoff * backoff_w
        since = now - self._last_spoke
        if since < effective_interval:
            return False, f"冷却中({int(since)}/{int(effective_interval)}s)"

        # ③ 每小时上限
        self._prune(now)
        if len(self._spoke_times) >= self._hourly_cap:
            return False, f"已达每小时上限({self._hourly_cap})"

        return True, "通过"

    # ---------- 状态更新 ----------

    def record_spoke(self) -> None:
        """主动发言成功：进每小时上限统计 + 刷新间隔。
        注意：不重置 backoff——开口不等于被回应，backoff 只在用户真正回应时（record_interaction）清零。
        只应由主动路径(心跳)调用。"""
        now = time.time()
        self._last_spoke = now
        self._spoke_times.append(now)

    def record_interaction(self) -> None:
        """用户主动找 ta 对话(反应路径)：刷新间隔 + 重置 backoff，但**不**计入每小时上限。
        理由：每小时上限限制的是 ta 主动开口打扰用户，用户自己来聊不该消耗该额度；
        但刚聊完不该马上又主动开口，所以仍更新 _last_spoke 让主动间隔从最后互动算起。"""
        self._last_spoke = time.time()
        self._backoff = 0.0

    def apply_frequency(self, min_interval: int, hourly_cap: int) -> None:
        """self_config 频率档位变更时同步到运行中的实例。"""
        self._min_interval = min_interval
        self._hourly_cap = hourly_cap

    @staticmethod
    def _clamp01(v) -> float:
        try:
            return max(0.0, min(1.0, float(v)))
        except (TypeError, ValueError):
            return 0.5

    def set_clinginess(self, value) -> None:
        """切角色时由 orchestrator 用角色卡 extensions.clinginess 覆盖性格取向。
        None/非法值回退 0.5(中性)。"""
        self._clinginess = self._clamp01(value if value is not None else 0.5)

    def refresh(self, config: Config) -> None:
        """管理平台改配置热重载后，重新读取全部闸门参数（勿扰时段/间隔/上限/backoff）。

        只更新规则参数，不动运行时计数状态（_last_spoke/_spoke_times/_backoff）。
        """
        p = "proactive."
        self._quiet_start = _parse_hhmm(config.get(p + "quiet_start", "23:00"), dtime(23, 0))
        self._quiet_end = _parse_hhmm(config.get(p + "quiet_end", "07:00"), dtime(7, 0))
        self._min_interval = config.get(p + "min_interval_seconds", 900)
        self._hourly_cap = config.get(p + "hourly_cap", 5)
        self._max_backoff = config.get(p + "max_backoff_minutes", 240) * 60
        # clinginess 全局默认重读；角色卡 extensions.clinginess 由 orchestrator
        # 在切角色 / 启动时通过 set_clinginess 覆盖（优先级更高），故此处仅刷新兜底值。
        self._clinginess = self._clamp01(config.get(p + "clinginess", 0.5))

    def record_ignored(self) -> None:
        """主动说了但没得到回应：指数增长 backoff，避免连续打扰。"""
        self._backoff = min(self._max_backoff, max(60.0, self._backoff * 1.5 or 60.0))

    def set_clinginess(self, value) -> None:
        """设置黏人度（角色卡 extensions.clinginess 优先于全局默认）。

        由 orchestrator 在启动 / 切角色时调用：角色卡有该字段就用卡里的，
        没有则回退全局 proactive.clinginess。非法值钳到 [0,1]，None 时不改（保留现值）。
        """
        if value is None:
            return
        self._clinginess = self._clamp01(value)

    # ---------- 内部 ----------

    @staticmethod
    def _clamp01(value) -> float:
        """把任意输入钳到 [0,1]，非数字回退 0.5（中性）。"""
        try:
            return max(0.0, min(1.0, float(value)))
        except (TypeError, ValueError):
            return 0.5

    def _in_quiet_hours(self) -> bool:
        now = datetime.now().time()
        s, e = self._quiet_start, self._quiet_end
        if s <= e:
            return s <= now < e
        return now >= s or now < e  # 跨午夜

    def _prune(self, now: float) -> None:
        while self._spoke_times and now - self._spoke_times[0] > 3600:
            self._spoke_times.popleft()
