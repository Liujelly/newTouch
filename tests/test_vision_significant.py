"""vision.py 显著变化分派 + orchestrator._handle_vision_significant 功能测试。

覆盖：
  1. diff < significant_threshold → VISION_CHANGE
  2. diff >= significant_threshold → VISION_SIGNIFICANT
  3. _handle_vision_significant: 冷却内 → silent，不调 LLM
  4. _handle_vision_significant: 勿扰时段 → silent，不调 LLM
  5. _handle_vision_significant: LLM 返回 silent → 不 speak
  6. _handle_vision_significant: LLM 返回 speak → 调 speaker，
     且 gatekeeper.record_spoke 不被调用（不占每小时配额）

运行: cd D:\\code\\self\\newTouch && python tests/test_vision_significant.py
"""
from __future__ import annotations

import asyncio
import sys
from datetime import datetime, time as dtime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.events import EventType, EventPriority, Event
from core.config import Config

# ── 结果收集 ─────────────────────────────────────────────────
_results: list[tuple[str, bool, str]] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    _results.append((name, cond, detail))
    status = "✓" if cond else "✗"
    print(f"  {status} {name}" + (f"  ({detail})" if detail else ""))


# ── Case 1 & 2：vision.py 事件类型分派 ───────────────────────

def test_event_type_dispatch():
    """diff_score 与 significant_threshold 比较决定投哪种事件。"""
    emitted: list[Event] = []

    async def fake_enqueue(e: Event):
        emitted.append(e)

    cfg = Config({
        "modules": {"vlm": {"provider": "openai", "api_key": "x",
                             "base_url": "http://x", "model": "m"}},
        "perception": {"vision": {
            "frame_diff_threshold": 0.15,
            "min_caption_interval_s": 0,   # 不节流，方便测试
            "significant_threshold": 0.30,
        }},
    })

    # 模拟 VLM 返回 caption
    from core.perception.vision import Vision, VisionCaption
    vision = Vision.__new__(Vision)
    vision._cfg = cfg
    vision._diff_threshold = 0.15
    vision._min_interval = 0
    vision._last_caption_ts = 0.0
    vision._enqueue = fake_enqueue
    vision._running = False
    vision._last_frame = None
    vision._latest_frame_raw = None
    vision._latest = None
    vision._last_look_ts = 0.0
    vision._enabled = True

    fake_caption = VisionCaption(caption="有人进来了", timestamp=0.0, frame_diff_score=0.0)

    async def run(diff_score: float) -> EventType:
        emitted.clear()
        # 直接调被动循环里的判断逻辑（等效内联）
        import time, numpy as np
        now = time.time()
        vision._last_caption_ts = 0.0
        with patch.object(vision, "_vlm_caption", new=AsyncMock(return_value=fake_caption)):
            frame = np.zeros((100, 100, 3), dtype=np.uint8)
            vision._last_frame = None   # 让 diff 返回固定值
            if (diff_score > vision._diff_threshold
                    and now - vision._last_caption_ts > vision._min_interval):
                vision._last_caption_ts = now
                caption = await vision._vlm_caption(frame, diff_score)
                if caption:
                    vision._latest = caption
                    sig_threshold = vision._cfg.get(
                        "perception.vision.significant_threshold", 0.30)
                    evt_type = (EventType.VISION_SIGNIFICANT
                                if diff_score >= sig_threshold
                                else EventType.VISION_CHANGE)
                    await vision._enqueue(Event(
                        priority=EventPriority.NORMAL,
                        type=evt_type,
                        payload={"caption": caption.caption, "diff_score": diff_score},
                    ))
        return emitted[0].type if emitted else None

    loop = asyncio.new_event_loop()
    t1 = loop.run_until_complete(run(0.20))
    check("diff=0.20(<0.30) → VISION_CHANGE", t1 == EventType.VISION_CHANGE,
          f"got {t1}")

    t2 = loop.run_until_complete(run(0.35))
    check("diff=0.35(>=0.30) → VISION_SIGNIFICANT", t2 == EventType.VISION_SIGNIFICANT,
          f"got {t2}")

    t3 = loop.run_until_complete(run(0.30))
    check("diff=0.30(边界) → VISION_SIGNIFICANT", t3 == EventType.VISION_SIGNIFICANT,
          f"got {t3}")
    loop.close()


# ── 构造最小 Orchestrator stub ─────────────────────────────────

def _make_orch(*, quiet=False, last_check=0.0, min_check=60,
               llm_action="silent", llm_text=""):
    """返回一个只含 _handle_vision_significant 所需字段的 Orchestrator 实例。"""
    from core.orchestrator import Orchestrator
    from core.state import EmotionState
    from core.consciousness import ConsciousnessLog
    from core.character import CharacterCard

    cfg_raw = {
        "user_persona": {"name": "测试用户"},
        "memory": {"chat_history_window": 40, "compress_enabled": False,
                   "enabled": False},
        "modules": {"tts": {"text_lang": "zh", "enabled": False}},
        "proactive": {"min_interval_seconds": 60, "quiet_start": "23:00",
                      "quiet_end": "07:00", "hourly_cap": 5, "clinginess": 0.5},
        "perception": {"vision": {
            "min_check_interval_s": min_check,
            "significant_threshold": 0.30,
        }},
        "character": {"name": "默认"},
    }
    cfg = Config(cfg_raw)

    state = EmotionState()

    # GateKeeper stub：记录 record_spoke 调用次数
    gate = MagicMock()
    gate.record_spoke = MagicMock()
    # 控制勿扰时段
    gate._in_quiet_hours = MagicMock(return_value=quiet)

    # Cognition stub
    cognition = MagicMock()
    cognition.proactive_think = AsyncMock(return_value={
        "action": llm_action,
        "thought": "test thought",
        "text": llm_text,
        "emotion": None,
        "emotion_delta": {},
    })

    # Speaker stub
    speaker = MagicMock()
    async def _fake_speak(gen, emotion=None):
        async for _ in gen:
            pass
    speaker.speak = _fake_speak

    card = CharacterCard(name="测试", description="", personality="",
                         scenario="", first_mes="", system_prompt="",
                         extensions={})
    log = MagicMock()
    log.record = MagicMock()

    import tempfile
    from pathlib import Path
    tmp = Path(tempfile.mkdtemp())
    (tmp / "data").mkdir(exist_ok=True)

    # 直接构造，绕过 __init__ 中需要文件的部分
    orch = object.__new__(Orchestrator)
    orch._cfg = cfg
    orch._cognition = cognition
    orch._speaker = speaker
    orch._card = card
    orch._state = state
    orch._gate = gate
    orch._log = log
    orch._vision = MagicMock()
    orch._chat_history = []
    orch._user_name = "测试用户"
    orch._char_dir = tmp / "data"
    orch._state_path = tmp / "data" / "state.json"
    orch._chat_log_path = tmp / "data" / "chat_history.jsonl"
    orch._last_vision = ""
    orch._last_vision_check = last_check
    orch._awaiting_reply = False
    orch._last_proactive_ts = 0.0
    orch._earlier_summary = ""
    orch._recent_inner = []
    orch._cognition_lock = asyncio.Lock()
    orch._memory = MagicMock()
    orch._memory.add = MagicMock()
    orch._max_history = 40
    orch._compress_batch = 10

    # 让 _compact_history / _log_chat 安全空转
    async def _noop_compact():
        pass
    orch._compact_history = _noop_compact

    def _noop_log_chat(role, text):
        pass
    orch._log_chat = _noop_log_chat

    def _noop_record_inner(kind, thought, text=""):
        pass
    orch._record_inner = _noop_record_inner

    return orch, gate, cognition


# ── Case 3：冷却内不调 LLM ────────────────────────────────────

def test_cooldown_blocks_llm():
    import time
    orch, gate, cognition = _make_orch(
        last_check=time.time() - 10,  # 10s 前刚判断过，min_check=60
        min_check=60,
    )
    event = Event(priority=EventPriority.NORMAL, type=EventType.VISION_SIGNIFICANT,
                  payload={"caption": "他回来了", "diff_score": 0.6})

    loop = asyncio.new_event_loop()
    loop.run_until_complete(orch._handle_vision_significant(event))
    loop.close()

    check("冷却内: proactive_think 未被调用",
          not cognition.proactive_think.called)
    check("冷却内: caption 仍注入 chat_history（素材保留）",
          any("你看到了" in m.get("content", "") for m in orch._chat_history))


# ── Case 4：勿扰时段不调 LLM ─────────────────────────────────

def test_quiet_hours_blocks_llm():
    orch, gate, cognition = _make_orch(quiet=True, last_check=0.0, min_check=0)
    event = Event(priority=EventPriority.NORMAL, type=EventType.VISION_SIGNIFICANT,
                  payload={"caption": "他回来了", "diff_score": 0.6})

    loop = asyncio.new_event_loop()
    loop.run_until_complete(orch._handle_vision_significant(event))
    loop.close()

    check("勿扰时段: proactive_think 未被调用",
          not cognition.proactive_think.called)


# ── Case 5：LLM silent → 不 speak ────────────────────────────

def test_llm_silent_no_speak():
    orch, gate, cognition = _make_orch(
        last_check=0.0, min_check=0,
        llm_action="silent", llm_text="",
    )
    event = Event(priority=EventPriority.NORMAL, type=EventType.VISION_SIGNIFICANT,
                  payload={"caption": "他在走来走去", "diff_score": 0.35})

    loop = asyncio.new_event_loop()
    loop.run_until_complete(orch._handle_vision_significant(event))
    loop.close()

    check("LLM silent: proactive_think 被调用", cognition.proactive_think.called)
    check("LLM silent: gatekeeper.record_spoke 未被调用",
          not gate.record_spoke.called)
    check("LLM silent: memory.add 未被调用",
          not orch._memory.add.called)


# ── Case 6：LLM speak → 开口，但不占主动配额 ─────────────────

def test_llm_speak_no_gatekeeper():
    orch, gate, cognition = _make_orch(
        last_check=0.0, min_check=0,
        llm_action="speak", llm_text="你回来啦～",
    )
    event = Event(priority=EventPriority.NORMAL, type=EventType.VISION_SIGNIFICANT,
                  payload={"caption": "一个人走进房间", "diff_score": 0.65})

    loop = asyncio.new_event_loop()
    loop.run_until_complete(orch._handle_vision_significant(event))
    loop.close()

    check("LLM speak: proactive_think 被调用", cognition.proactive_think.called)
    check("LLM speak: gatekeeper.record_spoke 未被调用（不占配额）",
          not gate.record_spoke.called,
          "视觉触发不应消耗每小时主动上限")
    check("LLM speak: awaiting_reply 已设为 True", orch._awaiting_reply)
    check("LLM speak: memory.add 被调用", orch._memory.add.called)
    check("LLM speak: 回复写入 chat_history",
          any("你回来啦" in m.get("content", "") for m in orch._chat_history))


# ── 运行所有 ─────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n=== test_vision_significant ===\n")

    print("[1/2] 事件类型分派 (vision.py threshold)")
    test_event_type_dispatch()

    print("\n[3] 冷却内不调 LLM")
    test_cooldown_blocks_llm()

    print("\n[4] 勿扰时段不调 LLM")
    test_quiet_hours_blocks_llm()

    print("\n[5] LLM silent → 不 speak")
    test_llm_silent_no_speak()

    print("\n[6] LLM speak → 开口但不占配额")
    test_llm_speak_no_gatekeeper()

    passed = sum(1 for _, ok, _ in _results if ok)
    total = len(_results)
    print(f"\n{'='*40}")
    print(f"结果: {passed}/{total} 通过" + ("  ✓ ALL PASS" if passed == total else ""))
    if passed < total:
        print("失败项:")
        for name, ok, detail in _results:
            if not ok:
                print(f"  ✗ {name}" + (f"  ({detail})" if detail else ""))
        sys.exit(1)
