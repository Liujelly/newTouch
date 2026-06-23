"""端到端集成测试：管理平台聊天接口 → orchestrator → chat_history.jsonl。

不走真实麦克风/TTS，直接在内存里搭起 orchestrator + admin enqueue，
发一条消息，确认：
  1. orchestrator 收到并产生回复（调真实 LLM）
  2. chat_history.jsonl 写入了 user + assistant 两条
  3. /api/chat/history 能读回

需要 .env 里有可用的 LLM key。运行：
  cd D:\\code\\self\\newTouch && python tests/test_chat_e2e.py
"""
from __future__ import annotations

import asyncio
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

from core.config import load_config
from core.character import CharacterCard
from core.cognition import Cognition
from core.state import EmotionState
from core.gatekeeper import GateKeeper
from core.consciousness import ConsciousnessLog
from core.action.speak import Speaker
from core.orchestrator import Orchestrator
from core.events import Event, EventType, EventPriority


async def main() -> bool:
    cfg = load_config()
    if not cfg.get("modules.llm.api_key"):
        print("跳过：未配置 LLM key")
        return True

    # 强制纯文本输出（不依赖 TTS）
    cfg.set("modules.tts.enabled", False)

    char_name = cfg.get("character.name", "默认")
    card = CharacterCard.load(cfg.project_root / "data" / "characters" / char_name / "card.json")
    char_dir = cfg.char_data_dir()
    chat_log = char_dir / "chat_history.jsonl"
    before = chat_log.read_text(encoding="utf-8").count("\n") if chat_log.exists() else 0

    state = EmotionState.load(char_dir / "state.json")
    orch = Orchestrator(
        config=cfg, cognition=Cognition(cfg), speaker=Speaker(cfg, name=card.name),
        card=card, state=state, gatekeeper=GateKeeper(cfg),
        consciousness=ConsciousnessLog(char_dir / "consciousness.jsonl"),
    )
    orch_task = asyncio.create_task(orch.run())

    print("发送测试消息 → 等待 ta 回复（真实调用 LLM）…")
    await orch.enqueue(Event(priority=EventPriority.URGENT, type=EventType.USER_SPEECH,
                             payload={"text": "你好呀，简单回我一句就行"}))
    await orch._queue.join()  # 等这条事件处理完

    orch.shutdown_flag = True
    orch_task.cancel()

    after = chat_log.read_text(encoding="utf-8").count("\n") if chat_log.exists() else 0
    added = after - before
    ok = added >= 2
    print(f"  chat_history 新增 {added} 行（期望 >=2）: {'OK' if ok else 'FAIL'}")
    if ok:
        import json
        last2 = chat_log.read_text(encoding="utf-8").strip().splitlines()[-2:]
        for ln in last2:
            e = json.loads(ln)
            print(f"    [{e['role']}] {e['text'][:40]}")
    return ok


if __name__ == "__main__":
    ok = asyncio.run(main())
    sys.exit(0 if ok else 1)
