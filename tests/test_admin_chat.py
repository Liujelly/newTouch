"""集成测试：admin 聊天接口 + orchestrator enqueue 接合（同进程，不起 HTTP/stdin）。

直接调用 admin 的路由函数，验证 set_enqueue 后 send_chat 能把消息投进队列，
orchestrator 处理后写入 chat_history.jsonl，get_chat_history 能读回。

运行：cd D:\\code\\self\\newTouch && python tests/test_admin_chat.py
"""
from __future__ import annotations

import asyncio
import sys
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
from api import admin


async def main() -> bool:
    cfg = load_config()
    if not cfg.get("modules.llm.api_key"):
        print("跳过：未配置 LLM key")
        return True
    cfg.set("modules.tts.enabled", False)

    char_name = cfg.get("character.name", "默认")
    card = CharacterCard.load(cfg.project_root / "data" / "characters" / char_name / "card.json")
    char_dir = cfg.char_data_dir()

    state = EmotionState.load(char_dir / "state.json")
    orch = Orchestrator(
        config=cfg, cognition=Cognition(cfg), speaker=Speaker(cfg, name=card.name),
        card=card, state=state, gatekeeper=GateKeeper(cfg),
        consciousness=ConsciousnessLog(char_dir / "consciousness.jsonl"),
    )
    orch_task = asyncio.create_task(orch.run())

    # 接合点：把 orchestrator 的 enqueue 注入 admin（main.py 里做的事）
    admin.set_enqueue(orch.enqueue)

    before = len(admin._chat_history_data(limit=999))

    # 走 admin 的发送路由（和前端 POST /api/chat/send 完全相同的代码路径）
    print("通过 admin.send_chat 发送 → 真实 LLM 回复…")
    await admin.send_chat(admin.ChatSend(text="给我讲个超短的笑话"))
    await orch._queue.join()

    orch_task.cancel()

    after = admin._chat_history_data(limit=999)
    added = len(after) - before
    ok = added >= 2
    print(f"  通过 get_chat_history 读回，新增 {added} 条（期望 >=2）: {'OK' if ok else 'FAIL'}")
    if ok:
        for m in after[-2:]:
            print(f"    [{m['role']}] {m['text'][:40]}")
    return ok


if __name__ == "__main__":
    ok = asyncio.run(main())
    sys.exit(0 if ok else 1)
