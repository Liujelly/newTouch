"""编排器 / 认知协程 (架构文档 5.2)。

阶段2: 反应路径 (USER_SPEECH) + 主动路径 (HEARTBEAT)。
主动路径流程: 情绪tick → GateKeeper硬闸门 → 内心独白 → (沉默?) → 出声。
旁听/视觉/日程路径的分发桩已留好，后续阶段填充。
"""
from __future__ import annotations

import asyncio
import json
import random
import time
from datetime import datetime

from .character import CharacterCard, WorldInfoManager, load_preset
from .cognition import Cognition
from .config import Config
from .consciousness import ConsciousnessLog
from .events import Event, EventType
from .gatekeeper import GateKeeper
from .state import EmotionState
from .memory.store import MemoryStore
from .action.speak import Speaker, load_voice_emotions, apply_voice_model, strip_leading_time_marker
from .sprite.store import load_face_emotions
from .logger import get_logger

log = get_logger("orch")


def _elapsed_desc(seconds: float) -> str:
    if seconds < 90:
        return "不到两分钟"
    if seconds < 3600:
        return f"{int(seconds // 60)}分钟"
    if seconds < 86400:
        return f"{int(seconds // 3600)}小时"
    return f"{int(seconds // 86400)}天"


# 主动独白的"思考切入点"，随机选一个作柔性引导，避免每次都从"沉默"起手
_THINK_SEEDS = [
    "突然想起ta之前提过的某件事",
    "好奇ta这会儿在忙什么",
    "冒出一个想跟ta分享的念头或小发现",
    "单纯有点想ta了",
    "回想你们上次聊到一半的话题",
    "想到一件可能对ta有用/ta会感兴趣的事",
    "只是想确认ta过得好不好",
]


def _time_context(now) -> str:
    """把当前时间转成口语化情境 + 精确钟点，如'工作日深夜 23:42'/'周末午后 14:28'。now 为 datetime。"""
    h = now.hour
    if 5 <= h < 8:
        seg = "清晨"
    elif 8 <= h < 12:
        seg = "上午"
    elif 12 <= h < 14:
        seg = "中午"
    elif 14 <= h < 18:
        seg = "下午"
    elif 18 <= h < 23:
        seg = "晚上"
    else:
        seg = "深夜"
    day = "周末" if now.weekday() >= 5 else "工作日"
    return f"{day}{seg} {now.hour}:{now.minute:02d}"


class Orchestrator:
    def __init__(
        self,
        config: Config,
        cognition: Cognition,
        speaker: Speaker,
        card: CharacterCard,
        state: EmotionState,
        gatekeeper: GateKeeper,
        consciousness: ConsciousnessLog,
        ready: asyncio.Event | None = None,
        vision=None,
    ):
        self._cfg = config
        self._cognition = cognition
        self._speaker = speaker
        self._card = card
        self._state = state
        self._gate = gatekeeper
        self._log = consciousness
        self._ready = ready
        self._vision = vision
        self._queue: asyncio.PriorityQueue[Event] = asyncio.PriorityQueue()
        self._cognition_lock = asyncio.Lock()
        self._chat_history: list[dict] = []
        self._user_name = config.get("user_persona.name", "你")
        # 角色专属数据目录（按角色名隔离 state/意识流/聊天历史）
        self._char_dir = config.char_data_dir()
        self._state_path = self._char_dir / "state.json"
        self._chat_log_path = self._char_dir / "chat_history.jsonl"
        self._last_vision = ""
        self._history_sink = None
        self._running = False
        self._awaiting_reply = False  # 主动发言后等用户回应；无回应时触发 on_ignored
        self._last_proactive_ts = 0.0  # 上次主动发言的时间戳，用于"未回应"最小等待判定
        # 未回应宽限 / 思考后冷却 / 近期内心活动条数 均在使用处现读 config（改完即生效，无需重启）。
        # 思考后冷却：调了 LLM 却没开口/没说后，歇一段再触发下一次思考，避免"想了没说→下一拍
        # (20~30s)立刻又冒出来/改口"的突兀感。与 speak 长冷却(gatekeeper)完全独立，复用
        # proactive.min_interval_seconds 配置值。仅 silent/look-未说 后生效，speak 不受此限。
        self._last_think_ts = 0.0
        self._last_vision_check = 0.0  # 上次视觉触发 LLM 判断的时间戳（独立节流）
        # A: 最近的"内心活动"滚动缓冲（含未说出口的 thought + look/speak 结局），
        # 注入下次独白，让 ta 的思绪连贯——记得自己刚才想过什么、看了/说了没。不进长期记忆。
        self._recent_inner: list[str] = []
        # 短期对话窗口长度（条数，user+assistant 各算一条）。可在 config/管理平台调。
        self._max_history = config.get("memory.chat_history_window", 40)
        # 增量压缩：窗口溢出达一批(_compress_batch)时，把最旧一批 LLM 浓缩进 _earlier_summary，
        # 而非直接丢。每丢一批才调一次 LLM（非每轮），保留早期脉络。
        self._compress_batch = config.get("memory.compress_batch_size", 10)
        self._earlier_summary = ""
        self._wi = WorldInfoManager(config, card)
        self._memory = MemoryStore(config)
        # 黏人度同步到闸门：角色卡 extensions.clinginess 优先，无则用全局默认
        self._sync_clinginess()
        # 重启续聊：从磁盘回填最近的对话历史到短期窗口
        self._load_recent_history()

    def _sync_clinginess(self) -> None:
        """把当前角色卡的 extensions.clinginess 同步给 GateKeeper（决定被冷落时该催还是该退）。
        角色卡没配该字段时回退全局 proactive.clinginess（set_clinginess 内部兜底）。"""
        ext = self._card.extensions or {}
        val = ext.get("clinginess")
        if val is None:
            val = self._cfg.get("proactive.clinginess", 0.5)
        self._gate.set_clinginess(val)

    def _get_reply_lang_config(self) -> tuple[str, str]:
        """获取回复语言和翻译语言配置。

        优先从角色卡 extensions 读取，没有则回退到全局配置。

        Returns:
            (reply_lang, translation_lang) 元组
        """
        ext = self._card.extensions or {}

        # 回复语言：优先角色卡，无则回退 TTS 语言，再无则中文
        reply_lang = ext.get("reply_lang")
        if not reply_lang:
            reply_lang = self._cfg.get("modules.tts.text_lang", "zh")

        # 翻译语言：优先角色卡，无则回退全局配置，再无则空（不翻译）
        translation_lang = ext.get("translation_lang")
        if not translation_lang:
            translation_lang = self._cfg.get("character.default_translation_lang", "")

        return reply_lang, translation_lang

    def _face_emotions(self) -> list[str]:
        """当前角色立绘库有哪些表情档（库驱动，供 prompt 列出可选表情）。空库返回 []。

        用目录名（config.character.name，隔离键）找立绘库，不用 card.name（展示名）——
        立绘库按角色目录存（data/characters/{目录名}/），目录名与展示名可能不同（v2.21）。
        """
        char_dir_name = self._cfg.get("character.name", "默认")
        return load_face_emotions(self._cfg, char_dir_name)

    def _load_recent_history(self) -> None:
        """从 chat_history.jsonl 尾部读最近 _max_history 条回填短期窗口，实现重启续聊。

        落盘格式 {ts,role,text}，短期窗口用 {role,content}，此处做字段转换。
        只回填 user/assistant；旁听/视觉等注入行不落盘故天然不在内。失败静默降级为空。
        """
        if self._max_history <= 0 or not self._chat_log_path.exists():
            return
        try:
            lines = self._chat_log_path.read_text(encoding="utf-8").strip().splitlines()
            recent = []
            for ln in lines[-self._max_history:]:
                try:
                    e = json.loads(ln)
                except (json.JSONDecodeError, ValueError):
                    continue
                if e.get("role") in ("user", "assistant") and e.get("text"):
                    recent.append({"role": e["role"], "content": e["text"], "ts": e.get("ts", "")})
            self._chat_history = recent[-self._max_history:]
            if self._chat_history:
                log.info("回填短期对话窗口 %d 条（重启续聊）", len(self._chat_history))
        except OSError:
            pass

    async def _compact_history(self) -> None:
        """短期窗口增量压缩：溢出达一批时把最旧那批浓缩进 _earlier_summary。

        批处理触发（非每轮）：允许窗口涨到 max+batch，溢出达 batch 才压一次，
        每 ~batch/2 轮对话调一次 LLM（反应路径每轮 +2 条），不拖慢日常对话。
        压缩失败/禁用 → 降级直接丢最旧 batch（等于现行为），不报错。
        """
        overflow = len(self._chat_history) - self._max_history
        if overflow < self._compress_batch:
            return
        batch = self._chat_history[:self._compress_batch]
        self._chat_history = self._chat_history[self._compress_batch:]

        # 只有启用压缩且 LLM 可用时才尝试摘要；否则已完成"直接丢"降级
        if not self._cfg.get("memory.compress_enabled", True):
            return
        batch_text = "\n".join(
            f"{m['role']}: {m.get('content', '')}" for m in batch
        )
        try:
            async with self._cognition_lock:
                summary = await self._cognition.summarize_history(
                    batch_text, prev_summary=self._earlier_summary
                )
            if summary:
                self._earlier_summary = summary
        except Exception as e:  # noqa: BLE001
            log.warning("历史压缩失败，已直接丢弃: %s", e)

    def _log_chat(self, role: str, text: str) -> None:
        """追加一条对话记录到 chat_history.jsonl。"""
        if not text:
            return
        entry = {"ts": datetime.now().isoformat(timespec="seconds"),
                 "role": role, "text": text}
        with open(self._chat_log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def bind_history_sink(self, sink) -> None:
        self._history_sink = sink

    def bind_character_refresh(self, callback) -> None:
        """绑定角色切换时的回调（用于刷新 MicInput 的唤醒词等）。"""
        self._char_refresh_callback = callback

    def reload_card(self) -> None:
        """热重载当前角色卡（管理平台保存角色卡后调用，立即生效语言等设置）。"""
        card_path = self._cfg.project_root / "data" / "characters" / self._cfg.get("character.name", "默认") / "card.json"
        if card_path.exists():
            self._card = CharacterCard.load(card_path)
            # 重建世界书管理器（可能改了 extensions.world）
            self._wi = WorldInfoManager(self._cfg, self._card)
            # 同步黏人度（可能改了 extensions.clinginess）
            self._sync_clinginess()
            log.info("角色卡已热重载: %s", self._card.name)
        else:
            log.warning("角色卡文件不存在，跳过重载")

    def switch_character(self, new_name: str) -> None:
        """运行期切换角色（admin 改 character.name 后调用）。

        保存当前状态 → 加载新角色卡/世界书/路径/state → 清空短期对话历史。
        cognition/speaker/gatekeeper 不变（只跟 provider 绑定，不跟角色绑定）。
        """
        # 保存当前角色的情绪状态
        self._state.save(self._state_path)

        # 更新角色专属目录
        self._char_dir = self._cfg.char_data_dir(new_name)
        self._state_path = self._char_dir / "state.json"
        self._chat_log_path = self._char_dir / "chat_history.jsonl"

        # 重建意识流日志（指向新角色目录）
        self._log = ConsciousnessLog(self._char_dir / "consciousness.jsonl")

        # 加载新角色卡
        card_path = self._cfg.project_root / "data" / "characters" / new_name / "card.json"
        if card_path.exists():
            self._card = CharacterCard.load(card_path)
        else:
            log.warning("角色卡 %s/card.json 不存在，保留当前角色", new_name)

        # 重建世界书管理器（绑定新角色卡）
        self._wi = WorldInfoManager(self._cfg, self._card)
        # 同步新角色的黏人度到闸门
        self._sync_clinginess()

        # 从磁盘加载新角色的情绪状态（不存在则重置）
        self._state = EmotionState.load(self._state_path)

        # 加载新角色的短期对话历史（从该角色的 chat_history.jsonl 回填，续聊）
        self._max_history = self._cfg.get("memory.chat_history_window", 40)
        self._chat_history = []
        self._load_recent_history()

        log.info("已切换到角色: %s", new_name)
        # 切角色后若 TTS 开启，fire-and-forget 应用绑定语音库的模型权重
        if self._cfg.get("modules.tts.enabled", True):
            asyncio.create_task(apply_voice_model(self._cfg))

        # 通知输入源（MicInput）刷新唤醒词
        if hasattr(self, '_char_refresh_callback') and self._char_refresh_callback:
            self._char_refresh_callback(new_name)

    async def enqueue(self, event: Event) -> None:
        await self._queue.put(event)

    async def run(self) -> None:
        self._running = True
        if self._ready:
            self._ready.set()
        while self._running:
            event = await self._queue.get()
            try:
                await self._dispatch(event)
            except Exception as e:  # noqa: BLE001
                log.error("事件分发异常: %s", e)
            finally:
                self._queue.task_done()
                if self._history_sink:
                    self._history_sink(self._chat_history)
                if self._ready and event.type == EventType.USER_SPEECH:
                    self._ready.set()

    async def shutdown(self) -> None:
        self._running = False
        self._state.save(self._state_path)

    async def _dispatch(self, event: Event) -> None:
        if event.type == EventType.USER_SPEECH:
            await self._handle_reactive(event)
        elif event.type == EventType.HEARTBEAT:
            await self._handle_proactive(event)
        elif event.type == EventType.EAVESDROP:
            await self._handle_eavesdrop(event)
        elif event.type == EventType.VISION_CHANGE:
            await self._handle_vision(event)
        elif event.type == EventType.VISION_SIGNIFICANT:
            await self._handle_vision_significant(event)

    async def _handle_eavesdrop(self, event: Event) -> None:
        text = (event.payload.get("text") or "").strip()
        if not text:
            return
        cls = event.payload.get("classification", "other")
        self._chat_history.append({"role": "user", "content": f"[旁听·{cls}] {text}", "ts": datetime.now().isoformat(timespec="seconds")})
        await self._compact_history()
        self._log.record(trigger=f"旁听·{cls}", action="silent",
                         text=text, emotion=self._state.snapshot(), gate="旁听记录，不回应")

    async def _handle_vision(self, event: Event) -> None:
        caption = event.payload.get("caption", "")
        if not caption:
            return
        self._last_vision = caption
        # 标记「自动抓取」：被动帧差抓的画面，可能已过时；LLM 要看"现在"应调 look 工具
        self._chat_history.append({"role": "user", "content": f"（系统自动抓取到画面：{caption}。注意：这是被动抓取，可能已过时；要看此刻画面请调 look 工具）", "ts": datetime.now().isoformat(timespec="seconds")})
        await self._compact_history()
        self._log.record(trigger="视觉", action="silent",
                         text=caption, emotion=self._state.snapshot(), gate="画面变化已记录")

    async def _handle_vision_significant(self, event: Event) -> None:
        """显著视觉变化：注入 caption + 立刻调 LLM 智能判断要不要开口。

        与心跳路径的区别：
        - 不走 gatekeeper（不消耗每小时主动配额，不受间隔/backoff/上限约束）
        - 独立节流（perception.vision.min_check_interval_s，默认60s），防"人一直在画面里"频繁触发
        - 只保留勿扰时段拦截（可选）
        - speak 后不调 gatekeeper.record_spoke()，不影响主动发言计数
        """
        caption = event.payload.get("caption", "")
        if not caption:
            return

        # caption 先注入短期窗口（无论是否开口，视觉素材都留给后续消化）
        # 标记「自动抓取」：被动显著变化抓的画面，可能已过时；LLM 要看"现在"应调 look 工具
        self._last_vision = caption
        self._chat_history.append({
            "role": "user",
            "content": f"（系统自动抓取到画面：{caption}。注意：这是被动抓取，可能已过时；要看此刻画面请调 look 工具）",
            "ts": datetime.now().isoformat(timespec="seconds"),
        })
        await self._compact_history()

        # 独立节流：距上次视觉 LLM 判断不足 min_check_interval_s，跳过本次判断
        now = time.time()
        min_check = self._cfg.get("perception.vision.min_check_interval_s", 60)
        if now - self._last_vision_check < min_check:
            self._log.record(trigger="视觉·显著变化", action="silent",
                             text=caption, emotion=self._state.snapshot(),
                             gate=f"视觉判断冷却中({int(now - self._last_vision_check)}/{int(min_check)}s)")
            return

        # 勿扰时段：默认遵守（半夜不打扰）
        if self._gate._in_quiet_hours():
            self._log.record(trigger="视觉·显著变化", action="silent",
                             text=caption, emotion=self._state.snapshot(), gate="勿扰时段")
            return

        self._last_vision_check = now

        # diff_score 越高说明变化越明显，用于给 LLM 提示紧迫度
        diff_score = event.payload.get("diff_score", 0.0)
        urgency = "（这是比较明显的变化，可能值得回应）" if diff_score > 0.5 else "（也可以继续观察不说话）"
        trigger_reason = f"你看到了：{caption}。{urgency}"

        preset = load_preset(self._cfg)
        reply_lang, translation_lang = self._get_reply_lang_config()
        voice_emotions = load_voice_emotions(self._cfg)
        face_emotions = self._face_emotions()
        elapsed = time.time() - self._state.last_interaction
        time_context = _time_context(datetime.now())

        async with self._cognition_lock:
            result = await self._cognition.proactive_think(
                card=self._card, user_name=self._user_name,
                trigger_reason=trigger_reason,
                emotion_summary=self._state.summary(),
                chat_history=self._chat_history,
                elapsed_desc=_elapsed_desc(elapsed), can_look=False,
                preset=preset, reply_lang=reply_lang, translation_lang=translation_lang,
                voice_emotions=voice_emotions,
                face_emotions=face_emotions,
                recent_inner=list(self._recent_inner),
                time_context=time_context,
            )

        thought = result.get("thought", "")
        reply = result.get("text", "")
        # 兜底扒掉开头被复读的系统时间标记 [X分钟前]（LLM 照抄历史标记进独白 text）
        reply = strip_leading_time_marker(reply)

        if result.get("action") != "speak" or not reply:
            self._log.record(trigger="视觉·显著变化", action="silent", thought=thought,
                             emotion=self._state.snapshot(), gate="ta观察后选择不说")
            self._record_inner("silent", thought)
            return

        self._log.record(trigger="视觉·显著变化", action="speak",
                         thought=thought, text=reply, emotion=self._state.snapshot())
        self._log_chat("assistant", reply)

        async def _once():
            yield reply

        await self._speaker.speak(_once(), emotion=result.get("emotion") or None,
                                 face=result.get("face") or None,
                                 translation_lang=translation_lang)

        delta = result.get("emotion_delta") or {}
        if delta:
            self._state.apply_delta(delta)
        self._state.on_proactive_spoke()
        self._awaiting_reply = True
        self._last_proactive_ts = time.time()
        self._state.save(self._state_path)

        self._memory.add(
            f"我看到{caption}后对{self._user_name}说:{reply}",
            self._state.snapshot(), tags=["对话", "主动", "视觉"],
        )
        self._chat_history.append({
            "role": "assistant", "content": reply,
            "ts": datetime.now().isoformat(timespec="seconds"),
        })
        self._record_inner("speak", thought, reply)
        await self._compact_history()

    async def _handle_reactive(self, event: Event) -> None:
        user_text = (event.payload.get("text") or "").strip()
        if not user_text:
            return
        if self._speaker.is_speaking():
            self._speaker.interrupt()

        # 用户发言第一时间落盘：必须在记忆/世界书/LLM 之前。后续 recall/Wi/LLM 耗时
        # 数秒，若 user 行等它们之后才落盘，前端 sendChat 第一轮轮询拉到的历史还没有
        # 这条 user，会用旧列表覆盖掉乐观显示 → 用户消息"先消失、再和回复一起出现"。
        self._log_chat("user", user_text)

        # 注：反应路径"主动看"已改成 look 工具（v2.59）——LLM 在 react_stream 里自主
        # 决定调不调，不再靠关键词预筛拦截。问"看看我"→调 look；问"看天气"→调 get_weather。
        wi_result = {"before": [], "after": [], "depth": []}
        if self._wi.has_books():
            scan = "\n".join(m.get("content", "") for m in self._chat_history[-4:]) + "\n" + user_text
            wi_result = self._wi.activate(scan, round_no=len(self._chat_history) // 2)

        # 反应路径记忆注入开关（memory.reactive_auto_recall，默认开=现状）：
        # 开→每句自动 recall 注入 system；关→不自动注入，靠 LLM 调 memory_search 工具。
        if self._cfg.get("memory.reactive_auto_recall", True):
            memories = self._memory.recall(user_text, limit=3)
        else:
            memories = []

        preset = load_preset(self._cfg)
        reply_lang, translation_lang = self._get_reply_lang_config()
        time_context = _time_context(datetime.now())
        # 用户发言已在上方第一时间落盘，此处不再重复。

        # on_text：文本生成完毕即触发（此时语音可能还在播放），立即写 assistant 行，
        # 聊天页 3s 轮询即可刷出完整回复，不必等整段语音放完。
        logged = {"done": False}

        def _on_reply_text(reply_text: str) -> None:
            if logged["done"] or not reply_text:
                return
            logged["done"] = True
            self._log_chat("assistant", reply_text)

        async with self._cognition_lock:
            stream = self._cognition.react_stream(
                self._card, self._user_name, user_text, self._chat_history,
                user_persona=self._cfg.get("user_persona.description", ""),
                world_before=wi_result["before"], world_after=wi_result["after"],
                emotion_summary=self._state.summary(), memories=memories,
                preset=preset, reply_lang=reply_lang, translation_lang=translation_lang,
                voice_emotions=load_voice_emotions(self._cfg),
                face_emotions=self._face_emotions(),
                earlier_summary=self._earlier_summary,
                time_context=time_context,
            )
            reply = await self._speaker.speak(stream, on_text=_on_reply_text,
                                              translation_lang=translation_lang)
        self._state.on_interaction(positive=True)
        self._awaiting_reply = False  # 用户回应了，清除等待标记
        self._state.save(self._state_path)
        self._gate.record_interaction()  # 反应路径：刷新间隔但不计入每小时主动上限
        _ts = datetime.now().isoformat(timespec="seconds")
        self._chat_history.append({"role": "user", "content": user_text, "ts": _ts})
        self._chat_history.append({"role": "assistant", "content": reply, "ts": _ts})
        await self._compact_history()
        # 兜底：on_text 未触发（如空回复被跳过）时这里补写
        if not logged["done"]:
            self._log_chat("assistant", reply)
        self._memory.add(
            f"{self._user_name}说:{user_text} / 我回:{reply}",
            self._state.snapshot(), tags=["对话"],
        )
        # 情绪接线（反应路径）：fire-and-forget 跑一次轻量 LLM 判断这轮对话对情绪的影响，
        # 不阻塞下一轮对话。最多晚几秒情绪才更新，对陪伴体可接受。失败静默不改情绪。
        if reply:
            asyncio.create_task(self._apply_emotion_from_reply(user_text, reply))

    async def _apply_emotion_from_reply(self, user_text: str, reply: str) -> None:
        """反应路径异步情绪更新：判断这轮对话的情绪增量并应用。失败静默。"""
        try:
            delta = await self._cognition.judge_emotion_delta(user_text, reply)
        except Exception as e:  # noqa: BLE001
            log.warning("情绪判断失败（已忽略）: %s", e)
            return
        if delta:
            self._state.apply_delta(delta)
            self._state.save(self._state_path)

    def _record_inner(self, kind: str, thought: str, text: str = "") -> None:
        """把一次主动思考的结局摘要追加进近期内心活动缓冲（A：让下次思考延续思绪）。

        kind: "silent"(想了没说) / "look"(看了一眼) / "speak"(开口说了)。
        只保留最近 _recent_inner_max 条，注入下次独白 prompt。不落盘（重启随对话重建）。
        thought 截断防过长；speak 额外带上说了什么。
        """
        th = (thought or "").strip()
        if not th and not text:
            return
        th = th[:80]
        if kind == "speak":
            entry = f"刚开口说了「{(text or '').strip()[:40]}」（当时想：{th}）" if th else f"刚开口说了「{(text or '').strip()[:40]}」"
        elif kind == "look":
            entry = f"刚想看看ta在干嘛（{th}）" if th else "刚想看看ta在干嘛"
        else:  # silent
            entry = f"想了想没开口（{th}）" if th else "想了想没开口"
        self._recent_inner.append(entry)
        cap = self._cfg.get("proactive.recent_inner_max", 5)
        if len(self._recent_inner) > cap:
            self._recent_inner = self._recent_inner[-cap:] if cap > 0 else []

    async def _handle_proactive(self, event: Event) -> None:
        self._state.tick()

        # 未回应闭环：上次主动发言至今没等到用户回应 → 情绪递进 + 拉长冷却。
        # 但必须距上次主动发言超过宽限期才算"被无视"，否则说完下一拍心跳
        # （间隔仅 20~30s）就误判被无视、worry 立刻飙升，不合理。现读 config（改完即生效）。
        ignored_grace = self._cfg.get("proactive.ignored_grace_seconds", 300)
        if self._awaiting_reply and (time.time() - self._last_proactive_ts) >= ignored_grace:
            self._state.on_ignored()
            self._gate.record_ignored()
            self._awaiting_reply = False

        allowed, reason = self._gate.check(self._state)
        if not allowed:
            self._log.record(trigger="心跳", action="silent",
                             emotion=self._state.snapshot(), gate=reason)
            return

        threshold = self._cfg.get("proactive.loneliness_threshold", 0.4)
        if self._state.loneliness < threshold:
            self._log.record(trigger="心跳", action="silent",
                             emotion=self._state.snapshot(),
                             gate=f"孤独感{self._state.loneliness:.2f}<{threshold}")
            return

        # B：思考后冷却。上一次主动思考（任何结局：silent/look/speak）后，独立的短冷却内
        # 不再调 LLM 思考，避免"想了没说→下一拍(20~30s)又冒出来甚至改口说话"的突兀感。
        # 与 speak 的长冷却(gatekeeper)分离：用独立字段 _last_think_ts，不碰 _last_spoke /
        # 每小时上限 / backoff，故"想了没说"不会被罚长冷却、也不推迟下一次 speak。
        # 冷却时长复用 proactive.min_interval_seconds（与 speak 冷却同一配置项，不新增字段）。
        think_cd = self._cfg.get("proactive.min_interval_seconds", 900)
        since_think = time.time() - self._last_think_ts
        if since_think < think_cd:
            self._log.record(trigger="心跳", action="silent",
                             emotion=self._state.snapshot(),
                             gate=f"思考冷却中({int(since_think)}/{int(think_cd)}s)")
            return

        elapsed = time.time() - self._state.last_interaction
        can_look = (self._vision is not None
                    and self._cfg.get("perception.vision.enabled", False))
        preset = load_preset(self._cfg)
        reply_lang, translation_lang = self._get_reply_lang_config()
        voice_emotions = load_voice_emotions(self._cfg)
        face_emotions = self._face_emotions()

        # 被忽略次数：提前获取 base_reason，供后续路线 B 使用
        base_reason = event.payload.get("reason", "心跳")

        # 主动独白素材：长期记忆检索
        # 读取开关：是否用 LLM 生成精准的检索 query（路线 B）
        use_query_gen = self._cfg.get("token_intensive.memory_query_generation", False)

        if use_query_gen:
            # 路线 B：LLM 生成检索问题
            context = f"{base_reason}·孤独感{self._state.loneliness:.2f}"
            queries = self._memory.generate_recall_queries(self._user_name, context=context)
            if queries:
                # 多 query 检索，每个 query 取 2 条，去重后最多约 3-6 条
                memories = self._memory.recall_multi_query(queries, limit_per_query=2)
                # 限制总数，避免注入过多
                memories = memories[:5]
            else:
                # LLM 生成失败，降级到路线 A
                recent_user = next((m["content"] for m in reversed(self._chat_history)
                                    if m.get("role") == "user"), "")
                recall_q = recent_user or f"{self._user_name} 最近 喜好 日常"
                memories = self._memory.recall(recall_q, limit=3)
        else:
            # 路线 A（当前方案）：用最近 user 发言做 query，没有则泛查
            recent_user = next((m["content"] for m in reversed(self._chat_history)
                                if m.get("role") == "user"), "")
            recall_q = recent_user or f"{self._user_name} 最近 喜好 日常"
            memories = self._memory.recall(recall_q, limit=3)

        time_context = _time_context(datetime.now())
        think_seed = random.choice(_THINK_SEEDS)
        # 被忽略次数写进 trigger_reason，让 LLM 按人设演绎"越来越担心/失落/生气"
        n = self._state.consecutive_ignored
        trigger_reason = (
            f"{base_reason}（你已经主动找过{n}次，对方都没有回应）" if n > 0 else base_reason
        )
        # 过了所有闸、确定要思考：记下思考时间戳（B 的独立短冷却起点）。
        # 无论这次思考结局是 silent/look/speak，都从此刻起算 think 冷却。
        self._last_think_ts = time.time()
        async with self._cognition_lock:
            result = await self._cognition.proactive_think(
                card=self._card, user_name=self._user_name,
                trigger_reason=trigger_reason,
                emotion_summary=self._state.summary(),
                chat_history=self._chat_history,
                elapsed_desc=_elapsed_desc(elapsed), can_look=can_look,
                preset=preset, reply_lang=reply_lang, translation_lang=translation_lang,
                voice_emotions=voice_emotions,
                face_emotions=face_emotions,
                memories=memories, earlier_summary=self._earlier_summary,
                time_context=time_context, think_seed=think_seed,
                recent_inner=list(self._recent_inner),
            )
        thought = result.get("thought", "")

        if result.get("action") == "look":
            self._log.record(trigger="心跳·主动看", action="look",
                             thought=thought, emotion=self._state.snapshot())
            vc = await self._vision.look_now()
            if vc:
                caption = vc.caption
                self._last_vision = caption
                look_ctx = self._chat_history + [{"role": "user", "content": f"（你看到了：{caption}）", "ts": datetime.now().isoformat(timespec="seconds")}]
                async with self._cognition_lock:
                    result = await self._cognition.proactive_think(
                        card=self._card, user_name=self._user_name,
                        trigger_reason=f"刚主动看了一眼：{caption}",
                        emotion_summary=self._state.summary(),
                        chat_history=look_ctx,
                        elapsed_desc=_elapsed_desc(elapsed), can_look=False,
                        preset=preset, reply_lang=reply_lang, translation_lang=translation_lang,
                        voice_emotions=voice_emotions,
                        face_emotions=face_emotions,
                        memories=memories, earlier_summary=self._earlier_summary,
                        time_context=time_context, think_seed=think_seed,
                    )
                thought = result.get("thought", "")
            else:
                result = {"action": "silent", "thought": thought, "text": ""}

        reply = result.get("text", "")
        # 兜底扒掉开头被复读的系统时间标记 [X分钟前]（LLM 照抄历史标记进独白 text）
        reply = strip_leading_time_marker(reply)

        if result.get("action") != "speak" or not reply:
            self._log.record(trigger="心跳", action="silent", thought=thought,
                             emotion=self._state.snapshot(), gate="ta选择不说")
            self._record_inner("silent", thought)
            return

        allowed, reason = self._gate.check(self._state)
        if not allowed:
            self._log.record(trigger="心跳", action="silent", thought=thought,
                             emotion=self._state.snapshot(), gate=f"末层兜底:{reason}")
            self._record_inner("silent", thought)
            return

        self._log.record(trigger="心跳", action="speak", thought=thought,
                         text=reply, emotion=self._state.snapshot())

        async def _once():
            yield reply

        # 主动发言文本已知，先落盘（语音播放期间管理平台即可见），再播放
        self._log_chat("assistant", reply)
        await self._speaker.speak(_once(), emotion=result.get("emotion") or None,
                                 face=result.get("face") or None,
                                 translation_lang=translation_lang)
        self._gate.record_spoke()
        # 独白自带的情绪增量（这次内心活动让 ta 情绪如何变化）→ 直接 apply，零额外调用
        delta = result.get("emotion_delta") or {}
        if delta:
            self._state.apply_delta(delta)
        self._state.on_proactive_spoke()   # 小增依恋，但不清零孤独感/未回应计数
        self._awaiting_reply = True         # 等待用户回应；无回应时下次心跳触发 on_ignored
        self._last_proactive_ts = time.time()  # 记主动发言时刻，未回应判定需距此超 _ignored_grace
        self._state.save(self._state_path)
        # 主动发言写记忆：ta 才记得自己主动关心过你（带触发原因+情绪快照）
        trigger = event.payload.get("reason", "心跳")
        self._memory.add(
            f"我主动对{self._user_name}说:{reply}（触发:{trigger}）",
            self._state.snapshot(), tags=["对话", "主动"],
        )
        self._chat_history.append({"role": "assistant", "content": reply, "ts": datetime.now().isoformat(timespec="seconds")})
        self._record_inner("speak", thought, reply)
        await self._compact_history()
