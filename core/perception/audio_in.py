"""音频输入与对话对象分流 (架构文档 5.6 / 4.4)。

两种输入源:
  TextInput  — 文本输入 (无音频依赖)
  MicInput   — 麦克风: Silero VAD + faster-whisper STT + 三步分流

三步分流 (classify):
  1. 唤醒词/称呼匹配  (本地免费) → @assistant + 开启持续对话窗口
  2. 持续对话窗口内   (本地免费) → @assistant 默认
  3. LLM 快速分类    (廉价)     → @assistant / @other / @self / @ignore

@assistant → USER_SPEECH (URGENT)
@other / @self → EAVESDROP (NORMAL)
@ignore → 丢弃
"""
from __future__ import annotations

import asyncio
import time
from enum import Enum

from ..config import Config
from ..events import Event, EventPriority, EventType, user_speech
from ..logger import get_logger

log = get_logger("audio")


# ─────────────────── TextInput ────────────────────────────────────────────

class TextInput:
    def __init__(self, config: Config, enqueue, ready: asyncio.Event | None = None):
        self._cfg = config
        self._enqueue = enqueue
        self._ready = ready
        self._running = False

    async def start(self) -> None:
        self._running = True
        print("【文本模式】输入文字和 ta 对话，/quit 退出。\n")
        while self._running:
            if self._ready:
                await self._ready.wait()
                self._ready.clear()
            try:
                text = await asyncio.to_thread(input, "\n你 > ")
            except (EOFError, KeyboardInterrupt):
                break
            text = text.strip()
            if not text:
                if self._ready:
                    self._ready.set()
                continue
            if text in ("/quit", "/exit"):
                break
            await self._enqueue(user_speech(text))
        self._running = False

    def stop(self) -> None:
        self._running = False


# ─────────────────── 对话对象分类 ──────────────────────────────────────────

class AudioClassification(Enum):
    ASSISTANT = "assistant"
    OTHER = "other"
    SELF = "self"
    IGNORE = "ignore"


class Classifier:
    """三步决策树: 唤醒词 → 对话窗口 → LLM 分类。"""

    def __init__(self, config: Config, llm_client, llm_backend: str,
                 llm_model: str, char_name: str):
        self._cfg = config
        self._char_name = char_name
        # 唤醒词 = 角色名 + 全局通用词
        self._wake_words = self._build_wake_words()

        self._window_s: float = config.get(
            "perception.audio.dialog_window_seconds", 15
        )
        self._client = llm_client
        self._backend = llm_backend   # "anthropic" | "openai"
        self._model = llm_model
        self._window_until: float = 0.0

    def _build_wake_words(self) -> list[str]:
        """构建唤醒词列表：角色名 + 角色卡自定义 + 全局通用词。"""
        words = [self._char_name]

        # 尝试读取角色卡的自定义唤醒词
        try:
            from core.character import CharacterCard
            char_dir = self._cfg.char_data_dir(self._char_name)
            card_path = char_dir / "card.json"
            if card_path.exists():
                card = CharacterCard.load(card_path)
                if card.extensions and "wake_words" in card.extensions:
                    custom_words = card.extensions["wake_words"]
                    if isinstance(custom_words, list):
                        words.extend(custom_words)
        except Exception:
            pass  # 读取失败则跳过，不影响基础功能

        # 全局通用词
        global_words = self._cfg.get("perception.audio.global_wake_words", [])
        words.extend(global_words)

        # 去重（保持顺序）
        return list(dict.fromkeys(words))

    def refresh_character(self, new_char_name: str) -> None:
        """切换角色时更新唤醒词。"""
        self._char_name = new_char_name
        self._wake_words = self._build_wake_words()
        log.info("唤醒词已更新: %s", self._wake_words)

    def open_window(self) -> None:
        self._window_until = time.time() + self._window_s

    async def classify(self, text: str, chat_history: list[dict]) -> AudioClassification:
        """三步决策：唤醒词（完整词匹配）→ 对话窗口 → LLM 分类。"""
        text_lower = text.lower()

        # 唤醒词完整匹配：防止子串误触发（如"小触摸屏"、"newtouchpad"）
        # 策略：唤醒词前后不能是 ASCII 字母/数字（防止英文复合词），中文边界允许
        for w in self._wake_words:
            w_lower = w.lower()
            # 查找所有出现位置
            start = 0
            while True:
                pos = text_lower.find(w_lower, start)
                if pos == -1:
                    break

                # 检查前后边界：前后不能是 ASCII 字母/数字
                before_char = text_lower[pos-1] if pos > 0 else ''
                after_char = text_lower[pos + len(w_lower)] if pos + len(w_lower) < len(text_lower) else ''

                before_ok = (not before_char or not before_char.isascii() or not before_char.isalnum())
                after_ok = (not after_char or not after_char.isascii() or not after_char.isalnum())

                if before_ok and after_ok:
                    self.open_window()
                    return AudioClassification.ASSISTANT

                start = pos + 1

        if time.time() < self._window_until:
            return AudioClassification.ASSISTANT
        return await self._llm_classify(text, chat_history)

    async def _llm_classify(self, text: str, chat_history: list[dict]) -> AudioClassification:
        recent = "\n".join(f"{m['role']}: {m['content']}" for m in chat_history[-3:])
        prompt = (
            f"角色名: {self._char_name}\n"
            f"最近对话:\n{recent}\n\n"
            f"用户刚说: \"{text}\"\n\n"
            f"判断这句话的对话对象：\n"
            f"- assistant：明确在和{self._char_name}对话（喊名字、回应上文、提问、命令）\n"
            f"- other：在和别人说话（称呼第三方、话题明显不相关）\n"
            f"- self：自言自语（嘀咕、感叹、独白）\n"
            f"- ignore：背景音、电视声、不清晰的话\n\n"
            f"**判断原则**：只有明确的对话特征（称呼/回应/提问）才返回 assistant，"
            f"其他情况优先判断为 other/self/ignore，避免打扰。\n\n"
            f"只输出一个词: assistant / other / self / ignore"
        )
        try:
            if self._backend == "anthropic":
                resp = await self._client.messages.create(
                    model=self._model, max_tokens=10,
                    messages=[{"role": "user", "content": prompt}],
                )
                result = resp.content[0].text.strip().lower()
            else:
                resp = await self._client.chat.completions.create(
                    model=self._model, max_tokens=10,
                    messages=[{"role": "user", "content": prompt}],
                )
                result = (resp.choices[0].message.content or "").strip().lower()
        except Exception:  # noqa: BLE001
            return AudioClassification.ASSISTANT  # 出错保守走反应路径

        if result.startswith("other"):
            return AudioClassification.OTHER
        if result.startswith("self"):
            return AudioClassification.SELF
        if result.startswith("ignore"):
            return AudioClassification.IGNORE
        return AudioClassification.ASSISTANT


# ─────────────────── MicInput ──────────────────────────────────────────────

class MicInput:
    """麦克风输入: 连续流式 VAD 缓冲 + faster-whisper STT + 三步分流。

    半双工门控 (speaker)：TTS 播放期间及播放后 echo_cooldown 秒内，丢弃录到的音频，
    避免麦克风录到 ta 自己的语音造成自反馈循环。
    """

    def __init__(self, config: Config, enqueue, classifier: Classifier,
                 ready: asyncio.Event | None = None, speaker=None):
        self._cfg = config
        self._enqueue = enqueue
        self._classifier = classifier
        self._ready = ready
        self._speaker = speaker  # 有 is_speaking() 的 Speaker，用于半双工门控
        self._echo_cooldown = config.get("perception.audio.echo_cooldown_s", 0.6)
        self._muted_until = 0.0  # 播放结束后的静音截止时间
        self._sr = config.get("perception.audio.sample_rate", 16000)
        self._device = config.get("perception.audio.device_index", None)
        self._chat_history: list[dict] = []
        self._running = False

    def update_history(self, history: list[dict]) -> None:
        self._chat_history = history

    def refresh_character(self, new_char_name: str) -> None:
        """切换角色时刷新分类器的唤醒词。"""
        self._classifier.refresh_character(new_char_name)

    def _is_muted(self) -> bool:
        """TTS 正在播放、或刚播完冷却期内 → 静音麦克风。"""
        if self._speaker is not None and self._speaker.is_speaking():
            self._muted_until = time.time() + self._echo_cooldown
            return True
        return time.time() < self._muted_until

    async def start(self) -> None:
        try:
            import torch
            import sounddevice as sd
            from silero_vad import load_silero_vad
            import numpy as np
        except ImportError as e:
            log.error("缺少依赖，麦克风不可用: %s", e)
            return

        # STT 引擎按 modules.stt.provider 选择（faster-whisper / funasr），加载失败则降级
        from .stt import load_stt_engine
        stt_engine = load_stt_engine(self._cfg)
        if stt_engine is None:
            log.warning("STT 引擎不可用，麦克风停用")
            return

        vad_model = load_silero_vad()
        lang = self._cfg.get("modules.stt.language", "zh")
        vad_threshold = self._cfg.get("perception.vad.threshold", 0.5)
        min_silence_ms = self._cfg.get("perception.vad.min_silence_ms", 800)
        frame_samples = 512  # silero vad 要求 512 采样点/帧

        self._running = True
        log.info("麦克风输入就绪，持续监听中…")

        speech_buf: list[np.ndarray] = []  # 累积的语音帧
        silence_frames = 0
        min_silence_frames = int(min_silence_ms / 1000 * self._sr / frame_samples)
        in_speech = False

        q: asyncio.Queue[np.ndarray] = asyncio.Queue()
        loop = asyncio.get_running_loop()

        def _callback(indata, frames, t, status):
            loop.call_soon_threadsafe(q.put_nowait, indata[:, 0].copy())

        with sd.InputStream(
            samplerate=self._sr, channels=1, dtype="float32",
            blocksize=frame_samples, device=self._device,
            callback=_callback,
        ):
            while self._running:
                chunk = await q.get()

                # 半双工门控：TTS 播放期间/冷却期内丢弃音频，避免录到 ta 自己的声音
                if self._is_muted():
                    if speech_buf:
                        speech_buf.clear()
                    silence_frames = 0
                    in_speech = False
                    continue

                t = torch.tensor(chunk)
                with torch.no_grad():
                    prob = vad_model(t, self._sr).item()
                is_voice = prob > vad_threshold

                if is_voice:
                    in_speech = True
                    silence_frames = 0
                    speech_buf.append(chunk)
                elif in_speech:
                    speech_buf.append(chunk)  # 把尾部静音也留着
                    silence_frames += 1
                    if silence_frames >= min_silence_frames:
                        # 说完一句话：送 STT
                        audio = np.concatenate(speech_buf)
                        speech_buf.clear()
                        silence_frames = 0
                        in_speech = False
                        await self._process(audio, lang, stt_engine)

    async def _process(self, audio, lang: str, stt_engine) -> None:
        # 送 STT 前再确认没在播放（这段音频可能含 ta 自己的尾音）
        if self._is_muted():
            return
        text = await asyncio.to_thread(stt_engine.transcribe, audio, lang)
        text = (text or "").strip()
        if not text:
            return
        print(f"\n  [STT] {text}")
        cls = await self._classifier.classify(text, self._chat_history)
        if cls == AudioClassification.ASSISTANT:
            self._classifier.open_window()
            await self._enqueue(user_speech(text))
        elif cls in (AudioClassification.OTHER, AudioClassification.SELF):
            await self._enqueue(Event(
                priority=EventPriority.NORMAL,
                type=EventType.EAVESDROP,
                payload={"text": text, "classification": cls.value},
            ))

    def stop(self) -> None:
        self._running = False
