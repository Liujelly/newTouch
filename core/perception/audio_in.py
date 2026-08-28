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
import re
import time
from datetime import datetime
from enum import Enum

from ..config import Config
from ..events import Event, EventPriority, EventType, user_speech
from ..logger import get_logger

log = get_logger("audio")


_SENSEVOICE_TAG = re.compile(r"<\|([^|>]+)\|>")
_NON_SPEECH_TAGS = {"bgm", "music", "noise"}
_SHORT_ACKS = {
    "嗯", "恩", "唔", "啊", "哦", "噢", "诶", "额", "呃", "对", "是", "好", "好的", "行",
    "yes", "yeah", "yep", "ok", "okay", "uh", "um", "hmm", "mhm", "no", "nah",
    "うん", "はい", "ええ", "あ", "そう",
}
def _spoken_text(text: str) -> str:
    """移除 SenseVoice 标签，得到用于本地判断的可见文本。"""
    return _SENSEVOICE_TAG.sub("", text or "").strip()


def _has_non_speech_tag(text: str) -> bool:
    """SenseVoice 明确标成 BGM/Music/Noise 的片段不进入认知链路。"""
    tags = {tag.strip().lower() for tag in _SENSEVOICE_TAG.findall(text or "")}
    return bool(tags & _NON_SPEECH_TAGS)


def _is_short_ack(text: str) -> bool:
    """识别缺少独立语义的短回应，避免对话窗口把背景音直接当用户回复。"""
    spoken = _spoken_text(text).strip(" \t\r\n，。！？!?、…~～,.\"'“”‘’")
    return spoken.casefold() in _SHORT_ACKS


def _relative_time(ts: str, now: datetime) -> str:
    try:
        dt = datetime.fromisoformat(ts)
    except (TypeError, ValueError):
        return ""
    seconds = max(0, (now - dt).total_seconds())
    if seconds < 120:
        return "刚刚"
    if seconds < 3600:
        return f"{int(seconds // 60)}分钟前"
    if seconds < 86400:
        return f"{int(seconds // 3600)}小时前"
    return f"{int(seconds // 86400)}天前"


def _recent_context(chat_history: list[dict], limit: int = 4) -> str:
    """给音频分类器的历史附上时间，避免把数小时前的话当作当前上下文。"""
    now = datetime.now()
    lines = []
    for message in chat_history[-limit:]:
        relative = _relative_time(message.get("ts", ""), now)
        prefix = f"[{relative}] " if relative else ""
        lines.append(f"{prefix}{message.get('role', '')}: {message.get('content', '')}")
    return "\n".join(lines)


def _recent_assistant_supports_short_reply(chat_history: list[dict]) -> bool:
    """近期确有角色发言时，允许“嗯/好/yeah”作为正常简短承接。"""
    now = datetime.now()
    for message in reversed(chat_history):
        role = message.get("role")
        if role not in ("user", "assistant"):
            continue
        if role != "assistant":
            return False
        ts = message.get("ts", "")
        if not ts:
            return False
        try:
            age = (now - datetime.fromisoformat(ts)).total_seconds()
        except (TypeError, ValueError):
            return False
        if age < 0 or age > 30:
            return False
        content = _spoken_text(message.get("content", "")).rstrip()
        return bool(content)
    return False


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
    SUSPICIOUS = "suspicious"


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
        self._last_short_ack = ""
        self._last_short_ack_at = 0.0
        self._short_ack_repeat_window = config.get(
            "perception.audio.suspicious_repeat_window_s", 20
        )

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
        if _has_non_speech_tag(text):
            return AudioClassification.IGNORE

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

        if _is_short_ack(text):
            now = time.time()
            spoken = _spoken_text(text).casefold()
            repeated = (
                spoken == self._last_short_ack
                and now - self._last_short_ack_at <= self._short_ack_repeat_window
            )
            self._last_short_ack = spoken
            self._last_short_ack_at = now
            if repeated:
                return AudioClassification.SUSPICIOUS
            if now < self._window_until and _recent_assistant_supports_short_reply(chat_history):
                return AudioClassification.ASSISTANT
            return AudioClassification.SUSPICIOUS

        if time.time() < self._window_until:
            return AudioClassification.ASSISTANT
        return await self._llm_classify(text, chat_history)

    async def _llm_classify(self, text: str, chat_history: list[dict]) -> AudioClassification:
        recent = _recent_context(chat_history)
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
        except Exception as e:  # noqa: BLE001
            log.warning("音频对象分类失败，忽略本段避免误打扰: %s", e)
            return AudioClassification.IGNORE

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
        self._suspicious_repeat_window = config.get(
            "perception.audio.suspicious_repeat_window_s", 20
        )
        self._pending_suspicious_text = ""
        self._pending_suspicious_at = 0.0
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
            self._pending_suspicious_text = ""
            self._pending_suspicious_at = 0.0
            self._classifier.open_window()
            await self._enqueue(user_speech(text, source="microphone"))
        elif cls == AudioClassification.SUSPICIOUS:
            now = time.time()
            spoken = _spoken_text(text).casefold()
            repeated = (
                spoken == self._pending_suspicious_text
                and now - self._pending_suspicious_at <= self._suspicious_repeat_window
            )
            if repeated:
                self._pending_suspicious_text = ""
                self._pending_suspicious_at = 0.0
                log.info("短音频连续出现，进入待确认反应: %s", text)
                await self._enqueue(user_speech(
                    text, source="microphone", uncertain_audio=True,
                ))
            else:
                self._pending_suspicious_text = spoken
                self._pending_suspicious_at = now
                log.info("短音频含义不明，暂不打扰并等待是否重复: %s", text)
        elif cls in (AudioClassification.OTHER, AudioClassification.SELF):
            await self._enqueue(Event(
                priority=EventPriority.NORMAL,
                type=EventType.EAVESDROP,
                payload={"text": text, "classification": cls.value},
            ))

    def stop(self) -> None:
        self._running = False
