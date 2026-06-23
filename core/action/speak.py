"""语音输出 (架构文档 5.8)。

流式分句 -> GPT-SoVITS POST /tts -> sounddevice 播放，支持 barge-in 打断。
TTS 关闭 (modules.tts.enabled=false) 时退化为纯文本打印。

关键设计：_tts_request 每次现读 config（不缓存），管理平台改配置后下句立即生效。
text_split_method 固定 cut0：newTouch 的 _split_sentences 已按标点切好句子逐句发给
TTS，若再用 cut5 会二次切碎导致语音不连贯。
"""
from __future__ import annotations

import asyncio
import io
import json
import re
from pathlib import Path
from typing import AsyncIterator

import httpx

from ..config import Config
from ..logger import get_logger

log = get_logger("speak")

_SENTENCE_END_CHARS = frozenset("。！？!?\n…")
_OPEN_BRACKETS = frozenset("（(")
_CLOSE_BRACKETS = frozenset("）)")
# 匹配成对括号内的动作描述/翻译，如 （默默看着你） 或 (sighs) 或 （中文翻译）
_ACTION_BRACKET = re.compile(r"[（(][^）)]*[）)]")
# 流式截断 / 切句残留的未闭合括号片段兜底
_DANGLING_OPEN = re.compile(r"[（(][^）)]*$")
_DANGLING_CLOSE = re.compile(r"^[^（(]*[）)]")
# 回复开头的情绪标签，如 <emo:happy>。只控制 TTS 语气，不朗读、不入聊天记录。
_EMO_PREFIX = re.compile(r"^\s*<\s*emo\s*:\s*([A-Za-z_]+)\s*>", re.IGNORECASE)
# 文本中任意位置的情绪标签（用于清理 LLM 可能在末尾/中间误输出的标签）
_EMO_TAG_ANYWHERE = re.compile(r"<\s*emo\s*:\s*[A-Za-z_]+\s*>", re.IGNORECASE)


def parse_emotion_prefix(text: str) -> tuple[str | None, str]:
    """抠出回复开头的 <emo:情绪> 标签，返回 (情绪|None, 去标签后的文本)。"""
    m = _EMO_PREFIX.match(text or "")
    if not m:
        return None, text
    return m.group(1).lower(), text[m.end():]


def strip_all_emotion_tags(text: str) -> str:
    """移除文本中所有位置的 <emo:xxx> 标签（开头/中间/末尾）。

    用于清理 LLM 可能在非开头位置误输出的标签，确保标签不进 TTS/聊天记录/记忆。
    """
    return _EMO_TAG_ANYWHERE.sub("", text)


def _looks_like_partial_emo(stripped: str) -> bool:
    """判断缓冲文本是否像未闭合的 <emo:…> 前缀（用于流式跨 chunk 缓冲）。"""
    return bool(re.match(r"^<\s*e?m?o?\s*:?\s*[A-Za-z_]*$", stripped, re.IGNORECASE))


_VOICE_LIB_RESERVED = frozenset(("gpt_weights", "sovits_weights"))


def _voice_lib_path(config: Config) -> Path:
    """按当前角色 card.json extensions.voice_lib 绑定找库文件；未绑定回退 library.json。"""
    name = config.get("character.name", "默认")
    voices_dir = config.project_root / "data" / "voices"
    card_path = config.project_root / "data" / "characters" / name / "card.json"
    lib = "library"
    try:
        ext = json.loads(card_path.read_text(encoding="utf-8")).get("extensions") or {}
        lib = ext.get("voice_lib") or "library"
    except (OSError, json.JSONDecodeError, ValueError):
        pass
    return voices_dir / f"{lib}.json"


def _parse_voice_lib_raw(config: Config) -> dict:
    """读语音库文件的原始 dict；不存在/解析失败返回 {}。"""
    p = _voice_lib_path(config)
    if not p.exists():
        return {}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (json.JSONDecodeError, ValueError, OSError):
        return {}


def load_voice_library(config: Config) -> dict:
    """读情绪语音库（每次现读，管理平台改完即生效）。

    新格式：{gpt_weights, sovits_weights, emotions: {情绪: {...}}}
    旧格式：{情绪: {...}} 平铺（向后兼容）
    返回纯情绪表 {情绪: {ref_audio_path, prompt_text, prompt_lang}}。
    """
    data = _parse_voice_lib_raw(config)
    if isinstance(data.get("emotions"), dict):
        return data["emotions"]
    # 旧格式平铺：剔除库级保留键
    return {k: v for k, v in data.items()
            if k not in _VOICE_LIB_RESERVED and isinstance(v, dict)}


def load_voice_meta(config: Config) -> dict:
    """读语音库级字段 {gpt_weights, sovits_weights}（均为空串时表示未配置）。"""
    data = _parse_voice_lib_raw(config)
    return {
        "gpt_weights":    data.get("gpt_weights", ""),
        "sovits_weights": data.get("sovits_weights", ""),
    }


async def apply_voice_model(config: Config) -> dict:
    """按当前角色绑定的语音库，向 GPT-SoVITS 应用 gpt_weights / sovits_weights。

    两个路径都为空则直接返回 {}；失败静默返回错误信息，不抛出。
    """
    meta = load_voice_meta(config)
    gpt = meta.get("gpt_weights", "")
    sovits = meta.get("sovits_weights", "")
    if not gpt and not sovits:
        return {}
    endpoint = config.get("modules.tts.endpoint", "http://127.0.0.1:9880")
    results = {}
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            if gpt:
                r = await client.get(f"{endpoint}/set_gpt_weights",
                                     params={"weights_path": gpt})
                results["gpt"] = "ok" if r.status_code == 200 else r.text
            if sovits:
                r = await client.get(f"{endpoint}/set_sovits_weights",
                                     params={"weights_path": sovits})
                results["sovits"] = "ok" if r.status_code == 200 else r.text
    except Exception as e:  # noqa: BLE001
        results["error"] = str(e)
    return results


def load_voice_emotions(config: Config) -> list[str]:
    """语音库里有哪些情绪档（供主脑 prompt 列出可选情绪，方案3 库驱动）。"""
    return list(load_voice_library(config).keys())



def _strip_actions(text: str) -> str:
    """去掉括号内的动作/表情/翻译，只保留要朗读的语言文字。

    先剔成对括号，再兜底清掉残缺括号片段（防止流式截断把半个括号送进 TTS）。
    """
    text = _ACTION_BRACKET.sub("", text)
    text = _DANGLING_OPEN.sub("", text)
    text = _DANGLING_CLOSE.sub("", text)
    return text.strip()


def _find_split(buf: str) -> int | None:
    """返回 buf 中处于括号外的句末标点串结束位置；括号内的标点不触发切句。

    这样翻译/动作括号内的 ？！。… 不会把句子切碎，保证切出的句子括号成对，
    _strip_actions 才能完整剔除括号内容（含非中文回复附带的中文翻译）。
    """
    depth = 0
    n = len(buf)
    for i, c in enumerate(buf):
        if c in _OPEN_BRACKETS:
            depth += 1
        elif c in _CLOSE_BRACKETS:
            if depth > 0:
                depth -= 1
        elif depth == 0 and c in _SENTENCE_END_CHARS:
            j = i + 1
            while j < n and buf[j] in _SENTENCE_END_CHARS:
                j += 1
            return j
    return None


async def _split_sentences(stream: AsyncIterator[str]) -> AsyncIterator[str]:
    buf = ""
    async for chunk in stream:
        buf += chunk
        while True:
            idx = _find_split(buf)
            if idx is None:
                break
            sentence = buf[:idx].strip()
            buf = buf[idx:]
            if sentence:
                yield sentence
    if buf.strip():
        yield buf.strip()


class Speaker:
    def __init__(self, config: Config, name: str = "ta"):
        self._cfg = config
        self._name = name
        self._interrupt = asyncio.Event()
        self._speaking = False
        self._cur_emotion: str | None = None  # 本轮回复的情绪（决定参考音频）

    # 读当前配置值的便捷方法（每次现读，不缓存）
    def _t(self, key: str, default=None):
        return self._cfg.get(f"modules.tts.{key}", default)

    def is_speaking(self) -> bool:
        return self._speaking

    def interrupt(self) -> None:
        self._interrupt.set()

    async def speak(self, text_stream: AsyncIterator[str], on_text=None,
                    emotion: str | None = None) -> str:
        """消费文本流并合成播放。

        producer/consumer 解耦：producer 尽快读完 LLM 流拿到全文，consumer 在后台
        逐句合成播放。文本一读完立即触发 on_text(full_text) 回调——这样聊天页能在
        语音还在播放时就显示完整回复，不必等播放结束。speak() 仍等播放完成才返回，
        保留 is_speaking()/barge-in/回声抑制语义。
        on_text 可能在播放尚未结束时被调用；TTS 关闭时退化为纯文本打印后即触发。

        emotion：主动路径直接传入（决定参考音频）；反应路径传 None，由流开头的
        <emo:情绪> 标签自动解析。标签被剥离，不进 TTS、不进 on_text 文本。
        """
        self._interrupt.clear()
        self._speaking = True
        self._cur_emotion = emotion  # 主动路径显式传；反应路径下面从标签解析覆盖
        # 反应路径：包一层流，剥掉开头 <emo:xxx> 标签并设 _cur_emotion
        if emotion is None:
            text_stream = self._strip_emotion_stream(text_stream)
        full: list[str] = []
        tts_on = self._t("enabled", True)
        fired = False

        def _fire() -> None:
            nonlocal fired
            if not fired and on_text:
                fired = True
                try:
                    # 清理所有残留的 <emo:xxx> 标签后再传给回调
                    cleaned = strip_all_emotion_tags("".join(full))
                    on_text(cleaned)
                except Exception as e:  # noqa: BLE001
                    log.warning("on_text 回调异常: %s", e)

        try:
            if not tts_on:
                # 纯文本：边读边打印，读完即触发回调
                printed_header = False
                async for sentence in _split_sentences(text_stream):
                    if self._interrupt.is_set():
                        break
                    full.append(sentence)
                    if not printed_header:
                        print(f"\n{self._name} > ", end="", flush=True)
                        printed_header = True
                    print(sentence, end=" ", flush=True)
                if printed_header:
                    print()
                _fire()
                return strip_all_emotion_tags("".join(full))

            # TTS：producer 读流入队 + 攒全文，consumer 后台播放
            queue: asyncio.Queue = asyncio.Queue()

            async def _producer() -> None:
                try:
                    async for sentence in _split_sentences(text_stream):
                        if self._interrupt.is_set():
                            break
                        full.append(sentence)
                        await queue.put(sentence)
                finally:
                    await queue.put(None)  # 结束哨兵
                    _fire()                # 全文已读完，立即通知（播放可能还在进行）

            async def _consumer() -> None:
                while True:
                    sentence = await queue.get()
                    if sentence is None:
                        break
                    if self._interrupt.is_set():
                        continue  # 排空队列直到哨兵，不再播放
                    await self._synth_and_play(sentence)

            await asyncio.gather(_producer(), _consumer())
        finally:
            _fire()  # 兜底：异常路径也确保回调触发一次
            self._speaking = False
        # 最终清理：移除全文中所有残留的 <emo:xxx> 标签（LLM 可能在末尾/中间误输出）
        return strip_all_emotion_tags("".join(full))

    async def _strip_emotion_stream(self, stream: AsyncIterator[str]) -> AsyncIterator[str]:
        """从流开头解析 <emo:情绪> 标签：设 self._cur_emotion，剥掉标签后把余下文本透传。

        标签可能跨多个 chunk，故先缓冲到能判定为止（攒够 '>' 或确定开头不是标签）。
        """
        buf = ""
        decided = False
        async for chunk in stream:
            if decided:
                yield chunk
                continue
            buf += chunk
            # 还没攒到能判定的程度：开头是 '<emo:' 的前缀且未闭合，继续等
            stripped = buf.lstrip()
            if "<" in stripped and ">" not in stripped and len(buf) < 40:
                # 可能是未闭合的标签，继续缓冲（上限 40 字防止无限等）
                if _looks_like_partial_emo(stripped):
                    continue
            emo, rest = parse_emotion_prefix(buf)
            if emo is not None:
                self._cur_emotion = emo
            decided = True
            if rest:
                yield rest
        # 流结束仍未 decided（极短流）：兜底解析一次
        if not decided and buf:
            emo, rest = parse_emotion_prefix(buf)
            if emo is not None:
                self._cur_emotion = emo
            if rest:
                yield rest

    async def _synth_and_play(self, sentence: str) -> None:
        sentence = _strip_actions(sentence)
        if not sentence:
            return
        try:
            audio = await self._tts_request(sentence)
            if audio and not self._interrupt.is_set():
                await self._play(audio)
        except Exception as e:  # noqa: BLE001
            print(f"  💬 {sentence}   [TTS失败: {e}]")
            log.error("TTS 失败（%s）: %s", sentence, e)

    def _ref_for_emotion(self) -> tuple[str, str, str]:
        """按本轮情绪从语音库选 (ref_audio_path, prompt_text, prompt_lang)。

        支持多参考音频：每个情绪档可配置多个 refs，随机选取其中一个。
        新格式：{"refs": [{"ref_audio_path":..., "prompt_text":..., "prompt_lang":...}]}
        旧格式：{"ref_audio_path":..., "prompt_text":..., "prompt_lang":...}（自动兼容）

        命中情绪档用该档；未命中（空库/库外情绪/无标签）回退全局默认
        modules.tts.ref_audio_path/prompt_text/prompt_lang。
        """
        import random

        default = (self._t("ref_audio_path", ""), self._t("prompt_text", ""),
                   self._t("prompt_lang", "zh"))
        emo = self._cur_emotion
        if not emo:
            return default

        entry = load_voice_library(self._cfg).get(emo)
        if not isinstance(entry, dict):
            return default

        # 新格式：refs 数组（多个参考音频）
        if "refs" in entry and isinstance(entry["refs"], list) and entry["refs"]:
            ref = random.choice(entry["refs"])
            if isinstance(ref, dict) and ref.get("ref_audio_path"):
                return (
                    ref.get("ref_audio_path") or default[0],
                    ref.get("prompt_text") or default[1],
                    ref.get("prompt_lang") or default[2]
                )
            # refs 列表为空或格式错误，回退默认
            return default

        # 旧格式：单个参考音频（向后兼容）
        if entry.get("ref_audio_path"):
            return (
                entry.get("ref_audio_path") or default[0],
                entry.get("prompt_text") or default[1],
                entry.get("prompt_lang") or default[2]
            )

        return default

    async def _tts_request(self, text: str) -> bytes:
        """POST /tts，每次现读 config，管理平台改参数后下句立即生效。

        text_split_method=cut0：newTouch 已在外部按标点切句，TTS 端不需要再切。
        参考音频按本轮情绪从语音库选（见 _ref_for_emotion），未命中回退全局默认。
        """
        endpoint = self._t("endpoint", "http://127.0.0.1:9880")
        ref_audio, prompt_text, prompt_lang = self._ref_for_emotion()
        payload = {
            "text": text,
            "text_lang":        self._t("text_lang", "zh"),
            "ref_audio_path":   ref_audio,
            "prompt_text":      prompt_text,
            "prompt_lang":      prompt_lang,
            "speed_factor":     self._t("speed_factor", 1.0),
            # 切分方式：外部已切句，TTS 端直接合成整句
            "text_split_method": self._t("text_split_method", "cut0"),
            # 采样参数（config 未配则用 GPT-SoVITS 默认值）
            "top_k":            self._t("top_k", 5),
            "top_p":            self._t("top_p", 1.0),
            "temperature":      self._t("temperature", 1.0),
            "repetition_penalty": self._t("repetition_penalty", 1.35),
            "batch_size":       self._t("batch_size", 1),
            "fragment_interval": self._t("fragment_interval", 0.3),
            "parallel_infer":   self._t("parallel_infer", True),
            "streaming_mode":   False,
            "return_fragment":  False,
        }
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(endpoint + "/tts", json=payload)
            resp.raise_for_status()
            return resp.content

    async def _play(self, wav_bytes: bytes) -> None:
        import soundfile as sf
        import sounddevice as sd

        data, samplerate = sf.read(io.BytesIO(wav_bytes), dtype="float32")

        def _blocking():
            sd.play(data, samplerate)
            sd.wait()

        await asyncio.to_thread(_blocking)
