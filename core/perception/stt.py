"""STT 引擎抽象层 (架构文档 5.6 / ADR-015)。

把语音转文字抽象成可切换的 provider，管理平台选 modules.stt.provider 即可换引擎：
  - faster-whisper：默认，CPU 1-2s/句，多语言通用，pip 装上首次自动下权重。
  - funasr (SenseVoiceSmall)：中文专优、CPU 更快，小智(xiaozhi)同款。首次从
    ModelScope/HF 下模型。

统一接口：engine.transcribe(audio_np, lang) -> str
  audio_np: float32、单声道、16kHz 的 numpy 一维数组（MicInput 的 VAD 缓冲产出）。
  lang:     语言码（zh/en/ja/ko/yue/auto），各引擎内部做映射。
  返回:     纯文本（已去除 SenseVoice 的情绪/事件标签）；无内容返回空串。

引擎在构造时加载模型（重，故 MicInput 启动时构造一次复用）；transcribe 由调用方
用 asyncio.to_thread 包到线程里跑，避免阻塞事件循环。
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Protocol

from ..logger import get_logger

log = get_logger("stt")

from ..config import Config


def _resolve_cache_dir(config: Config) -> str | None:
    """模型缓存目录：modules.stt.model_cache_dir，相对路径按项目根解析。

    留空则用各框架默认（~/.cache/...）。返回绝对路径字符串或 None。
    """
    raw = config.get("modules.stt.model_cache_dir", "") or ""
    if not raw:
        return None
    p = Path(raw)
    if not p.is_absolute():
        p = config.project_root / raw
    p.mkdir(parents=True, exist_ok=True)
    return str(p)


class STTEngine(Protocol):
    """STT 引擎统一接口。transcribe 是同步的（CPU 密集），调用方负责 to_thread。"""

    def transcribe(self, audio, lang: str) -> str: ...


class WhisperSTT:
    """faster-whisper 引擎。多语言、成熟、CPU int8 下 1-2s/句。"""

    def __init__(self, config: Config):
        # 必须在 import faster_whisper / huggingface_hub 之前设镜像端点
        hf_endpoint = config.get("modules.stt.hf_endpoint", "")
        if hf_endpoint:
            os.environ["HF_ENDPOINT"] = hf_endpoint
        from faster_whisper import WhisperModel

        download_root = config.get("modules.stt.download_root", "") or _resolve_cache_dir(config)
        log.info("加载 faster-whisper（首次需从 %s 下载，请稍候）…",
                 hf_endpoint or "HuggingFace官方")
        self._model = WhisperModel(
            config.get("modules.stt.model", "small"),
            device=config.get("modules.stt.device", "cpu"),
            compute_type=config.get("modules.stt.compute_type", "int8"),
            download_root=download_root,
        )
        self._beam_size = config.get("modules.stt.beam_size", 1)

    def transcribe(self, audio, lang: str) -> str:
        # faster-whisper 的 language 用 None 表示自动检测
        whisper_lang = None if lang in ("auto", "", None) else lang
        segments, _ = self._model.transcribe(
            audio, language=whisper_lang, beam_size=self._beam_size
        )
        return " ".join(s.text.strip() for s in segments).strip()


class FunASRSTT:
    """FunASR SenseVoiceSmall 引擎。中文专优、CPU 比 whisper 快，小智同款。

    SenseVoice 输出带情绪/事件标签（如 <|HAPPY|><|Speech|>），用官方
    rich_transcription_postprocess 清洗成纯文本。
    """

    # newTouch 语言码 → SenseVoice 语言码（SenseVoice 用 zn 而非 zh）
    _LANG_MAP = {"zh": "zn", "en": "en", "ja": "ja", "ko": "ko",
                 "yue": "yue", "auto": "auto", "": "auto"}

    def __init__(self, config: Config):
        hf_endpoint = config.get("modules.stt.hf_endpoint", "")
        if hf_endpoint:
            os.environ["HF_ENDPOINT"] = hf_endpoint
        # 模型缓存指向项目内目录（不依赖用户主目录）：ModelScope 认 MODELSCOPE_CACHE，
        # 它会在该目录下建 models/{model_id}/（新版结构）。留空则用 ModelScope 默认 ~/.cache/modelscope。
        cache_dir = _resolve_cache_dir(config)
        if cache_dir:
            os.environ["MODELSCOPE_CACHE"] = cache_dir
        from funasr import AutoModel
        from funasr.utils.postprocess_utils import rich_transcription_postprocess

        self._postprocess = rich_transcription_postprocess
        model_dir = config.get("modules.stt.funasr_model", "iic/SenseVoiceSmall")
        # 若传的是项目内相对路径且该目录存在，按本地模型目录解析（绝对路径/model_id 原样传）
        cand = config.project_root / model_dir
        if cand.exists():
            model_dir = str(cand)
        device = config.get("modules.stt.device", "cpu")
        self._use_itn = config.get("modules.stt.use_itn", True)
        # 保留 SenseVoice 原始情绪/事件标签（<|HAPPY|><|Speech|>…）让主脑感知用户情绪。
        # 默认保留；关掉则用官方后处理洗成纯文本（旧行为）。
        self._keep_emotion_tags = config.get("modules.stt.keep_emotion_tags", True)
        log.info("加载 FunASR %s（缓存目录 %s，首次需下载，请稍候）…",
                 model_dir, cache_dir or "ModelScope 默认")
        self._model = AutoModel(
            model=model_dir,
            trust_remote_code=False,
            device="cuda:0" if device == "cuda" else "cpu",
            disable_update=True,
        )

    def transcribe(self, audio, lang: str) -> str:
        sv_lang = self._LANG_MAP.get(lang, "auto")
        res = self._model.generate(
            input=audio,
            cache={},
            language=sv_lang,
            use_itn=self._use_itn,
            batch_size_s=60,
        )
        if not res:
            return ""
        raw = res[0]["text"]
        # 保留原始标签（含情绪/事件，供主脑判断用户情绪）或洗成纯文本
        if self._keep_emotion_tags:
            return raw.strip()
        return self._postprocess(raw).strip()


def load_stt_engine(config: Config) -> STTEngine | None:
    """按 modules.stt.provider 构造对应引擎。依赖缺失/加载失败返回 None（麦克风降级不可用）。"""
    provider = (config.get("modules.stt.provider", "faster-whisper") or "").lower()
    try:
        if provider == "funasr":
            return FunASRSTT(config)
        return WhisperSTT(config)
    except ImportError as e:
        pkg = "funasr" if provider == "funasr" else "faster-whisper"
        log.warning("%s 未安装，STT 不可用：%s", pkg, e)
        return None
    except Exception as e:  # noqa: BLE001
        log.error("STT 引擎加载失败：%s", e)
        return None
