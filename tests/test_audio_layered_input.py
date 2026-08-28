"""麦克风误收录的分层处理测试。"""
from __future__ import annotations

import asyncio
import tempfile
import time
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import numpy as np

from core.events import EventType
from core.orchestrator import Orchestrator
from core.perception.audio_in import AudioClassification, Classifier, MicInput


class DictConfig:
    def __init__(self, values=None):
        self.values = values or {}
        self.project_root = Path.cwd()

    def get(self, path, default=None):
        current = self.values
        for part in path.split("."):
            if not isinstance(current, dict) or part not in current:
                return default
            current = current[part]
        return current


class StaticSTT:
    def __init__(self, text):
        self.text = text

    def transcribe(self, audio, lang):
        return self.text


def _client(response="other"):
    client = MagicMock()
    result = MagicMock()
    result.choices = [MagicMock(message=MagicMock(content=response))]
    client.chat.completions.create = AsyncMock(return_value=result)
    return client


async def test_classifier_layers():
    config = DictConfig({"perception": {"audio": {"dialog_window_seconds": 15}}})
    classifier = Classifier(config, _client(), "openai", "test", "爱丽丝")
    classifier.open_window()

    assert await classifier.classify("<|BGM|>Yeah", []) == AudioClassification.IGNORE
    assert await classifier.classify("Yeah", []) == AudioClassification.SUSPICIOUS

    recent_history = [{
        "role": "assistant", "content": "今天辛苦啦。",
        "ts": datetime.now().isoformat(timespec="seconds"),
    }]
    assert await classifier.classify("嗯", recent_history) == AudioClassification.ASSISTANT
    assert await classifier.classify("嗯", recent_history) == AudioClassification.SUSPICIOUS

    old_history = [{
        "role": "assistant", "content": "等你回来。",
        "ts": (datetime.now() - timedelta(hours=2)).isoformat(timespec="seconds"),
    }]
    classifier._window_until = 0
    await classifier.classify("今天天气不错", old_history)
    prompt = _client_prompt(classifier._client)
    assert "[2小时前] assistant: 等你回来。" in prompt


def _client_prompt(client):
    return client.chat.completions.create.await_args.kwargs["messages"][0]["content"]


async def test_mic_repeats_then_marks_uncertain():
    enqueued = []

    async def enqueue(event):
        enqueued.append(event)

    config = DictConfig({
        "perception": {"audio": {
            "dialog_window_seconds": 15,
            "suspicious_repeat_window_s": 20,
        }}
    })
    classifier = Classifier(config, _client(), "openai", "test", "爱丽丝")
    classifier.open_window()
    mic = MicInput(config, enqueue, classifier)
    stt = StaticSTT("Yeah")
    audio = np.zeros(512, dtype=np.float32)

    await mic._process(audio, "zh", stt)
    assert enqueued == []

    await mic._process(audio, "zh", stt)
    assert len(enqueued) == 1
    assert enqueued[0].type == EventType.USER_SPEECH
    assert enqueued[0].payload["source"] == "microphone"
    assert enqueued[0].payload["uncertain_audio"] is True


async def test_uncertain_path_has_no_interaction_side_effects():
    temp_dir = tempfile.TemporaryDirectory()
    root = Path(temp_dir.name)
    reply = "刚才是在和我说话吗？"

    orchestrator = object.__new__(Orchestrator)
    orchestrator._cfg = DictConfig({
        "perception": {"audio": {"reactive_repeat_threshold": 0.6}},
        "modules": {"tts": {"text_lang": "zh"}},
        "character": {"default_translation_lang": ""},
        "memory": {"compress_enabled": False},
    })
    orchestrator._card = SimpleNamespace(extensions={})
    orchestrator._user_name = "老师"
    orchestrator._chat_history = [{
        "role": "assistant", "content": "我会等你的，路上小心。",
        "ts": datetime.now().isoformat(timespec="seconds"),
    }]
    orchestrator._earlier_summary = ""
    orchestrator._max_history = 40
    orchestrator._compress_batch = 10
    orchestrator._cognition_lock = asyncio.Lock()
    orchestrator._cognition = SimpleNamespace(
        respond_to_uncertain_audio=AsyncMock(return_value=reply)
    )
    orchestrator._speaker = SimpleNamespace(
        is_speaking=lambda: False,
        interrupt=MagicMock(),
        speak=AsyncMock(return_value=reply),
    )
    orchestrator._state = SimpleNamespace(
        snapshot=lambda: {},
        on_interaction=MagicMock(),
        save=MagicMock(),
    )
    orchestrator._gate = SimpleNamespace(record_interaction=MagicMock())
    orchestrator._memory = SimpleNamespace(add=MagicMock())
    orchestrator._log = SimpleNamespace(record=MagicMock())
    orchestrator._chat_log_path = root / "chat_history.jsonl"
    orchestrator._log_chat = MagicMock()
    orchestrator._face_emotions = lambda: []
    orchestrator._awaiting_reply = True

    await orchestrator._handle_uncertain_audio("Yeah")

    orchestrator._state.on_interaction.assert_not_called()
    orchestrator._gate.record_interaction.assert_not_called()
    orchestrator._memory.add.assert_not_called()
    assert orchestrator._awaiting_reply is True
    assert "可能误收录" in orchestrator._chat_history[-2]["content"]
    assert orchestrator._chat_history[-1]["content"] == reply
    temp_dir.cleanup()


async def test_uncertain_reply_rewrites_recent_repeat():
    temp_dir = tempfile.TemporaryDirectory()
    root = Path(temp_dir.name)
    old_reply = "我会等你的，路上小心。"
    new_reply = "老师，刚才是在和我说话吗？"

    orchestrator = object.__new__(Orchestrator)
    orchestrator._cfg = DictConfig({
        "perception": {"audio": {"reactive_repeat_threshold": 0.6}},
        "modules": {"tts": {"text_lang": "zh"}},
        "character": {"default_translation_lang": ""},
        "memory": {"compress_enabled": False},
    })
    orchestrator._card = SimpleNamespace(extensions={})
    orchestrator._user_name = "老师"
    orchestrator._chat_history = [{
        "role": "assistant", "content": old_reply,
        "ts": datetime.now().isoformat(timespec="seconds"),
    }]
    orchestrator._earlier_summary = ""
    orchestrator._max_history = 40
    orchestrator._compress_batch = 10
    orchestrator._cognition_lock = asyncio.Lock()
    responder = AsyncMock(side_effect=[old_reply, new_reply])
    orchestrator._cognition = SimpleNamespace(respond_to_uncertain_audio=responder)
    orchestrator._speaker = SimpleNamespace(
        is_speaking=lambda: False,
        interrupt=MagicMock(),
        speak=AsyncMock(return_value=new_reply),
    )
    orchestrator._state = SimpleNamespace(snapshot=lambda: {})
    orchestrator._log = SimpleNamespace(record=MagicMock())
    orchestrator._chat_log_path = root / "chat_history.jsonl"
    orchestrator._log_chat = MagicMock()
    orchestrator._face_emotions = lambda: []

    await orchestrator._handle_uncertain_audio("Yeah")

    assert responder.await_count == 2
    assert orchestrator._chat_history[-1]["content"] == new_reply
    temp_dir.cleanup()


async def main():
    await test_classifier_layers()
    await test_mic_repeats_then_marks_uncertain()
    await test_uncertain_path_has_no_interaction_side_effects()
    await test_uncertain_reply_rewrites_recent_repeat()
    print("test_audio_layered_input: all passed")


if __name__ == "__main__":
    asyncio.run(main())
