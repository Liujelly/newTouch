"""首句 TTS 就绪后再同步释放立绘和气泡。"""

from __future__ import annotations

import asyncio

from core.action.speak import Speaker


class DictConfig:
    def get(self, path, default=None):
        if path == "modules.tts.enabled":
            return True
        return default


class RecordingBroadcaster:
    def __init__(self, events):
        self.events = events

    async def push(self, face, character):
        self.events.append(("face", face))

    async def push_text(self, text, character):
        self.events.append(("text", text))

    async def push_text_end(self, character):
        self.events.append(("end", ""))


async def stream(text):
    yield text


async def test_first_audio_ready_releases_visuals_before_play():
    events = []
    speaker = Speaker(DictConfig(), "小触")
    speaker.set_emotion_broadcaster(RecordingBroadcaster(events))

    async def fake_tts(text):
        assert not any(kind in ("face", "text") for kind, _ in events)
        events.append(("tts_ready", text))
        return b"wav"

    async def fake_play(audio):
        events.append(("play", ""))

    speaker._tts_request = fake_tts
    speaker._play = fake_play

    await speaker.speak(stream("欢迎回来！"), emotion="happy", face="开心")
    await asyncio.sleep(0)

    kinds = [kind for kind, _ in events]
    assert kinds.index("tts_ready") < kinds.index("face")
    assert kinds.index("face") < kinds.index("text")
    assert kinds.index("text") < kinds.index("play")


async def test_tts_failure_still_releases_visuals():
    events = []
    speaker = Speaker(DictConfig(), "小触")
    speaker.set_emotion_broadcaster(RecordingBroadcaster(events))

    async def failed_tts(text):
        events.append(("tts_failed", text))
        raise RuntimeError("测试失败")

    speaker._tts_request = failed_tts
    speaker._play = lambda audio: None

    await speaker.speak(stream("听得到吗？"), emotion="neutral", face="思考")
    await asyncio.sleep(0)

    kinds = [kind for kind, _ in events]
    assert "face" in kinds and "text" in kinds
    assert kinds.index("tts_failed") < kinds.index("face")


async def test_only_first_sentence_waits_for_visual_start():
    events = []
    speaker = Speaker(DictConfig(), "小触")
    speaker.set_emotion_broadcaster(RecordingBroadcaster(events))

    async def fake_tts(text):
        events.append(("tts_ready", text))
        return text.encode("utf-8")

    async def fake_play(audio):
        events.append(("play", audio.decode("utf-8")))

    speaker._tts_request = fake_tts
    speaker._play = fake_play

    await speaker.speak(stream("第一句。第二句。"), emotion="happy", face="开心")

    assert [value for kind, value in events if kind == "tts_ready"] == ["第一句。", "第二句。"]
    assert [value for kind, value in events if kind == "play"] == ["第一句。", "第二句。"]
    assert sum(1 for kind, _ in events if kind == "face") == 1


async def main():
    await test_first_audio_ready_releases_visuals_before_play()
    await test_tts_failure_still_releases_visuals()
    await test_only_first_sentence_waits_for_visual_start()
    print("test_tts_visual_sync: all passed")


if __name__ == "__main__":
    asyncio.run(main())
