"""端到端测试：模拟 LLM 流式输出 → speak 处理 → on_text 回调，验证标签完全清理。"""
import asyncio
import tempfile
import yaml
from pathlib import Path
from core.action.speak import Speaker
from core.config import Config


async def mock_llm_stream(text: str):
    """模拟 LLM 流式输出，逐字符 yield。"""
    for char in text:
        yield char
        await asyncio.sleep(0.001)


async def test_e2e():
    """端到端测试：验证带标签的 LLM 输出经过 speak 后完全清理。"""
    # 创建临时配置文件，禁用 TTS
    temp_cfg = {
        "modules": {
            "tts": {
                "enabled": False,
                "provider": "gpt-sovits",
                "endpoint": "http://127.0.0.1:9880"
            }
        }
    }

    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False, encoding='utf-8') as f:
        yaml.dump(temp_cfg, f)
        temp_path = f.name

    try:
        cfg = Config(temp_path)
        speaker = Speaker(cfg, "TestBot")
        # 强制禁用 TTS（覆盖 _t 方法的默认行为）
        original_t = speaker._t
        def mock_t(key, default=None):
            if key == "enabled":
                return False
            return original_t(key, default)
        speaker._t = mock_t

        test_cases = [
            # (LLM输出, 期望情绪, 期望的最终文本)
            ("<emo:happy>你好呀！", "happy", "你好呀！"),
            ("<emo:happy>你好呀！<emo:happy>", "happy", "你好呀！"),  # 末尾重复标签
            ("<emo:gentle>我在这里等你<emo:affection>", "gentle", "我在这里等你"),  # 末尾不同标签
            ("回复没有标签", None, "回复没有标签"),  # 无标签
            ("<emo:neutral>前面<emo:happy>中间<emo:sad>末尾", "neutral", "前面中间末尾"),  # 多处标签
        ]

        print("=" * 70)
        print("End-to-End Test: LLM Stream -> speak -> on_text callback")
        print("=" * 70)

        for llm_output, exp_emotion, exp_clean in test_cases:
            print(f"\n[Test] LLM output: {llm_output!r}")

            # 收集 on_text 回调结果
            callback_result = {"text": None}

            def on_text(text: str):
                callback_result["text"] = text

            # 模拟 speak 消费流（emotion=None 触发标签解析）
            final_text = await speaker.speak(mock_llm_stream(llm_output), on_text=on_text)

            # 验证
            callback_ok = callback_result["text"] == exp_clean
            return_ok = final_text == exp_clean
            emotion_ok = speaker._cur_emotion == exp_emotion

            print(f"  Expected emotion: {exp_emotion}, actual: {speaker._cur_emotion} "
                  f"{'[OK]' if emotion_ok else '[FAIL]'}")
            print(f"  Expected clean text: {exp_clean!r}")
            print(f"  on_text callback got: {callback_result['text']!r} "
                  f"{'[OK]' if callback_ok else '[FAIL]'}")
            print(f"  speak() returned: {final_text!r} "
                  f"{'[OK]' if return_ok else '[FAIL]'}")

            if callback_ok and return_ok and emotion_ok:
                print("  => [PASS] All checks passed!")
            else:
                print("  => [FAIL] Some checks failed!")

    finally:
        # 清理临时文件
        Path(temp_path).unlink(missing_ok=True)


if __name__ == "__main__":
    asyncio.run(test_e2e())
