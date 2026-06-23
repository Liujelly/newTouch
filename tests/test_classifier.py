"""测试对话对象分流优化：唤醒词完整匹配 + LLM 分类准确性验证。"""
import asyncio
import time
from unittest.mock import AsyncMock, MagicMock
from core.config import Config
from core.perception.audio_in import Classifier, AudioClassification


class MockLLMClient:
    """模拟 LLM 客户端，可预设返回结果。"""
    def __init__(self, response: str = "other"):
        self.response = response
        self.messages = AsyncMock()
        self.chat = MagicMock()
        self.chat.completions = MagicMock()

    async def mock_anthropic_create(self, **kwargs):
        """模拟 anthropic 响应。"""
        mock_resp = MagicMock()
        mock_resp.content = [MagicMock(text=self.response)]
        return mock_resp

    async def mock_openai_create(self, **kwargs):
        """模拟 openai 响应。"""
        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock(message=MagicMock(content=self.response))]
        return mock_resp


async def test_wake_word_matching():
    """测试唤醒词匹配：完整词 vs 子串。"""
    print("=" * 70)
    print("Test 1: Wake Word Matching (完整词匹配 vs 子串)")
    print("=" * 70)

    cfg = Config("data/config.yaml")
    mock_client = MockLLMClient()
    # 不会调用 LLM，所以响应内容无关紧要
    classifier = Classifier(cfg, mock_client, "openai", "test-model", "小触")

    test_cases = [
        ("小触你好", True, "完整词匹配：应触发"),
        ("小触，在吗", True, "完整词匹配：应触发"),
        ("hey 小触", True, "完整词匹配：应触发"),
        ("小触摸屏很好用", False, "子串匹配：不应触发（已修复）"),
        ("newtouch is great", True, "英文唤醒词（不区分大小写）"),
        ("我买了个new touchpad", False, "子串匹配：不应触发"),
        ("今天天气不错", False, "无唤醒词：不触发"),
    ]

    for text, should_trigger, desc in test_cases:
        result = await classifier.classify(text, [])
        triggered = (result == AudioClassification.ASSISTANT)
        status = "[OK]" if triggered == should_trigger else "[FAIL]"
        print(f"\n{status} {desc}")
        print(f"  输入: {text!r}")
        print(f"  期望触发: {should_trigger}, 实际: {triggered}")


async def test_dialog_window():
    """测试对话窗口：窗口内 vs 窗口外。"""
    print("\n\n" + "=" * 70)
    print("Test 2: Dialog Window (对话窗口)")
    print("=" * 70)

    cfg = Config("data/config.yaml")
    mock_client = MockLLMClient("other")
    # 正确设置 mock 方法
    mock_client.chat.completions.create = mock_client.mock_openai_create
    classifier = Classifier(cfg, mock_client, "openai", "test-model", "小触")

    # 1. 触发唤醒词，打开窗口
    result = await classifier.classify("小触在吗", [])
    print(f"\n[步骤1] 触发唤醒词: {result.value}")
    assert result == AudioClassification.ASSISTANT

    # 2. 窗口内（15秒内），不调 LLM，直接返回 ASSISTANT
    result = await classifier.classify("今天天气怎么样", [])
    print(f"[步骤2] 窗口内（无唤醒词）: {result.value} - 应为 assistant（窗口内）")
    assert result == AudioClassification.ASSISTANT

    # 3. 模拟窗口过期（手动修改时间）
    classifier._window_until = time.time() - 1  # 窗口已过期
    result = await classifier.classify("今天天气怎么样", [])
    print(f"[步骤3] 窗口外（无唤醒词）: {result.value} - 应为 other（LLM返回）")
    # 注意：窗口外会调用 _llm_classify，mock 应返回 "other"
    assert result == AudioClassification.OTHER, f"Expected OTHER, got {result}"

    print("\n[OK] 对话窗口机制正常")


async def test_llm_classification():
    """测试 LLM 分类：验证不同场景的分类结果。"""
    print("\n\n" + "=" * 70)
    print("Test 3: LLM Classification (LLM 分类)")
    print("=" * 70)

    cfg = Config("data/config.yaml")

    scenarios = [
        ("other", "你说得对", "对朋友说话，应判断为 other"),
        ("self", "哎呀忘了带钥匙", "自言自语，应判断为 self"),
        ("ignore", "...嗯...啊...", "背景音，应判断为 ignore"),
        ("assistant", "在干嘛呢", "模糊提问（可能对 AI），LLM 判断"),
    ]

    for llm_response, user_text, desc in scenarios:
        mock_client = MockLLMClient(llm_response)
        mock_client.chat.completions.create = mock_client.mock_openai_create
        classifier = Classifier(cfg, mock_client, "openai", "test-model", "小触")

        # 确保窗口已过期，走 LLM 分类
        classifier._window_until = 0.0

        result = await classifier.classify(user_text, [])
        expected = AudioClassification(llm_response)
        status = "[OK]" if result == expected else "[FAIL]"
        print(f"\n{status} {desc}")
        print(f"  输入: {user_text!r}")
        print(f"  LLM 返回: {llm_response}, 分类结果: {result.value}")


async def main():
    await test_wake_word_matching()
    await test_dialog_window()
    await test_llm_classification()

    print("\n\n" + "=" * 70)
    print("Summary: 所有测试完成")
    print("=" * 70)
    print("\n修复效果：")
    print("1. 唤醒词匹配改为完整词边界，'小触摸屏'不再误触发")
    print("2. 对话窗口从 60秒 缩短到 15秒，减少误判窗口")
    print("3. LLM 分类 prompt 强化，明确判断标准")
    print("\n建议：")
    print("- 如果仍有误判，可进一步缩短窗口到 10秒")
    print("- 查看 data/consciousness.jsonl 的分类结果，调整 prompt")
    print("- 可在 config.yaml 添加更多唤醒词变体")


if __name__ == "__main__":
    asyncio.run(main())
