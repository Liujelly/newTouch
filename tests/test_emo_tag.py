"""测试情绪标签解析：验证回复末尾的 <emo:xxx> 是否被正确剥离。"""
import asyncio
import re
from core.action.speak import parse_emotion_prefix, strip_all_emotion_tags, _EMO_PREFIX


async def mock_llm_stream(text: str):
    """模拟LLM流式输出，逐字符yield。"""
    for char in text:
        yield char
        await asyncio.sleep(0.001)


async def test_strip_emotion():
    """测试 parse_emotion_prefix 和流式剥离逻辑。"""

    test_cases = [
        # (输入文本, 期望的情绪, 期望的剥离后文本)
        ("<emo:happy>你好呀！", "happy", "你好呀！"),
        ("  <emo:neutral>  嗯，我在这里。", "neutral", "嗯，我在这里。"),
        ("普通回复没有标签", None, "普通回复没有标签"),
        ("<emo:happy>回复开头有标签<emo:sad>", "happy", "回复开头有标签<emo:sad>"),  # 开头被剥，末尾不剥
        ("回复末尾有标签<emo:happy>", None, "回复末尾有标签<emo:happy>"),  # ❌ 末尾标签不会被剥离
        ("<emo:excited>前面有<emo:gentle>中间也有<emo:shy>", "excited", "前面有<emo:gentle>中间也有<emo:shy>"),
    ]

    print("=" * 60)
    print("测试 parse_emotion_prefix (仅解析开头)")
    print("=" * 60)

    for inp, exp_emo, exp_text in test_cases:
        emo, text = parse_emotion_prefix(inp)
        status = "[OK]" if (emo == exp_emo and text == exp_text) else "[FAIL]"
        print(f"\n{status} Input: {inp!r}")
        print(f"  Expected: emo={exp_emo}, text={exp_text!r}")
        print(f"  Actual:   emo={emo}, text={text!r}")


    print("\n\n" + "=" * 60)
    print("模拟问题场景：LLM在回复末尾输出 <emo:xxx>")
    print("=" * 60)

    # 模拟主脑可能输出的末尾带标签回复
    problem_replies = [
        "<emo:happy>你好呀，今天过得怎么样？<emo:happy>",  # 开头+末尾都有
        "<emo:gentle>我一直在这里等你哦。<emo:affection>",  # 开头有，末尾是另一个情绪
        "嗯，我明白了。<emo:neutral>",  # 只有末尾
    ]

    for reply in problem_replies:
        emo, text = parse_emotion_prefix(reply)
        has_tail_tag = bool(re.search(r'<\s*emo\s*:\s*\w+\s*>\s*$', text))
        print(f"\nOriginal: {reply!r}")
        print(f"  Parsed emotion: {emo}")
        print(f"  After parse_emotion_prefix: {text!r}")
        print(f"  {'[BUG] Tail tag remains!' if has_tail_tag else '[OK] No tail tag'}")

        # 测试新增的 strip_all_emotion_tags
        cleaned = strip_all_emotion_tags(text)
        has_any_tag = bool(re.search(r'<\s*emo\s*:\s*\w+\s*>', cleaned))
        print(f"  After strip_all_emotion_tags: {cleaned!r}")
        print(f"  {'[BUG] Tags still present!' if has_any_tag else '[FIXED] All tags removed!'}")


if __name__ == "__main__":
    asyncio.run(test_strip_emotion())
