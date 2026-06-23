"""测试唤醒词角色卡配置：验证管理平台配置 + 后端读取。"""
import json
from pathlib import Path
from core.config import load_config
from core.perception.audio_in import Classifier
from unittest.mock import MagicMock


def test_card_custom_wake_words():
    """测试：角色卡自定义唤醒词优先级最高。"""
    print("=" * 70)
    print("Test: Custom wake words from character card")
    print("=" * 70)

    cfg = load_config()
    mock_client = MagicMock()

    # 创建测试角色卡
    test_char_dir = Path("D:/code/self/newTouch/data/characters/测试角色")
    test_char_dir.mkdir(parents=True, exist_ok=True)

    test_card = {
        "name": "测试角色",
        "extensions": {
            "wake_words": ["小测", "测试", "test"]
        }
    }
    (test_char_dir / "card.json").write_text(
        json.dumps(test_card, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    # 测试 Classifier 读取
    classifier = Classifier(cfg, mock_client, "openai", "test-model", "测试角色")

    # 预期：角色名 + 自定义词 + 全局通用词
    expected = ["测试角色", "小测", "测试", "test", "hey", "嘿", "喂"]
    actual = classifier._wake_words

    status = "[OK]" if actual == expected else "[FAIL]"
    print(f"\n{status} 角色: 测试角色")
    print(f"  期望唤醒词: {expected}")
    print(f"  实际唤醒词: {actual}")

    if actual != expected:
        print(f"  差异: 缺失={set(expected)-set(actual)}, 多余={set(actual)-set(expected)}")

    # 清理测试数据
    (test_char_dir / "card.json").unlink()
    test_char_dir.rmdir()

    return actual == expected


def test_card_no_custom_wake_words():
    """测试：角色卡没有自定义唤醒词时，使用默认（角色名 + 全局）。"""
    print("\n\n" + "=" * 70)
    print("Test: Default wake words when card has no custom config")
    print("=" * 70)

    cfg = load_config()
    mock_client = MagicMock()

    # 使用现有角色（假设没有 wake_words 配置）
    classifier = Classifier(cfg, mock_client, "openai", "test-model", "默认")

    # 预期：角色名 + 全局通用词
    expected = ["默认", "hey", "嘿", "喂"]
    actual = classifier._wake_words

    status = "[OK]" if actual == expected else "[FAIL]"
    print(f"\n{status} 角色: 默认（无自定义唤醒词）")
    print(f"  期望唤醒词: {expected}")
    print(f"  实际唤醒词: {actual}")

    return actual == expected


def test_wake_words_priority():
    """测试：唤醒词优先级 = 角色名 > 自定义 > 全局。"""
    print("\n\n" + "=" * 70)
    print("Test: Wake words priority and deduplication")
    print("=" * 70)

    cfg = load_config()
    mock_client = MagicMock()

    # 创建测试角色卡（自定义词包含重复 + 全局词）
    test_char_dir = Path("D:/code/self/newTouch/data/characters/优先级测试")
    test_char_dir.mkdir(parents=True, exist_ok=True)

    test_card = {
        "name": "优先级测试",
        "extensions": {
            "wake_words": ["优先级测试", "hey", "小优"]  # 重复角色名和全局词
        }
    }
    (test_char_dir / "card.json").write_text(
        json.dumps(test_card, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    classifier = Classifier(cfg, mock_client, "openai", "test-model", "优先级测试")

    # 预期：去重后的列表（保持顺序：角色名 > 自定义 > 全局）
    expected = ["优先级测试", "hey", "小优", "嘿", "喂"]
    actual = classifier._wake_words

    status = "[OK]" if actual == expected else "[FAIL]"
    print(f"\n{status} 角色: 优先级测试（含重复词）")
    print(f"  期望唤醒词（去重）: {expected}")
    print(f"  实际唤醒词: {actual}")

    # 清理测试数据
    (test_char_dir / "card.json").unlink()
    test_char_dir.rmdir()

    return actual == expected


if __name__ == "__main__":
    results = []
    results.append(("Custom wake words", test_card_custom_wake_words()))
    results.append(("Default wake words", test_card_no_custom_wake_words()))
    results.append(("Priority and dedup", test_wake_words_priority()))

    print("\n\n" + "=" * 70)
    print("Summary: Character card wake words configuration")
    print("=" * 70)

    for name, passed in results:
        status = "[PASS]" if passed else "[FAIL]"
        print(f"{status} {name}")

    all_passed = all(r[1] for r in results)
    if all_passed:
        print("\nAll tests passed!")
        print("\nFeatures:")
        print("[OK] Read custom wake_words from character card extensions")
        print("[OK] Fallback to default (character name + global) when no custom")
        print("[OK] Priority: character name > custom > global")
        print("[OK] Automatic deduplication")
        print("\nUsage:")
        print("1. Open admin panel -> Character Card tab")
        print("2. Select a character")
        print("3. Fill in 'Wake Words' field (comma-separated)")
        print("4. Save card")
        print("5. Restart main.py or switch character to apply")
    else:
        print("\nSome tests failed. Check output above.")
