"""测试回复语言和翻译语言配置功能。"""
import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.character import _lang_instruction


def test_reply_chinese_no_translation():
    """中文回复，不翻译"""
    result = _lang_instruction("zh", "")
    assert result == "请始终用中文回复。"
    print("[PASS] 中文回复，不翻译")


def test_reply_english_no_translation():
    """英文回复，不翻译"""
    result = _lang_instruction("en", "")
    assert result == "请始终用English回复。"
    print("[PASS] 英文回复，不翻译")


def test_reply_english_translate_to_chinese():
    """英文回复，翻译成中文"""
    result = _lang_instruction("en", "zh")
    assert "English" in result
    assert "中文" in result
    assert "翻译" in result
    print("[PASS] 英文回复，翻译成中文")


def test_reply_japanese_translate_to_chinese():
    """日语回复，翻译成中文"""
    result = _lang_instruction("ja", "zh")
    assert "日本語" in result
    assert "中文" in result
    assert "翻译" in result
    print("[PASS] 日语回复，翻译成中文")


def test_reply_english_translate_to_japanese():
    """英文回复，翻译成日语"""
    result = _lang_instruction("en", "ja")
    assert "English" in result
    assert "日本語" in result
    assert "翻译" in result
    print("[PASS] 英文回复，翻译成日语")


def test_same_language_no_translation():
    """回复语言和翻译语言相同，不翻译"""
    result = _lang_instruction("en", "en")
    assert result == "请始终用English回复。"
    print("[PASS] 回复语言和翻译语言相同，不翻译")


def test_empty_reply_lang():
    """空回复语言，默认中文"""
    result = _lang_instruction("", "en")
    assert result == "请始终用中文回复。"
    print("[PASS] 空回复语言，默认中文")


def test_unknown_language_code():
    """未知语言码，使用原始码"""
    result = _lang_instruction("pt", "zh")
    assert "pt" in result
    assert "中文" in result
    print("[PASS] 未知语言码，使用原始码")


if __name__ == "__main__":
    test_reply_chinese_no_translation()
    test_reply_english_no_translation()
    test_reply_english_translate_to_chinese()
    test_reply_japanese_translate_to_chinese()
    test_reply_english_translate_to_japanese()
    test_same_language_no_translation()
    test_empty_reply_lang()
    test_unknown_language_code()
    print("\n[SUCCESS] All tests passed!")
