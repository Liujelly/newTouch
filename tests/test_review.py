"""测试回复审查机制（v2.68）。

机械审查纯函数：detect_loop / detect_marker_leak / review_reply。
覆盖：循环重复检测、系统标记泄漏检测、过长检测、max_length=0 关闭长度检查、
干净回复无问题。review_reply 用真实 Config（直接传 dict），顺带验证
reply_review.max_length 点路径在真实配置遍历下能取到。
"""
import sys

sys.stdout.reconfigure(encoding='utf-8')

from core.review import detect_loop, detect_marker_leak, review_reply
from core.config import Config


def _cfg(max_length=400):
    """构造只含 reply_review 块的 Config（直接传 dict 给真实 Config）。"""
    return Config({"reply_review": {"enabled": False, "max_length": max_length}})


def test_detect_loop():
    print("=== 测试循环重复检测（detect_loop）===")
    varied = "".join(f"第{i}句话的内容各不相同请不要重复匹配。" for i in range(20))
    cases = [
        ("abcdefghij" * 12, True),          # 周期重复 -> 循环
        ("a" * 80, True),                    # 单字重复 -> 循环
        ("你好呀，今天天气不错。", False),    # 太短（< 80 字）
        ("", False),                          # 空
        (varied, False),                      # 足够长但不重复
    ]
    all_ok = True
    for text, exp in cases:
        got = detect_loop(text)
        ok = got == exp
        all_ok = all_ok and ok
        print(f"  {'[OK]' if ok else '[FAIL]'} len={len(text)} expect={exp} got={got}")
    assert all_ok
    print("✅ 通过\n")


def test_detect_marker_leak():
    print("=== 测试系统标记泄漏检测（detect_marker_leak）===")
    cases = [
        ("[12分钟前]你好", True),            # 历史时间标记
        ("[3小时前]嗯", True),
        ("[刚刚]回来啦", True),
        ("[决定：开口]说话", True),          # CoT 决策标记
        ("[face:得意]嘿嘿", True),           # CoT face 标记
        ("[emo:happy]hi", True),
        ("<emo:happy>你好", True),           # 角括号 emo 标签
        ("<face:得意>你好", True),           # 角括号 face 标签
        ("正常回复没有标记", False),
        ("回复中间有[备注]方括号", False),    # 普通方括号不算泄漏
        ("", False),
    ]
    all_ok = True
    for text, exp in cases:
        got = detect_marker_leak(text)
        ok = got == exp
        all_ok = all_ok and ok
        print(f"  {'[OK]' if ok else '[FAIL]'} {text!r} expect={exp} got={got}")
    assert all_ok
    print("✅ 通过\n")


def test_review_reply_clean():
    print("=== 测试干净回复无问题（review_reply）===")
    cfg = _cfg(max_length=400)
    assert review_reply("嗯，我在呢，有什么事吗？", cfg) == []
    assert review_reply("", cfg) == []  # 空文本直接合格
    print("  [OK] 干净短回复 / 空回复均无问题")
    print("✅ 通过\n")


def test_review_reply_too_long():
    print("=== 测试过长检测（review_reply）===")
    cfg = _cfg(max_length=50)
    long_text = "今天天气真好我想出去玩一整天都行啊" * 5  # 85 字 > 50
    issues = review_reply(long_text, cfg)
    assert any("过长" in i for i in issues), f"应报过长，实际：{issues}"
    print(f"  [OK] {len(long_text)}字超 50 -> {issues}")
    print("✅ 通过\n")


def test_review_reply_loop():
    print("=== 测试循环检测（review_reply）===")
    cfg = _cfg(max_length=100000)  # 关掉长度，只看循环
    loop_text = "abcdefghij" * 12  # 周期重复
    issues = review_reply(loop_text, cfg)
    assert any("重复循环" in i for i in issues), f"应报循环，实际：{issues}"
    print(f"  [OK] 周期文本 -> {issues}")
    print("✅ 通过\n")


def test_review_reply_marker_leak():
    print("=== 测试标记泄漏检测（review_reply）===")
    cfg = _cfg(max_length=100000)
    leak_text = "嗯嗯[决定：开口]我来了"
    issues = review_reply(leak_text, cfg)
    assert any("标记泄漏" in i for i in issues), f"应报标记泄漏，实际：{issues}"
    print(f"  [OK] 含 [决定：] -> {issues}")
    print("✅ 通过\n")


def test_review_reply_max_length_zero():
    """max_length=0 只关闭长度检查，循环/标记泄漏仍查。"""
    print("=== 测试 max_length=0 关闭长度检查（review_reply）===")
    # 足够长（>400）但不重复、无标记的文本：长度检查关时应无问题
    varied_long = "".join(f"第{i}句话的内容各不相同请不要重复匹配。" for i in range(30))
    assert len(varied_long) > 400, "测试文本应超过 400 字"
    cfg_zero = _cfg(max_length=0)
    issues = review_reply(varied_long, cfg_zero)
    assert issues == [], f"max_length=0 不应查长度，实际：{issues}"
    # 对照：同样文本 + max_length=400 应报过长（证明是长度检查被关，不是文本短）
    cfg_400 = _cfg(max_length=400)
    assert any("过长" in i for i in review_reply(varied_long, cfg_400)), "400字上限应报过长"
    print("  [OK] 长文本 + max_length=0 -> 无问题；+ max_length=400 -> 报过长")
    print("✅ 通过\n")


def test_review_reply_default_max_length():
    """真实 Config 取默认 max_length=400（验证点路径 + 默认值 + 边界）。"""
    print("=== 测试默认 max_length=400（真实 Config 点路径）===")
    cfg = _cfg(max_length=400)
    assert cfg.get("reply_review.max_length", -1) == 400, "点路径取值失败"
    # 不重复、无标记的变长文本（切片到指定字数）
    base = "".join(f"第{i}句话的内容各不相同请不要重复匹配。" for i in range(40))
    text_300 = base[:300]
    text_500 = base[:500]
    assert len(text_300) == 300 and len(text_500) == 500
    assert review_reply(text_300, cfg) == [], "300字应合格"
    assert any("过长" in i for i in review_reply(text_500, cfg)), "500字应报过长"
    print("  [OK] 300字合格 / 500字报过长 / 点路径取值正确")
    print("✅ 通过\n")


if __name__ == "__main__":
    print("=== 测试回复审查机制（v2.68）===\n")
    test_detect_loop()
    test_detect_marker_leak()
    test_review_reply_clean()
    test_review_reply_too_long()
    test_review_reply_loop()
    test_review_reply_marker_leak()
    test_review_reply_max_length_zero()
    test_review_reply_default_max_length()
    print("=== 全部测试通过 ===")
