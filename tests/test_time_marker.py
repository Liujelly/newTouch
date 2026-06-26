"""测试系统时间标记 [X分钟前] 的兜底扒除（v2.56）。

LLM 会照抄历史消息里的 [X分钟前] 标记到自己回复开头，prompt 禁令不可靠，
改在收口处硬扒。覆盖：
1. strip_leading_time_marker 纯函数（各种标记格式 / 连写 / 正文方括号不动）。
2. 反应路径 _strip_emotion_stream 流式跨 chunk 扒（标记跨 chunk 切割）。
3. _consume_prefix_tags emo/face/时间标记混合顺序。
"""
import asyncio
import tempfile
from pathlib import Path

import yaml

from core.action.speak import (
    Speaker,
    strip_leading_time_marker,
    _looks_like_partial_time_marker,
)
from core.config import Config


async def _mock_stream(text: str):
    """逐字符流式 yield（最易触发跨 chunk 切割问题）。"""
    for ch in text:
        yield ch
        await asyncio.sleep(0)


async def _mock_stream_chunks(chunks):
    """按给定 chunk 列表 yield（精准模拟标记跨 chunk 边界）。"""
    for c in chunks:
        yield c
        await asyncio.sleep(0)


def _make_speaker():
    temp_cfg = {"modules": {"tts": {"enabled": False, "provider": "gpt-sovits",
                                    "endpoint": "http://127.0.0.1:9880"}}}
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False,
                                     encoding="utf-8") as f:
        yaml.dump(temp_cfg, f)
        path = f.name
    cfg = Config(path)
    speaker = Speaker(cfg, "TestBot")
    original_t = speaker._t
    speaker._t = lambda key, default=None: False if key == "enabled" else original_t(key, default)
    return speaker, path


def test_strip_leading_time_marker():
    print("=" * 70)
    print("测试 strip_leading_time_marker（纯函数）")
    print("=" * 70)
    cases = [
        ("[12分钟前] うんうん、いい子だね。", "うんうん、いい子だね。"),
        ("[2分钟前]あ、やっと気づいた！", "あ、やっと気づいた！"),
        ("[3小时前]嗯，我在。", "嗯，我在。"),
        ("[2天前]那天说的事...", "那天说的事..."),
        ("[刚刚]你回来啦！", "你回来啦！"),
        ("   [5分钟前]  前导空白也剥。", "前导空白也剥。"),
        ("[12分钟前][8分钟前]连写两个都剥。", "连写两个都剥。"),
        ("正常回复没有标记", "正常回复没有标记"),
        ("回复中间有[备注]方括号不动", "回复中间有[备注]方括号不动"),  # 仅扒开头
        ("[1234天前]极端长数字也认。", "极端长数字也认。"),
    ]
    all_ok = True
    for inp, exp in cases:
        got = strip_leading_time_marker(inp)
        ok = got == exp
        all_ok = all_ok and ok
        print(f"  {'[OK]' if ok else '[FAIL]'} {inp!r}")
        if not ok:
            print(f"         expect={exp!r} got={got!r}")
    return all_ok


def test_partial_time_marker():
    print("\n" + "=" * 70)
    print("测试 _looks_like_partial_time_marker（跨 chunk 缓冲判断）")
    print("=" * 70)
    cases = [
        ("[12分", True), ("[3小时", True), ("[2天", True), ("[刚", True),
        ("[12分钟前", True),
        ("[12分钟前]", False),     # 已闭合，不再是 partial
        ("[完整正文", False),       # 含非数字单位字符
        ("正文开头", False),        # 不以 [ 开头
        ("[abcdefghijklmnopqrstuvwxyz]", False),  # 超长
        ("", False),
    ]
    all_ok = True
    for inp, exp in cases:
        got = _looks_like_partial_time_marker(inp)
        ok = got == exp
        all_ok = all_ok and ok
        print(f"  {'[OK]' if ok else '[FAIL]'} {inp!r} -> {got} (expect {exp})")
    return all_ok


async def test_stream_strip_time_marker():
    """反应路径：_strip_emotion_stream 流式扒时间标记（逐字符 + 跨 chunk）。"""
    print("\n" + "=" * 70)
    print("测试反应路径流式扒时间标记（_strip_emotion_stream）")
    print("=" * 70)
    speaker, path = _make_speaker()
    try:
        cases = [
            # (流式输入, 期望 _strip_emotion_stream 产出拼接)
            ("[12分钟前] うんうん、いい子だね。", "うんうん、いい子だね。"),
            ("[2分钟前]あ、やっと気づいた！", "あ、やっと気づいた！"),
            # emo 标签 + 时间标记（顺序：标签在前）
            ("<emo:happy>[5分钟前]你好呀！", "你好呀！"),
            # 时间标记 + emo 标签（顺序：标记在前）
            ("[5分钟前]<emo:happy>你好呀！", "你好呀！"),
            # emo + face + 时间标记 三者混合
            ("<emo:happy><face:得意>[5分钟前]嘿嘿！", "嘿嘿！"),
            # 无标记
            ("普通回复", "普通回复"),
        ]
        all_ok = True
        for inp, exp in cases:
            # 逐字符流（最严苛，标记必跨 chunk）
            out = []
            async for piece in speaker._strip_emotion_stream(_mock_stream(inp)):
                out.append(piece)
            got = "".join(out)
            ok = got == exp
            all_ok = all_ok and ok
            print(f"  {'[OK]' if ok else '[FAIL]'} {inp!r}")
            if not ok:
                print(f"         expect={exp!r} got={got!r}")
        return all_ok
    finally:
        Path(path).unlink(missing_ok=True)


async def test_stream_cross_chunk_boundary():
    """精准模拟标记跨 chunk 边界：'[12分' + '钟前] うん' + 'うん'。"""
    print("\n" + "=" * 70)
    print("测试标记跨 chunk 边界（精准 chunk 切分）")
    print("=" * 70)
    speaker, path = _make_speaker()
    try:
        chunks = ["[12分", "钟前] うん", "うん、いい子だね。"]
        out = []
        async for piece in speaker._strip_emotion_stream(_mock_stream_chunks(chunks)):
            out.append(piece)
        got = "".join(out)
        exp = "うんうん、いい子だね。"
        ok = got == exp
        print(f"  {'[OK]' if ok else '[FAIL]'} chunks={chunks}")
        if not ok:
            print(f"         expect={exp!r} got={got!r}")
        return ok
    finally:
        Path(path).unlink(missing_ok=True)


async def test_proactive_reply_strip():
    """主动路径：reply 字符串经 strip_leading_time_marker 扒除。"""
    print("\n" + "=" * 70)
    print("测试主动路径 reply 扒除（strip_leading_time_marker）")
    print("=" * 70)
    cases = [
        ("[12分钟前]小触会一直在这里。", "小触会一直在这里。"),
        ("[1小时前]我想你了。", "我想你了。"),
        ("没标记的独白", "没标记的独白"),
    ]
    all_ok = True
    for inp, exp in cases:
        got = strip_leading_time_marker(inp)
        ok = got == exp
        all_ok = all_ok and ok
        print(f"  {'[OK]' if ok else '[FAIL]'} {inp!r} -> {got!r}")
    return all_ok


async def main():
    results = [
        test_strip_leading_time_marker(),
        test_partial_time_marker(),
        await test_stream_strip_time_marker(),
        await test_stream_cross_chunk_boundary(),
        await test_proactive_reply_strip(),
    ]
    print("\n" + "=" * 70)
    print(f"总览: {'全部 PASS' if all(results) else '有 FAIL'} ({sum(results)}/{len(results)})")
    print("=" * 70)
    return 0 if all(results) else 1


if __name__ == "__main__":
    import sys
    sys.exit(asyncio.run(main()))
