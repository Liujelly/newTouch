"""回复审查（v2.68）：机械检查 AI 回复是否符合标准，不合格则交 LLM 在原回复上修正。

只做机械检查（免费、快、确定性）：循环重复 / 过长 / 系统标记泄漏。
LLM 修正（cognition.fix_reply）由 orchestrator 在检查不合格时调用。
审查标准可扩（翻译缺失等后续再加）。

启用条件：config.reply_review.enabled（默认 false）。开启后反应路径回复
不走流式--先缓存全文->审查->不合格调 LLM 修正->_once 一次性播。
"""
from __future__ import annotations

import re

from .logger import get_logger

log = get_logger("review")

# 系统标记泄漏：这些不该出现在给用户的回复正文里
_MARKER_PATTERNS = [
    re.compile(r"\[\d+(?:分钟|小时|天)前\]"),   # [X分钟前] 历史时间标记
    re.compile(r"\[刚刚\]"),
    re.compile(r"\[决定[：:]"),                  # [决定：开口/沉默] CoT 决策标记
    re.compile(r"\[face[：:]"),                  # [face:xxx] CoT face 标记
    re.compile(r"\[emo[：:]"),                   # [emo:xxx]
    re.compile(r"<emo:[^>]*>"),                  # <emo:xxx>（leading 已被 parse 掉，残留即泄漏）
    re.compile(r"<face:[^>]*>"),                 # <face:xxx>
]

# 循环检测：找不重叠重复出现的块
_LOOP_BLOCK_LEN = 40   # 块最小长度（字）
_LOOP_MIN_REPEAT = 2   # 重复此次数即判循环


def detect_loop(text: str, block_len: int = _LOOP_BLOCK_LEN,
                min_repeat: int = _LOOP_MIN_REPEAT) -> bool:
    """检测 text 中是否有 >= block_len 的子串不重叠重复 >= min_repeat 次。"""
    n = len(text)
    if n < block_len * min_repeat:
        return False
    seen: dict[str, int] = {}
    for i in range(n - block_len + 1):
        block = text[i:i + block_len]
        j = seen.get(block)
        if j is not None and i - j >= block_len:
            return True
        if j is None:
            seen[block] = i
    return False


def detect_marker_leak(text: str) -> bool:
    """检测 text 是否含系统标记泄漏（[X分钟前]/[决定：]/散落 <emo:><face:> 等）。"""
    return any(p.search(text) for p in _MARKER_PATTERNS)


def review_reply(text: str, config) -> list[str]:
    """机械审查回复正文，返回问题清单（空 = 合格）。

    text 应是剥掉开头 emo/face 标签后的可见正文。
    """
    if not text:
        return []
    issues: list[str] = []
    if detect_loop(text):
        issues.append("内容有重复循环（同一段落重复出现）")
    max_len = int(config.get("reply_review.max_length", 400))
    if max_len > 0 and len(text) > max_len:
        issues.append(f"回复过长（{len(text)}字，超过{max_len}）")
    if detect_marker_leak(text):
        issues.append("含有系统标记泄漏（如 [X分钟前]、[决定：]、散落的 <emo:>/<face:>）")
    return issues
