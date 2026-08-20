"""测试近期内心活动缓冲的时间标记（v2.69）。

验证 _record_inner/_inner_for_prompt/_is_repeat_of_recent 三者协作：
1. _record_inner 存 {ts, text}，且条数上限 recent_inner_max 生效
2. _inner_for_prompt：距今 <2 分钟不加前缀（"刚…"文案本身准确），
   更久的加 "[X分钟前]/[X小时前] " 前缀，旧格式裸字符串退化为原文
3. 格式改 dict 后 _is_repeat_of_recent 去重不受影响（新旧格式都能比）
"""
import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8")

from core.config import load_config
from core.orchestrator import Orchestrator


def _orch():
    """构造一个最小 Orchestrator（只为用 _record_inner/_inner_for_prompt/_is_repeat_of_recent）。"""
    cfg = load_config()
    orch = Orchestrator.__new__(Orchestrator)
    orch._cfg = cfg
    orch._recent_inner = []
    return orch


def _ts_ago(seconds: float) -> str:
    return (datetime.now() - timedelta(seconds=seconds)).isoformat(timespec="seconds")


def test_record_inner_stores_ts():
    """_record_inner 存 {ts, text}，ts 可解析。"""
    orch = _orch()
    orch._record_inner("speak", "有点想他了", "ご飯はちゃんと食べてる？")
    assert len(orch._recent_inner) == 1
    e = orch._recent_inner[0]
    assert isinstance(e, dict) and "ts" in e and "text" in e
    # ts 能被 fromisoformat 解析（_rel_time 依赖）
    datetime.fromisoformat(e["ts"])
    assert "刚开口说了「ご飯はちゃんと食べてる？」" in e["text"]
    print("✅ _record_inner 存 {ts, text}，ts 可解析")


def test_record_inner_cap():
    """条数上限 recent_inner_max 生效（旧的被挤掉）。"""
    orch = _orch()
    cap = orch._cfg.get("proactive.recent_inner_max", 5)
    for i in range(cap + 3):
        orch._record_inner("silent", f"第{i}次想了他")
    assert len(orch._recent_inner) == cap
    # 留下的是最新的 cap 条
    assert f"第{cap + 2}次" in orch._recent_inner[-1]["text"]
    assert f"第0次" not in "".join(e["text"] for e in orch._recent_inner)
    print(f"✅ 条数上限 {cap} 生效，保留最新条目")


def test_inner_for_prompt_fresh_no_prefix():
    """距今 <2 分钟：不加前缀（条目文案"刚…"本身准确）。"""
    orch = _orch()
    orch._record_inner("silent", "想知道他在忙什么")
    out = orch._inner_for_prompt()
    assert len(out) == 1
    assert out[0].startswith("想了想没开口"), f"新鲜条目不应带时间前缀: {out[0]}"
    print("✅ <2 分钟条目无时间前缀")


def test_inner_for_prompt_old_gets_prefix():
    """距今 5 分钟 / 3 小时 / 2 天：分别加 [5分钟前]/[3小时前]/[2天前] 前缀。"""
    orch = _orch()
    orch._recent_inner = [
        {"ts": _ts_ago(5 * 60 + 10), "text": "刚开口说了「おはよう」"},
        {"ts": _ts_ago(3 * 3600 + 10), "text": "想了想没开口（有点寂寞）"},
        {"ts": _ts_ago(2 * 86400 + 10), "text": "刚想看看ta在干嘛"},
    ]
    out = orch._inner_for_prompt()
    assert out[0] == "[5分钟前] 刚开口说了「おはよう」", out[0]
    assert out[1] == "[3小时前] 想了想没开口（有点寂寞）", out[1]
    assert out[2] == "[2天前] 刚想看看ta在干嘛", out[2]
    print("✅ 旧条目按实际间隔加 [X分钟前]/[X小时前]/[X天前] 前缀")


def test_inner_for_prompt_bare_str_compat():
    """旧格式裸字符串（无 ts）：退化输出原文，不报错。"""
    orch = _orch()
    orch._recent_inner = ["刚开口说了「テスト」"]
    out = orch._inner_for_prompt()
    assert out == ["刚开口说了「テスト」"]
    print("✅ 裸字符串旧格式退化为原文")


def test_inner_for_prompt_empty():
    """空缓冲 -> 空列表。"""
    orch = _orch()
    assert orch._inner_for_prompt() == []
    print("✅ 空缓冲 -> 空列表")


def test_dedup_with_dict_entries():
    """dict 格式条目去重照常工作（新旧格式都能比）。"""
    orch = _orch()
    orch._recent_inner = [
        {"ts": _ts_ago(10 * 60), "text": "刚开口说了「お昼ご飯はもう食べた？ちゃんと食べてるといいな。」"},
        {"ts": _ts_ago(30 * 60), "text": "想了想没开口（有点想他）"},
    ]
    assert orch._is_repeat_of_recent("「お昼ご飯はもう食べた？ちゃんと食べてるといいな。」") is True
    assert orch._is_repeat_of_recent("「今日のFGOのイベント、もうクリアした？」") is False
    # 旧格式裸字符串也照常（回归 v2.63 行为）
    orch._recent_inner = ["刚开口说了「ご飯はちゃんと食べてる？」"]
    assert orch._is_repeat_of_recent("「ご飯はちゃんと食べてる？」") is True
    print("✅ dict/裸字符串两种格式去重均正常")


if __name__ == "__main__":
    print("=== 测试近期内心活动时间标记 ===\n")
    test_record_inner_stores_ts()
    test_record_inner_cap()
    test_inner_for_prompt_fresh_no_prefix()
    test_inner_for_prompt_old_gets_prefix()
    test_inner_for_prompt_bare_str_compat()
    test_inner_for_prompt_empty()
    test_dedup_with_dict_entries()
    print("\n=== 全部测试通过 ===")
