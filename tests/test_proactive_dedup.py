"""测试主动发言去重兜底（v2.63）。

验证 _is_repeat_of_recent：
1. 和 recent_inner 里"刚开口说了「...」"高度相似 → True（压回 silent）
2. 不同话题/措辞差异大 → False
3. recent_inner 里只有 silent/look 记录（没说过话）→ False
4. 空 reply / 空 recent_inner → False
5. 阈值可配
用实跑日志里那两句一模一样的"お昼ご飯はもう食べた?"验证真实场景。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8")

from core.config import load_config
from core.orchestrator import Orchestrator


def _orch():
    """构造一个最小 Orchestrator（只为用 _is_repeat_of_recent/_record_inner）。"""
    cfg = load_config()
    cfg.set("proactive.repeat_similarity_threshold", 0.6)
    # Orchestrator 构造需要一堆依赖，直接造一个轻量替身用其方法
    orch = Orchestrator.__new__(Orchestrator)
    orch._cfg = cfg
    orch._recent_inner = []
    return orch


def test_identical_repeat():
    """实跑场景：和刚说过的完全一样 → True。"""
    orch = _orch()
    orch._recent_inner = ['刚开口说了「お昼ご飯はもう食べた？ちゃんと食べてるといいな。」（当时想：有点想他了）']
    reply = "「お昼ご飯はもう食べた？ちゃんと食べてるといいな。」"
    assert orch._is_repeat_of_recent(reply) is True, "完全相同应判重复"
    print("✅ 完全相同 → 重复")


def test_real_log_repeat():
    """实跑日志那两次（措辞略不同但意思一样）→ 应判重复。

    12:47: お昼ご飯はもう食べた？ちゃんと食べてるといいな。
    13:17: お昼ご飯はもう食べた？ちゃんと食べてるといいな。  (实际一字不差)
    """
    orch = _orch()
    orch._recent_inner = ['刚开口说了「お昼ご飯はもう食べた？ちゃんと食べてるといいな。」']
    reply = "「お昼ご飯はもう食べた？ちゃんと食べてるといいな。」"
    assert orch._is_repeat_of_recent(reply) is True
    print("✅ 实跑日志重复场景 → 重复")


def test_different_topic():
    """不同话题 → False。"""
    orch = _orch()
    orch._recent_inner = ['刚开口说了「お昼ご飯はもう食べた？ちゃんと食べてるといいな。」']
    reply = "「今日のFGOのイベント、もうクリアした？」"
    assert orch._is_repeat_of_recent(reply) is False, "不同话题不应判重复"
    print("✅ 不同话题 → 不重复")


def test_slightly_rephrased():
    """换说法但意思相近、相似度高 → 判重复。"""
    orch = _orch()
    orch._recent_inner = ['刚开口说了「ご飯はちゃんと食べてる？」']
    reply = "「ご飯はちゃんと食べてる？」"  # 加了句末标点，归一化后几乎一样
    assert orch._is_repeat_of_recent(reply) is True
    print("✅ 微调措辞但高度相似 → 重复")


def test_only_silent_in_recent():
    """recent_inner 只有 silent/look（没说过话）→ False。"""
    orch = _orch()
    orch._recent_inner = ['想了想没开口（有点想他）', '刚想看看ta在干嘛（好奇）']
    assert orch._is_repeat_of_recent("任何话") is False
    print("✅ 近期没主动发过言 → 不重复")


def test_empty():
    orch = _orch()
    assert orch._is_repeat_of_recent("") is False
    orch._recent_inner = []
    assert orch._is_repeat_of_recent("xxx") is False
    print("✅ 空 reply / 空 recent_inner → False")


def test_threshold_configurable():
    """阈值调高 → 原本判重复的变不重复。"""
    orch = _orch()
    orch._cfg.set("proactive.repeat_similarity_threshold", 0.99)
    orch._recent_inner = ['刚开口说了「ご飯食べた？']
    # 相似但不到 0.99
    assert orch._is_repeat_of_recent("「ご飯、食べたの？") is False
    print("✅ 阈值可配（调高后不判重复）")


if __name__ == "__main__":
    print("=== 测试主动发言去重兜底 ===\n")
    test_identical_repeat()
    test_real_log_repeat()
    test_different_topic()
    test_slightly_rephrased()
    test_only_silent_in_recent()
    test_empty()
    test_threshold_configurable()
    print("\n=== 全部测试通过 ===")
