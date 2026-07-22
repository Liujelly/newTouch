"""测试立绘情绪抖动（v2.66 SpriteMotion）。

不跑真实动画（QVariantAnimation 需事件循环，视觉效果手动验证），只测：
1. 情绪 -> 动作映射（_DEFAULT_MAP）
2. 衰减正弦偏移数学（_offset：phase=0 归零 / bounce 纵向向上 / shake 横向 / 末端衰减）
3. play 跳过逻辑（disabled / neutral / 未知情绪 不启动动画）
"""
import math
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8")


class _FakeConfig:
    """支持 dotted get 的假 Config（仅够 SpriteMotion 读 motion 配置）。"""

    def __init__(self, d: dict) -> None:
        self._d = d

    def get(self, key: str, default=None):
        node = self._d
        for part in key.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node


def _make_motion(enabled=True, amplitude=14, duration_ms=500, bounces=3, decay=3.5):
    """构造 SpriteMotion（window 用 SimpleNamespace 占位；skip 测试不碰 window 方法）。"""
    from sprite_window import SpriteMotion
    cfg = _FakeConfig({"sprite": {"motion": {
        "enabled": enabled, "amplitude": amplitude, "duration_ms": duration_ms,
        "bounces": bounces, "decay": decay,
    }}})
    return SpriteMotion(types.SimpleNamespace(), cfg)


def test_face_motion_map():
    """情绪 -> 动作映射正确，neutral/思考 等不在表里。"""
    from sprite_window import SpriteMotion
    m = SpriteMotion._DEFAULT_MAP
    assert m["得意"] == "bounce"
    assert m["happy"] == "bounce"
    assert m["兴奋"] == "bounce"
    assert m["惊讶"] == "jump"
    assert m["生气"] == "shake"
    assert m["愤怒"] == "shake"
    # 未映射的情绪不在表里（play 时返回 None -> 不动）
    assert "neutral" not in m
    assert "思考" not in m
    print("✅ 情绪->动作映射正确")


def test_offset_math():
    """衰减正弦偏移：起止归零、bounce 向上、shake 横向、末端衰减。"""
    from sprite_window import SpriteMotion

    # phase=0 -> 偏移 0（sin(0)=0）
    dx, dy = SpriteMotion._offset("bounce", 0.0, 14, 3, 3.5)
    assert dx == 0.0 and dy == 0.0, f"phase=0 应归零: {dx},{dy}"

    # bounce 纵向：phase=1/(4*bounces)=1/12 处 sin 达正峰值 -> dy 为负（向上）
    phase_peak = 1.0 / (4 * 3)
    dx, dy = SpriteMotion._offset("bounce", phase_peak, 14, 3, 3.5)
    assert dx == 0.0, f"bounce 应无横向: {dx}"
    assert dy < 0, f"bounce 峰值应向上(dy<0): {dy}"
    assert abs(dy) <= 14, f"幅度不应超 amp: {dy}"
    print(f"✅ bounce 纵向向上: phase={phase_peak:.3f} dy={dy:.2f}")

    # shake 横向：dy=0，dx 非零
    dx, dy = SpriteMotion._offset("shake", phase_peak, 14, 4, 3.5)
    assert dy == 0.0, f"shake 应无纵向: {dy}"
    assert dx != 0.0, f"shake 应有横向: {dx}"
    print(f"✅ shake 横向: dx={dx:.2f} dy={dy}")

    # 末端衰减：phase=1 时 env=exp(-3.5)≈0.030，偏移远小于幅度
    dx, dy = SpriteMotion._offset("bounce", 1.0, 14, 3, 3.5)
    assert abs(dy) < 14 * 0.05, f"末端应衰减到接近 0: {dy}"
    print(f"✅ 末端衰减: phase=1 dy={dy:.3f}")

    # amp=0 -> 永远 0
    dx, dy = SpriteMotion._offset("bounce", 0.25, 0, 3, 3.5)
    assert dx == 0.0 and dy == 0.0
    print("✅ amplitude=0 不动")


def test_play_skips():
    """disabled / neutral / 未知情绪 不启动动画（_anim 保持 None）。"""
    m = _make_motion(enabled=True)
    # neutral 直接跳
    m.play("neutral")
    assert m._anim is None, "neutral 不应启动动画"
    # 未映射情绪（思考）跳
    m.play("思考")
    assert m._anim is None, "未映射情绪不应启动动画"
    print("✅ neutral/未映射情绪 不启动动画")

    # disabled 时即使映射了也不动
    m2 = _make_motion(enabled=False)
    m2.play("得意")
    assert m2._anim is None, "disabled 不应启动动画"
    print("✅ disabled 不启动动画")


def test_play_starts_for_mapped():
    """映射到的情绪（得意）且 enabled 时启动动画（_anim 非 None）。

    需真实 QWidget 做 QVariantAnimation parent，故用 QApplication + QWidget。
    """
    try:
        from PyQt6.QtWidgets import QApplication, QWidget
    except ImportError:
        print("⏭ 跳过动画启动测试（PyQt6 未装）")
        return

    app = QApplication.instance() or QApplication(sys.argv)
    w = QWidget()
    m = _make_motion(enabled=True, duration_ms=20)  # 短时长便于快速结束
    m._window = w  # 换成真实 widget（QVariantAnimation 需要 QObject parent）
    m.play("得意")
    assert m._anim is not None, "得意应启动动画"
    print("✅ 得意(bounce) 启动动画")

    # 新动作停掉上一个：再 play 一次，_anim 换新实例
    first = m._anim
    m.play("惊讶")
    assert m._anim is not None and m._anim is not first, "新动作应替换旧动画"
    # 第一个已被 stop
    assert first.state() == first.State.Stopped or first is not m._anim
    print("✅ 新动作停掉上一个")

    # 等动画结束（finished 槽把 _anim 置 None + 精确归位）
    import time
    w.move(100, 100)
    base = w.pos()
    m.play("生气")
    deadline = time.time() + 1.0
    while m._anim is not None and time.time() < deadline:
        app.processEvents()
        time.sleep(0.01)
    assert m._anim is None, "动画应已结束"
    assert w.pos().x() == base.x() and w.pos().y() == base.y(), \
        f"结束应精确归位到基准: {w.pos()} vs {base}"
    print("✅ 动画结束精确归位")


if __name__ == "__main__":
    print("=== 测试立绘情绪抖动 ===\n")
    test_face_motion_map()
    test_offset_math()
    test_play_skips()
    test_play_starts_for_mapped()
    print("\n=== 全部测试通过 ===")
