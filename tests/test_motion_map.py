"""测试立绘抖动动作映射（v2.66 LLM 生成 + per-character motion_map）。

1. _parse_motion_map：LLM 输出解析（动作校验/情绪过滤/无 JSON 兜底）
2. load_motion_map：从 sprites.json 读 motion_map（含非法值过滤）
3. SpriteMotion：motion_map 覆盖代码默认 + 显式 none 被尊重
4. admin POST /api/sprites/{char}/auto-motion：注入假 LLM 回调，验生成 + 无表情 400
"""
import json
import shutil
import sys
import tempfile
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8")

from core.cognition import _parse_motion_map
from core.sprite.store import load_motion_map
from core.config import Config


class _TmpConfig(Config):
    """project_root 指向临时目录（load_motion_map 依赖 project_root）。"""
    def __init__(self, root):
        super().__init__({})
        self._root = root

    @property
    def project_root(self):
        return self._root


class _FakeCfg:
    """支持 dotted get 的假 Config（够 SpriteMotion 读 motion 配置）。"""
    def __init__(self, d):
        self._d = d

    def get(self, key, default=None):
        node = self._d
        for part in key.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node


def test_parse_motion_map():
    emos = ["得意", "惊讶", "生气", "neutral", "思考"]
    raw = '{"得意":"bounce","惊讶":"jump","生气":"shake","neutral":"none","思考":"none"}'
    m = _parse_motion_map(raw, emos)
    assert m == {"得意": "bounce", "惊讶": "jump", "生气": "shake",
                 "neutral": "none", "思考": "none"}, m
    print("✅ 正常解析")

    # 非法动作被滤（wiggle/dance 不在四选一）
    m2 = _parse_motion_map('{"得意":"bounce","惊讶":"wiggle","生气":"dance"}', emos)
    assert m2 == {"得意": "bounce"}, m2
    print("✅ 非法动作过滤")

    # 大小写不敏感 + 中文别名归一
    m3 = _parse_motion_map('{"得意":"BOUNCE","惊讶":"猛跳","生气":"左右晃","neutral":"不动"}', emos)
    assert m3 == {"得意": "bounce", "惊讶": "jump", "生气": "shake", "neutral": "none"}, m3
    print("✅ 大小写不敏感 + 中文别名归一")

    # 不在情绪列表里的被滤
    m4 = _parse_motion_map('{"得意":"bounce","未知":"bounce"}', emos)
    assert m4 == {"得意": "bounce"}, m4
    print("✅ 非列表情绪过滤")

    # 无 JSON / 畸形 -> {}
    assert _parse_motion_map("没有json", emos) == {}
    assert _parse_motion_map("{bad json}", emos) == {}
    print("✅ 无/畸形 JSON 返回空")


def test_load_motion_map():
    tmp = Path(tempfile.mkdtemp(prefix="newtouch_mm_"))
    d = tmp / "data" / "characters" / "测试" / "sprites"
    d.mkdir(parents=True)
    (d / "sprites.json").write_text(json.dumps({
        "emotions": {"得意": {"image": "a.png"}, "惊讶": {"image": "b.png"}},
        "motion_map": {"得意": "bounce", "惊讶": "jump", "坏": "wiggle"},
    }), encoding="utf-8")
    cfg = _TmpConfig(tmp)
    mm = load_motion_map(cfg, "测试")
    assert mm == {"得意": "bounce", "惊讶": "jump"}, mm  # "坏" 值非法被滤
    print(f"✅ load_motion_map: {mm}")

    # 无 motion_map 字段 -> {}
    (d / "sprites.json").write_text(json.dumps({"emotions": {"得意": {"image": "a.png"}}}), encoding="utf-8")
    assert load_motion_map(cfg, "测试") == {}
    # 文件不存在 -> {}
    assert load_motion_map(cfg, "不存在") == {}
    print("✅ 无 motion_map / 无文件 返回空")
    shutil.rmtree(tmp, ignore_errors=True)


def test_sprite_motion_map_override():
    try:
        from sprite_window import SpriteMotion
    except ImportError:
        print("⏭ 跳过 SpriteMotion 测试（PyQt6 未装）")
        return
    cfg = _FakeCfg({"sprite": {"motion": {"enabled": True, "amplitude": 14,
        "duration_ms": 500, "bounces": 3, "decay": 3.5}}})
    # motion_map 覆盖默认：得意默认 bounce -> 覆盖成 shake
    m = SpriteMotion(types.SimpleNamespace(), cfg, motion_map={"得意": "shake"})
    assert m._map["得意"] == "shake", m._map
    assert m._map["惊讶"] == "jump"  # 默认仍生效
    # 自定义情绪（不在默认表）进 map
    m2 = SpriteMotion(types.SimpleNamespace(), cfg, motion_map={"害羞": "bounce"})
    assert m2._map.get("害羞") == "bounce"
    print("✅ motion_map 覆盖默认 + 保留默认 + 自定义情绪")

    # 显式 none 被尊重（play 跳过）：得意=none -> play 不启动动画
    m3 = SpriteMotion(types.SimpleNamespace(), cfg, motion_map={"得意": "none"})
    assert m3._map["得意"] == "none"
    m3.play("得意")
    assert m3._anim is None, "显式 none 不应启动动画"
    print("✅ 显式 none 被尊重")


def test_admin_auto_motion():
    from fastapi.testclient import TestClient
    from api import admin
    c = TestClient(admin.app)
    TEST = "spmottest"
    d = Path(__file__).resolve().parent.parent / "data" / "characters" / TEST / "sprites"
    if d.exists():
        shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True)
    (d / "sprites.json").write_text(json.dumps({"emotions": {
        "得意": {"image": "a.png"}, "惊讶": {"image": "b.png"},
        "neutral": {"image": "c.png"}}}), encoding="utf-8")

    async def fake_gen(emos):
        return {e: ("bounce" if e == "得意" else "jump" if e == "惊讶" else "none") for e in emos}
    admin._generate_motion_fn = fake_gen
    try:
        r = c.post(f"/api/sprites/{TEST}/auto-motion")
        assert r.status_code == 200, r.text
        mm = r.json()["motion_map"]
        assert mm == {"得意": "bounce", "惊讶": "jump", "neutral": "none"}, mm
        print(f"✅ auto-motion 生成: {mm}")

        # 无表情 -> 400
        (d / "sprites.json").write_text(json.dumps({"emotions": {}}), encoding="utf-8")
        r2 = c.post(f"/api/sprites/{TEST}/auto-motion")
        assert r2.status_code == 400, r2.text
        print("✅ 无表情 400")
    finally:
        admin._generate_motion_fn = None
        shutil.rmtree(d.parent, ignore_errors=True)


if __name__ == "__main__":
    print("=== 测试立绘抖动动作映射 ===\n")
    test_parse_motion_map()
    test_load_motion_map()
    test_sprite_motion_map_override()
    test_admin_auto_motion()
    print("\n=== 全部测试通过 ===")
