"""测试立绘浮窗子系统（v2.50）。

不依赖 PyQt6（GUI 手动验证），只测：
1. SpriteStore：加载 / 未命中回退 neutral / 再回退首张 / 空库 None
2. FaceBroadcaster：起 server，mock client 收 push JSON；无 client push 不崩
3. parse_face_prefix / strip_all_emotion_tags 清 face 标签
4. _parse_monologue(_cot) 解析 face 字段
"""
import asyncio
import asyncio
import json
import socket
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8")

from core.config import Config
from core.sprite.store import load_sprites, image_path, load_face_emotions
from core.sprite.broadcaster import FaceBroadcaster
from core.action.speak import (
    Speaker, parse_face_prefix, parse_emotion_prefix, strip_all_emotion_tags,
)
from core.cognition import _parse_monologue, _parse_monologue_cot


def _cfg_with_sprites(tmpdir: str, sprites_json: dict | None, char_name="test_char") -> Config:
    """构造一个 Config，立绘库写到临时目录 data/characters/{char}/sprites/sprites.json。"""
    d = Path(tmpdir) / "data" / "characters" / char_name / "sprites"
    d.mkdir(parents=True, exist_ok=True)
    if sprites_json is not None:
        (d / "sprites.json").write_text(json.dumps(sprites_json), encoding="utf-8")
    cfg = Config({"project_root_str": tmpdir})
    # Config.project_root 是写死的 _PROJECT_ROOT，store 用它。这里直接用临时 project_root
    # 覆盖：store._sprites_json_path 用 config.project_root，故构造一个临时 Config。
    return cfg


class _TmpConfig(Config):
    """Config 子类，project_root 指向临时目录（load_sprites 依赖 project_root）。"""
    def __init__(self, root: Path):
        super().__init__({})
        self._root = root

    @property
    def project_root(self):
        return self._root


def test_store_load_and_fallback():
    """SpriteStore 加载 + 回退逻辑。"""
    tmp = Path(tempfile.mkdtemp(prefix="newtouch_sprite_"))
    char_dir = tmp / "data" / "characters" / "小触" / "sprites"
    char_dir.mkdir(parents=True)
    (char_dir / "sprites.json").write_text(json.dumps({
        "emotions": {
            "neutral": {"image": "neutral.png"},
            "得意": {"image": "smug.png"},
            "思考": {"image": "thinking.png"},
        }
    }), encoding="utf-8")
    # 造 3 个空图文件供 path 存在性判断（image_path 只返回路径不查存在）
    for f in ("neutral.png", "smug.png", "thinking.png"):
        (char_dir / f).write_bytes(b"")

    cfg = _TmpConfig(tmp)
    mapping = load_sprites(cfg, "小触")
    assert set(mapping.keys()) == {"neutral", "得意", "思考"}, f"加载错误: {mapping}"
    assert mapping["得意"].endswith("smug.png"), f"路径错误: {mapping['得意']}"
    print("✅ load_sprites 加载正确")

    # 命中
    assert image_path("得意", mapping).endswith("smug.png")
    # 未命中回退 neutral
    assert image_path("惊讶", mapping).endswith("neutral.png")
    # 未命中且无 neutral 回退首张
    mapping2 = {"happy": "h.png", "sad": "s.png"}
    assert image_path("xxx", mapping2) == "h.png"
    # 空库
    assert image_path("happy", {}) is None
    assert image_path(None, {}) is None
    # face=None 回退 neutral
    assert image_path(None, mapping).endswith("neutral.png")
    print("✅ image_path 回退逻辑正确")

    # load_face_emotions
    emos = load_face_emotions(cfg, "小触")
    assert set(emos) == {"neutral", "得意", "思考"}
    print("✅ load_face_emotions 正确")

    import shutil
    shutil.rmtree(tmp, ignore_errors=True)


def test_store_empty():
    """无立绘库文件时返回空。"""
    tmp = Path(tempfile.mkdtemp(prefix="newtouch_sprite_e_"))
    cfg = _TmpConfig(tmp)
    assert load_sprites(cfg, "不存在") == {}
    assert load_face_emotions(cfg, "不存在") == []
    print("✅ 无立绘库返回空")
    import shutil
    shutil.rmtree(tmp, ignore_errors=True)


async def _broadcaster_scenario():
    """broadcaster 起 server + mock client 收 push。"""
    b = FaceBroadcaster("127.0.0.1", 0)  # port=0 让 OS 分配
    # 用 0 端口需要拿到实际端口；改用一个固定高位端口更稳
    import random
    port = random.randint(20000, 29999)
    b = FaceBroadcaster("127.0.0.1", port)
    await b.start()
    assert b._started, "broadcaster 应启动成功"

    # mock client 连上
    sock = socket.create_connection(("127.0.0.1", port), timeout=3)
    await asyncio.sleep(0.3)  # 等 server accept
    assert b.has_clients(), "应有 client 连上"

    # push
    await b.push("得意", "小触")
    await asyncio.sleep(0.3)
    data = sock.recv(4096).decode("utf-8").strip()
    obj = json.loads(data)
    assert obj == {"face": "得意", "character": "小触"}, f"收到错误: {obj}"
    print(f"✅ broadcaster 推送正确: {obj}")

    # 无 client 时 push 不崩（关掉 client）
    sock.close()
    await asyncio.sleep(0.3)
    await b.push("思考", "小触")  # 不应抛异常
    print("✅ 无 client 时 push 不崩")

    await b.stop()


def test_broadcaster():
    asyncio.run(_broadcaster_scenario())


def test_face_tag_parse():
    """parse_face_prefix / strip_all 清 face 标签。"""
    face, rest = parse_face_prefix("<face:得意>嘿嘿")
    assert face == "得意" and rest == "嘿嘿", f"face 解析错误: {face!r} {rest!r}"
    print("✅ parse_face_prefix 正确")

    # 无标签
    assert parse_face_prefix("普通文本") == (None, "普通文本")

    # emo + face 都在开头
    emo, rest = parse_emotion_prefix("<emo:happy><face:得意>文本")
    assert emo == "happy"
    face, rest2 = parse_face_prefix(rest)
    assert face == "得意" and rest2 == "文本"
    print("✅ emo+face 连续解析正确")

    # strip_all 清两种标签（任意位置）
    cleaned = strip_all_emotion_tags("<emo:happy>开头<face:得意>中间<emo:sad>末尾")
    assert cleaned == "开头中间末尾", f"清理错误: {cleaned!r}"
    print("✅ strip_all_emotion_tags 清理 emo+face 正确")


async def _stream_chunks(chunks):
    for c in chunks:
        yield c
        await asyncio.sleep(0.001)


def test_cross_chunk_face_recovery():
    """回归：emo 和 face 标签跨 chunk 分开发时，face 仍能被补广播、不进文本。

    场景：LLM 流式把 <emo:happy> 和 <face:得意> 分两个 chunk 发。
    开头解析只拿到 emo，face 落在后续 chunk；收尾从全文扫一次补广播。
    """
    async def run():
        cfg = Config({"modules": {"tts": {"enabled": False}}})
        sp = Speaker(cfg, "T")
        out = await sp.speak(
            _stream_chunks(["<emo:happy>", "<face:得意>", "嘿嘿，我在呀~"]),
            on_text=lambda t: None,
        )
        await asyncio.sleep(0.05)  # 等 create_task 的广播跑完
        assert sp._cur_emotion == "happy", f"emo 应解析: {sp._cur_emotion}"
        assert sp._cur_face == "得意", f"face 应被补广播: {sp._cur_face}"
        assert "face" not in out and "emo" not in out, f"标签不应进文本: {out!r}"
    asyncio.run(run())
    print("✅ 跨 chunk face 补广播、标签不进文本")


def test_parse_monologue_face():
    """JSON 独白解析 face 字段。"""
    raw = '{"thought":"想他了","action":"speak","text":"在吗","emotion":"happy","face":"得意"}'
    r = _parse_monologue(raw, use_cot=False)
    assert r["face"] == "得意", f"JSON face 错误: {r['face']}"
    assert r["emotion"] == "happy"
    print("✅ JSON 独白解析 face 正确")

    # 无 face 字段（向后兼容）
    raw2 = '{"thought":"想他了","action":"silent","text":""}'
    r2 = _parse_monologue(raw2, use_cot=False)
    assert r2["face"] == "", f"无 face 应为空: {r2['face']}"
    print("✅ JSON 无 face 向后兼容")


def test_parse_monologue_cot_face():
    """CoT 独白解析 [face:得意] 标记。"""
    raw = '嗯…他还没回。\n[决定：开口] "在忙吗" [face:得意]'
    r = _parse_monologue_cot(raw)
    assert r["action"] == "speak", f"action 错误: {r['action']}"
    assert r["text"] == "在忙吗", f"text 错误: {r['text']}"
    assert r["face"] == "得意", f"CoT face 错误: {r['face']}"
    # face 标记不应残留在 thought/text
    assert "face" not in r["thought"].lower()
    print("✅ CoT 独白解析 [face:] 正确")

    # silent 也带 face
    raw2 = '算了不打扰。\n[决定：沉默] [face:思考]'
    r2 = _parse_monologue_cot(raw2)
    assert r2["action"] == "silent"
    assert r2["face"] == "思考"
    print("✅ CoT silent 带 face 正确")

    # 无 face 标记（向后兼容）
    raw3 = '[决定：沉默]'
    r3 = _parse_monologue_cot(raw3)
    assert r3["face"] == ""
    print("✅ CoT 无 face 向后兼容")


if __name__ == "__main__":
    print("=== 测试立绘浮窗子系统 ===\n")
    test_store_load_and_fallback()
    test_store_empty()
    test_broadcaster()
    test_face_tag_parse()
    test_cross_chunk_face_recovery()
    test_parse_monologue_face()
    test_parse_monologue_cot_face()
    print("\n=== 全部测试通过 ===")
