"""立绘库加载：情绪名 → 图片路径。仿语音库 load_voice_library（core/action/speak.py）。

立绘库文件：data/characters/{角色}/sprites.json
格式（镜像语音库 {emotions:{...}}）：
    {"emotions": {"neutral": {"image": "neutral.png"}, "得意": {"image": "smug.png"}}}
图片放同目录 data/characters/{角色}/sprites/。

与语音库完全独立：face 标签值（<face:得意>）与本库 key 对上即命中，
未命中回退 neutral，再未命中回退首张，都没有返回 None。
"""
from __future__ import annotations

import json
from pathlib import Path

from ..config import Config
from ..logger import get_logger

log = get_logger("sprite")


def _sprites_dir(config: Config, char_name: str) -> Path:
    """角色立绘目录：data/characters/{char}/sprites/（不自动创建，存在性由调用方判）。"""
    return config.project_root / "data" / "characters" / char_name / "sprites"


def _sprites_json_path(config: Config, char_name: str) -> Path:
    return _sprites_dir(config, char_name) / "sprites.json"


def load_sprites(config: Config, char_name: str) -> dict[str, str]:
    """读立绘库，返回 {情绪名: 图片绝对路径}。不存在/格式错返回 {}。

    每次现读（管理平台改完即生效，与语音库一致）。image 路径相对 sprites/ 目录，
    也可写绝对路径。
    """
    p = _sprites_json_path(config, char_name)
    if not p.exists():
        return {}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, ValueError, OSError):
        log.warning("立绘库解析失败: %s", p)
        return {}
    emotions = data.get("emotions") if isinstance(data, dict) else None
    if not isinstance(emotions, dict):
        return {}
    base = p.parent
    out: dict[str, str] = {}
    for emo, spec in emotions.items():
        if not isinstance(spec, dict):
            continue
        img = spec.get("image")
        if not img:
            continue
        # 相对路径相对 sprites/ 目录；绝对路径原样
        img_path = Path(img)
        full = img_path if img_path.is_absolute() else base / img_path
        out[emo] = str(full)
    return out


def load_face_emotions(config: Config, char_name: str) -> list[str]:
    """立绘库里有哪些表情档（供主脑 prompt 列出可选表情，库驱动，同 load_voice_emotions）。"""
    return list(load_sprites(config, char_name).keys())


def load_motion_map(config: Config, char_name: str) -> dict[str, str]:
    """读立绘库的 motion_map（{情绪: 动作}），表情->抖动动作映射。不存在返回 {}。

    由 LLM 在立绘库管理页生成、用户确认保存进 sprites.json 的 motion_map 字段。
    浮窗 SpriteMotion 用它覆盖代码默认 _DEFAULT_MAP。动作值校验为四选一。
    """
    p = _sprites_json_path(config, char_name)
    if not p.exists():
        return {}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, ValueError, OSError):
        return {}
    mm = data.get("motion_map") if isinstance(data, dict) else None
    if not isinstance(mm, dict):
        return {}
    valid = {"bounce", "jump", "shake", "none"}
    return {str(k): str(v) for k, v in mm.items() if str(v) in valid}


def image_path(face: str | None, mapping: dict[str, str]) -> str | None:
    """按 face 查立绘图路径。未命中回退 neutral，再未命中回退首张，都没有 None。

    Args:
        face: <face:> 标签值（如 "得意" / "happy" / None）
        mapping: load_sprites 返回的 {情绪名: 路径}
    """
    if not mapping:
        return None
    if face and face in mapping:
        return mapping[face]
    # 回退 neutral
    if "neutral" in mapping:
        return mapping["neutral"]
    # 再回退首张
    return next(iter(mapping.values()))
