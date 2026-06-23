"""配置加载: 读 data/config.yaml，做 ${ENV_VAR} 环境变量替换。"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

import yaml

_ENV_PATTERN = re.compile(r"\$\{([A-Z_][A-Z0-9_]*)\}")
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_CONFIG_PATH = _PROJECT_ROOT / "data" / "config.yaml"


def _load_dotenv(path: Path) -> None:
    """极简 .env 加载: KEY=VALUE 逐行读入 os.environ (不覆盖已存在的)。"""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key, val = key.strip(), val.strip().strip('"').strip("'")
        os.environ.setdefault(key, val)  # 已设的系统/窗口变量优先


def _substitute_env(value: Any) -> Any:
    """递归地把字符串里的 ${VAR} 替换成环境变量值（缺失则保留原样并留空）。"""
    if isinstance(value, str):
        def repl(m: re.Match) -> str:
            return os.environ.get(m.group(1), "")
        return _ENV_PATTERN.sub(repl, value)
    if isinstance(value, dict):
        return {k: _substitute_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_substitute_env(v) for v in value]
    return value


class Config:
    """轻量配置包装：点路径访问 + 运行时可写（供热重载/自我配置用）。"""

    def __init__(self, data: dict):
        self._data = data

    def get(self, dotted: str, default: Any = None) -> Any:
        node: Any = self._data
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def set(self, dotted: str, value: Any) -> None:
        node = self._data
        parts = dotted.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value

    @property
    def raw(self) -> dict:
        return self._data

    def reload(self, path: Path | None = None) -> None:
        """从磁盘重新加载配置并**原地**替换内部数据。

        管理平台改配置时只写文件 config.yaml；运行中的各模块持有的是本 Config
        对象的引用、每次 get() 读 self._data。此方法把磁盘最新内容（含 ${ENV} 替换
        + self_config.json 覆盖）重新读进来替换 self._data，于是所有"现读"模块下次
        get() 即拿到新值——无需重建模块、无需重启进程。

        注意：只有"每次使用时才 get()"的字段会即时生效（如 TTS/视觉开关/语速等）；
        构造时就把值缓存进实例变量或建好了客户端的字段（LLM/VLM/记忆/STT 客户端）
        不会因 reload 改变，仍需重启。
        """
        cfg_path = path or _CONFIG_PATH
        _load_dotenv(_PROJECT_ROOT / ".env")
        with open(cfg_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        new = _substitute_env(data)
        # 原地替换：保持 self._data 是同一个 dict 对象的内容被换掉
        self._data.clear()
        self._data.update(new)
        _apply_overrides(self, _PROJECT_ROOT / "data" / "self_config.json")

    @property
    def project_root(self) -> Path:
        return _PROJECT_ROOT

    def char_data_dir(self, char_name: str | None = None) -> Path:
        """角色专属数据目录 data/characters/{name}/，不存在时自动创建。"""
        name = char_name or self.get("character.name", "默认")
        d = _PROJECT_ROOT / "data" / "characters" / name
        d.mkdir(parents=True, exist_ok=True)
        return d


def _apply_overrides(cfg: "Config", overrides_path: Path) -> None:
    """应用 AI 自我配置的白名单覆盖（data/self_config.json）。

    self_config.py 只把白名单子集写进这个文件；启动时叠加到 config 之上，
    使 ta 上次的自我配置（音色/频率/视觉/预设）在重启后仍生效。
    config.yaml 本身从不被回写（避免泄露 ${ENV} 实值）。
    """
    if not overrides_path.exists():
        return
    try:
        overrides = json.loads(overrides_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, ValueError, OSError):
        return
    if isinstance(overrides, dict):
        for dotted, value in overrides.items():
            cfg.set(dotted, value)


def load_config(path: Path | None = None) -> Config:
    _load_dotenv(_PROJECT_ROOT / ".env")   # 先把 .env 加载进 os.environ
    cfg_path = path or _CONFIG_PATH
    with open(cfg_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    cfg = Config(_substitute_env(data))
    _apply_overrides(cfg, _PROJECT_ROOT / "data" / "self_config.json")
    return cfg
