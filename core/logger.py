"""统一日志模块。

替换散落各处的 ``print``，把全流程运行日志（LLM 调用 / 工具 / TTS / STT /
视觉 / 记忆 / 心跳 / 错误）统一写到 ``data/logs/system.log``，按日轮转，
管理平台「日志」Tab 实时读取展示。

设计要点
--------
- 基于 stdlib ``logging``，``get_logger(module)`` 返回 ``newtouch.<module>``
  子 logger，调用方零成本。
- import 安全：模块导入即给 root 装一个 stdout handler 兜底，即便
  ``setup_logging`` 没被调用（如测试）也能看到输出。
- ``setup_logging(cfg)`` 在 main.py 早期调用一次：加 ``TimedRotatingFileHandler``
  按日轮转（``when='midnight'``，保留 ``backup_count`` 天），按 ``logging.level``
  设全局级别。stdout handler 保留（控制台行为不变）。
- 日志行格式固定可解析：
  ``2026-06-23 15:30:00 INFO [orch] 消息``
  ``read_logs`` 按此解析；不匹配的续行（如 traceback）并入上一条 ``msg``。
"""
from __future__ import annotations

import copy
import logging
import re
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path
from typing import Any

_ROOT_NAME = "newtouch"
_FMT = "%(asctime)s %(levelname)s [%(name)s] %(message)s"
_DATEFMT = "%Y-%m-%d %H:%M:%S"
# 子 logger 名去掉 ``newtouch.`` 前缀后写入 [module]；root logger 名即 "newtouch"
_LINE_RE = re.compile(
    r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) "
    r"(DEBUG|INFO|WARNING|ERROR|CRITICAL) \[([^\]]+)\] (.*)$"
)

_configured = False


class _ModuleFilter(logging.Filter):
    """只放行 ``newtouch`` 下的日志，避免 uvicorn/httpx 等第三方噪音进文件。"""

    def filter(self, record: logging.LogRecord) -> bool:
        return record.name == _ROOT_NAME or record.name.startswith(_ROOT_NAME + ".")


def _module_of(record: logging.LogRecord) -> str:
    name = record.name
    return name[len(_ROOT_NAME) + 1:] if name.startswith(_ROOT_NAME + ".") else name


class _Formatter(logging.Formatter):
    """把 ``newtouch.orch`` 显示成 ``[orch]``。

    注意：LogRecord 会被多个 handler 共享，**绝不能原地改 record.name**——
    否则第一个 handler 处理后，第二个 handler 的 filter 看到的 name 已被污染
    （newtouch.orch → orch），导致 filter 误拒、日志丢失。用临时副本格式化。
    """

    def format(self, record: logging.LogRecord) -> str:  # noqa: D401
        # 用一个改了 name 的浅拷贝做格式化，不动原 record
        rec = copy.copy(record)
        rec.name = _module_of(record)
        return super().format(rec)


def _ensure_stdout() -> None:
    root = logging.getLogger(_ROOT_NAME)
    for h in root.handlers:
        if isinstance(h, logging.StreamHandler) and not isinstance(h, TimedRotatingFileHandler):
            return
    h = logging.StreamHandler()
    h.setFormatter(_Formatter(_FMT, _DATEFMT))
    h.addFilter(_ModuleFilter())
    root.addHandler(h)
    root.setLevel(logging.INFO)
    root.propagate = False


def setup_logging(cfg: Any) -> None:
    """配置文件日志（按日轮转）+ 全局级别。main.py 启动早期调用一次。"""
    global _configured
    log_cfg = cfg.get("logging", {}) or {}
    if not log_cfg.get("enabled", True):
        _ensure_stdout()
        _configured = True
        return

    level_name = str(log_cfg.get("level", "INFO")).upper()
    level = getattr(logging, level_name, logging.INFO)

    log_dir = Path(log_cfg.get("dir", "data/logs"))
    if not log_dir.is_absolute():
        log_dir = cfg.project_root / log_dir
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "system.log"

    root = logging.getLogger(_ROOT_NAME)
    # 幂等：重复 setup 不重复加 handler
    for h in list(root.handlers):
        if isinstance(h, TimedRotatingFileHandler):
            root.removeHandler(h)
    fh = TimedRotatingFileHandler(
        str(log_path), when="midnight", backupCount=int(log_cfg.get("backup_count", 14)),
        encoding="utf-8", delay=True, utc=False,
    )
    fh.setFormatter(_Formatter(_FMT, _DATEFMT))
    fh.addFilter(_ModuleFilter())
    root.addHandler(fh)
    _ensure_stdout()
    root.setLevel(level)
    _configured = True
    logging.getLogger(_ROOT_NAME).info(
        "日志系统就绪 → %s (level=%s, backup=%s)", log_path, level_name, log_cfg.get("backup_count", 14)
    )


def get_logger(module: str) -> logging.Logger:
    """返回 ``newtouch.<module>`` 子 logger。import 安全（未 setup 亦有 stdout 兜底）。"""
    if not _configured:
        _ensure_stdout()
    return logging.getLogger(f"{_ROOT_NAME}.{module}")


def _log_path(cfg: Any) -> Path | None:
    log_cfg = cfg.get("logging", {}) or {}
    if not log_cfg.get("enabled", True):
        return None
    log_dir = Path(log_cfg.get("dir", "data/logs"))
    if not log_dir.is_absolute():
        log_dir = cfg.project_root / log_dir
    return log_dir / "system.log"


def read_logs(
    cfg: Any,
    level: str | None = None,
    module: str | None = None,
    limit: int = 300,
    since: str | None = None,
) -> list[dict]:
    """读取并过滤日志，返回结构化记录（newest-first）。

    - ``level``: 只保留该级别及以上（DEBUG/INFO/WARNING/ERROR）。
    - ``module``: 子串匹配 ``[module]``。
    - ``limit``: 返回最近 N 条（先按 limit*3 取尾部再过滤，防过滤后不足）。
    - ``since``: ISO 前缀，只保留此时间之后的记录。
    """
    path = _log_path(cfg)
    if path is None or not path.exists():
        return []

    level_rank = {"DEBUG": 10, "INFO": 20, "WARNING": 30, "ERROR": 40, "CRITICAL": 50}
    min_rank = level_rank.get((level or "").upper(), 0)
    mod = (module or "").strip().lower()

    # 先取尾部一批行（limit*5 余量，覆盖过滤掉的部分），再正向解析
    tail_lines: list[str] = []
    try:
        with path.open("r", encoding="utf-8", errors="replace") as f:
            all_lines = f.readlines()
    except OSError:
        return []
    tail_lines = all_lines[-(limit * 5 + 200):] if len(all_lines) > limit * 5 + 200 else all_lines

    records: list[dict] = []
    cur: dict | None = None
    for line in tail_lines:
        line = line.rstrip("\n")
        m = _LINE_RE.match(line)
        if m:
            if cur is not None:
                records.append(cur)
            ts, lvl, mod_name, msg = m.group(1), m.group(2), m.group(3), m.group(4)
            cur = {"ts": ts, "level": lvl, "module": mod_name, "msg": msg}
        else:
            # 续行（traceback / 多行消息）并入上一条
            if cur is not None:
                cur["msg"] += "\n" + line
            # 否则丢弃文件首部不完整行
    if cur is not None:
        records.append(cur)

    out: list[dict] = []
    for r in records:
        if level_rank.get(r["level"], 0) < min_rank:
            continue
        if mod and mod not in r["module"].lower():
            continue
        if since and r["ts"] < since:
            continue
        out.append(r)
    # newest-first，截断到 limit
    out.reverse()
    return out[:limit]
