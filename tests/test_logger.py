"""core/logger.py 功能测试（独立脚本，无需 pytest）。

覆盖：
  1. get_logger 返回子 logger，setup_logging 写入文件
  2. 日志行格式可被 read_logs 解析为结构化记录
  3. level 过滤（只保留该级别及以上）
  4. module 子串过滤
  5. limit 截断（newest-first）
  6. 多行续行（traceback）并入上一条 msg
  7. logging.enabled=false 时不写文件（read_logs 返回空）
  8. setup_logging 幂等（重复调用不重复加 file handler）

运行: cd D:\\ai\\newTouch && python tests/test_logger.py
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.config import Config
from core.logger import setup_logging, get_logger, read_logs


def _make_cfg(tmp: Path, *, enabled=True, level="INFO"):
    raw = {"logging": {"enabled": enabled, "level": level, "dir": "data/logs", "backup_count": 3}}

    class _PatchedConfig(Config):
        @property
        def project_root(self):
            return tmp

    return _PatchedConfig(raw)


def _reset_root():
    """每个用例前清掉 newtouch root 的 handler，避免上一例残留。"""
    import logging
    root = logging.getLogger("newtouch")
    for h in list(root.handlers):
        root.removeHandler(h)
    # 重置 _configured 标志，让 setup_logging 重新走完整流程
    import core.logger as L
    L._configured = False


def test_write_and_parse():
    _reset_root()
    tmp = Path(tempfile.mkdtemp())
    cfg = _make_cfg(tmp)
    setup_logging(cfg)
    log = get_logger("orch")
    log.info("已切换到角色: 爱丽丝")
    log.error("look_now 失败: 超时")
    recs = read_logs(cfg, limit=50)
    msgs = [r["msg"] for r in recs]
    assert any("已切换到角色" in m for m in msgs), f"info 未记录: {msgs}"
    assert any("look_now 失败" in m for m in msgs), f"error 未记录: {msgs}"
    # 字段齐全
    r = recs[0]
    assert set(r.keys()) >= {"ts", "level", "module", "msg"}, r
    print("  ✓ 写入 + 解析")


def test_level_filter():
    _reset_root()
    tmp = Path(tempfile.mkdtemp())
    cfg = _make_cfg(tmp, level="DEBUG")
    setup_logging(cfg)
    log = get_logger("memory")
    log.debug("调试细节")
    log.info("普通信息")
    log.warning("警告")
    log.error("错误")
    # level=WARNING → 只保留 WARNING/ERROR
    recs = read_logs(cfg, level="WARNING", limit=50)
    levels = {r["level"] for r in recs}
    assert levels <= {"WARNING", "ERROR"}, levels
    assert "INFO" not in levels and "DEBUG" not in levels
    print("  ✓ level 过滤")


def test_module_filter():
    _reset_root()
    tmp = Path(tempfile.mkdtemp())
    cfg = _make_cfg(tmp)
    setup_logging(cfg)
    get_logger("orch").info("orch 消息")
    get_logger("memory").info("memory 消息")
    recs = read_logs(cfg, module="mem", limit=50)
    assert all("mem" in r["module"].lower() for r in recs), recs
    assert any(r["module"] == "memory" for r in recs)
    print("  ✓ module 过滤")


def test_limit_newest_first():
    _reset_root()
    tmp = Path(tempfile.mkdtemp())
    cfg = _make_cfg(tmp)
    setup_logging(cfg)
    log = get_logger("stt")
    for i in range(10):
        log.info("line %d", i)
    recs = read_logs(cfg, limit=3)
    assert len(recs) == 3, len(recs)
    # newest-first：最后写的 line 9 在最前
    assert "line 9" in recs[0]["msg"], recs[0]["msg"]
    print("  ✓ limit 截断 + newest-first")


def test_multiline_continuation():
    _reset_root()
    tmp = Path(tempfile.mkdtemp())
    cfg = _make_cfg(tmp)
    setup_logging(cfg)
    log = get_logger("vision")
    log.error("VLM 调用失败: %s", "Traceback (most recent call last):\n  File x\nValueError: boom")
    recs = read_logs(cfg, limit=50)
    err = [r for r in recs if "VLM" in r["msg"]][0]
    assert "Traceback" in err["msg"], err["msg"]
    assert "ValueError: boom" in err["msg"], err["msg"]
    print("  ✓ 多行续行并入上一条")


def test_disabled_no_file():
    _reset_root()
    tmp = Path(tempfile.mkdtemp())
    cfg = _make_cfg(tmp, enabled=False)
    setup_logging(cfg)
    get_logger("orch").error("不该进文件")
    assert read_logs(cfg, limit=50) == [], "禁用时 read_logs 应返回空"
    print("  ✓ enabled=false 不写文件")


def test_setup_idempotent():
    _reset_root()
    tmp = Path(tempfile.mkdtemp())
    cfg = _make_cfg(tmp)
    setup_logging(cfg)
    setup_logging(cfg)
    setup_logging(cfg)
    import logging
    from logging.handlers import TimedRotatingFileHandler
    root = logging.getLogger("newtouch")
    fh_count = sum(1 for h in root.handlers if isinstance(h, TimedRotatingFileHandler))
    assert fh_count == 1, f"重复 setup 应只有 1 个 file handler，实际 {fh_count}"
    print("  ✓ setup 幂等")


def main():
    tests = [
        test_write_and_parse, test_level_filter, test_module_filter,
        test_limit_newest_first, test_multiline_continuation,
        test_disabled_no_file, test_setup_idempotent,
    ]
    for t in tests:
        t()
    print(f"\n{len(tests)}/{len(tests)} 通过 ✅")


if __name__ == "__main__":
    main()
