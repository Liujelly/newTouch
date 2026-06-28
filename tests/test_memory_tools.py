"""测试记忆检索工具 memory_search（v2.49）。

验证：
1. register_memory_tools 把 memory_search 注册进 registry
2. registry.call("memory_search", ...) 能召回已存的记忆
3. 开关关闭（tool_enabled=false / store 未启用 / store=None）时不注册
4. 开关 A（memory.reactive_auto_recall）配置层默认值正确

不依赖真实 LLM API（add 用 extract 关闭走原始摘要直存，recall 是本地向量检索）。
用临时 db_path 隔离，不污染真实记忆库。
"""
import asyncio
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8")

from core.config import load_config
from core.memory.store import MemoryStore
from core.tools import registry
from core.tools.memory_tools import register_memory_tools


def _fresh_store(db_path: str, enabled: bool = True):
    """构造一个用临时库的 MemoryStore（不碰真实记忆库）。"""
    cfg = load_config()
    cfg.set("memory.db_path", db_path)
    cfg.set("character.name", "memtool_test_user")
    cfg.set("memory.enabled", enabled)
    cfg.set("memory.infer", False)
    cfg.set("memory.extract", False)  # 关抽取，add 直接存原文，不调 LLM
    return cfg, MemoryStore(cfg)


def _cleanup_registry():
    """测试后从全局 registry 删掉 memory_search，避免污染其他测试。"""
    if "memory_search" in registry._REGISTRY:
        del registry._REGISTRY["memory_search"]


def test_register_and_recall():
    """注册工具后，registry.call 能召回已存记忆。"""
    tmp = tempfile.mkdtemp(prefix="newtouch_memtool_")
    cfg, store = _fresh_store(tmp)
    if not store._enabled:
        print("⏭️  记忆未启用（mem0/embedding 未配置），跳过召回测试")
        _cleanup_registry()
        return

    store.add("用户非常讨厌吃香菜，家里养了只橘猫叫团子。",
              {"valence": 0.5}, tags=["偏好"])
    import time
    time.sleep(2)  # 等 mem0 落库

    register_memory_tools(store, cfg)
    schemas = {s["name"] for s in registry.get_schemas()}
    assert "memory_search" in schemas, "memory_search 应已注册"
    print("✅ memory_search 已注册进 registry")

    result = asyncio.run(registry.call("memory_search", query="忌口"))
    print(f"  召回结果: {result}")
    assert "香菜" in result, f"应召回香菜相关记忆，实际：{result}"

    # 空 query
    result_empty = asyncio.run(registry.call("memory_search", query=""))
    assert "关键词" in result_empty, f"空 query 应提示，实际：{result_empty}"
    print("✅ 空 query 处理正确")

    import shutil
    shutil.rmtree(tmp, ignore_errors=True)
    _cleanup_registry()


def test_not_registered_when_disabled():
    """开关 B 关闭 / store 未启用 / store=None 时不注册。"""
    # 1) tools.memory_search=false（v2.58 新开关）
    tmp = tempfile.mkdtemp(prefix="newtouch_memtool_d_")
    cfg, store = _fresh_store(tmp)
    cfg.set("tools.memory_search", False)
    register_memory_tools(store, cfg)
    schemas = {s["name"] for s in registry.get_schemas()}
    assert "memory_search" not in schemas, "tools.memory_search=false 不应注册"
    print("✅ tools.memory_search=false 时不注册")

    # 1b) 旧开关 memory.tool_enabled=false 仍生效（向后兼容回退）
    cfg.set("tools.memory_search", None)  # 清掉新开关，走回退
    cfg.set("memory.tool_enabled", False)
    register_memory_tools(store, cfg)
    schemas = {s["name"] for s in registry.get_schemas()}
    assert "memory_search" not in schemas, "旧 memory.tool_enabled=false 应回退生效"
    print("✅ 旧 memory.tool_enabled=false 兼容回退生效")

    # 2) store 未启用
    cfg2, store2 = _fresh_store(tempfile.mkdtemp(), enabled=False)
    register_memory_tools(store2, cfg2)
    schemas = {s["name"] for s in registry.get_schemas()}
    assert "memory_search" not in schemas, "store 未启用不应注册"
    print("✅ store 未启用时不注册")

    # 3) store=None
    register_memory_tools(None, cfg)
    schemas = {s["name"] for s in registry.get_schemas()}
    assert "memory_search" not in schemas, "store=None 不应注册"
    print("✅ store=None 时不注册")

    import shutil
    shutil.rmtree(tmp, ignore_errors=True)
    _cleanup_registry()


def test_switch_a_config():
    """开关 A（reactive_auto_recall）配置层默认值 + 可设 False。"""
    cfg = load_config()
    # 默认 True（即使 config.yaml 没写也默认开）
    assert cfg.get("memory.reactive_auto_recall", True) is True, "默认应为 True"
    cfg.set("memory.reactive_auto_recall", False)
    assert cfg.get("memory.reactive_auto_recall", True) is False, "应能设为 False"
    print("✅ reactive_auto_recall 默认 True、可设 False")


if __name__ == "__main__":
    print("=== 测试记忆检索工具 memory_search ===\n")
    test_register_and_recall()
    test_not_registered_when_disabled()
    test_switch_a_config()
    print("\n=== 全部测试通过 ===")
