"""测试记忆检索 query 生成（Token 密集型优化 #4）。

测试 MemoryStore.generate_recall_queries() 和 recall_multi_query()。
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')  # Windows 控制台 UTF-8 输出

from core.config import load_config
from core.memory.store import MemoryStore


def test_generate_queries_basic():
    """测试基础 query 生成（需要真实 LLM API）。"""
    cfg = load_config()
    if not cfg.get("memory.enabled", False):
        print("⏭️  memory.enabled=false，跳过测试")
        return

    store = MemoryStore(cfg)

    # 测试生成 query
    queries = store.generate_recall_queries("小明", context="心跳·孤独感0.65")

    print(f"生成的 queries: {queries}")

    # 验证
    assert isinstance(queries, list), "应该返回列表"
    # LLM 可能生成 0-3 个 query，都是合理的
    assert len(queries) <= 3, "最多 3 个 query"

    if queries:
        for q in queries:
            assert isinstance(q, str), "每个 query 应该是字符串"
            assert len(q) > 5, "query 应该有实际内容"
            print(f"  ✅ 生成的 query: {q}")
    else:
        print("  ⚠️  LLM 未生成 query（可能是模型/配置问题）")


def test_generate_queries_disabled():
    """测试记忆禁用时的降级行为。"""
    cfg = load_config()
    # 临时禁用记忆
    original = cfg.get("memory.enabled", False)
    cfg._data["memory"] = cfg._data.get("memory", {})
    cfg._data["memory"]["enabled"] = False

    store = MemoryStore(cfg)
    queries = store.generate_recall_queries("用户")

    # 恢复原值
    cfg._data["memory"]["enabled"] = original

    # 禁用时应该返回空列表
    assert queries == [], "禁用时应该返回空列表"
    print("✅ 禁用时返回空列表")


def test_recall_multi_query():
    """测试多 query 检索和去重。"""
    cfg = load_config()
    if not cfg.get("memory.enabled", False):
        print("⏭️  memory.enabled=false，跳过测试")
        return

    store = MemoryStore(cfg)

    # 先添加几条测试记忆
    store.add("用户喜欢喝咖啡", {"valence": 0.3}, ["偏好"])
    store.add("用户在做一个重要项目", {"valence": 0.0}, ["工作"])
    store.add("用户周末喜欢爬山", {"valence": 0.5}, ["爱好"])

    # 测试多 query 检索
    queries = [
        "用户有什么兴趣爱好？",
        "用户最近在忙什么？",
        "用户喜欢什么饮料？",
    ]

    memories = store.recall_multi_query(queries, limit_per_query=2)

    print(f"检索到 {len(memories)} 条记忆:")
    for mem in memories:
        print(f"  - {mem}")

    # 验证
    assert isinstance(memories, list), "应该返回列表"
    # 应该有一些记忆被召回（如果向量库工作正常）
    if memories:
        for mem in memories:
            assert isinstance(mem, str), "每条记忆应该是字符串"
        print(f"✅ 成功检索到 {len(memories)} 条记忆")
    else:
        print("⚠️  未检索到记忆（可能是向量库刚初始化）")


def test_recall_multi_query_empty():
    """测试空 query 列表。"""
    cfg = load_config()
    store = MemoryStore(cfg)

    memories = store.recall_multi_query([])
    assert memories == [], "空 query 列表应该返回空列表"
    print("✅ 空 query 列表返回空列表")


def test_recall_multi_query_deduplication():
    """测试去重逻辑（模拟）。"""
    cfg = load_config()
    store = MemoryStore(cfg)

    # 模拟：相同前缀的记忆应该被去重
    # 实际测试依赖向量库返回结果，这里只测试去重函数本身

    # 测试至少不会崩溃
    queries = ["测试问题1", "测试问题2"]
    memories = store.recall_multi_query(queries, limit_per_query=1)
    assert isinstance(memories, list), "应该返回列表"
    print("✅ 去重逻辑不崩溃")


if __name__ == "__main__":
    print("=== 测试记忆检索 query 生成 ===\n")

    print("1. 测试基础 query 生成（需要 LLM API）")
    test_generate_queries_basic()
    print()

    print("2. 测试禁用时的降级")
    test_generate_queries_disabled()
    print()

    print("3. 测试多 query 检索")
    test_recall_multi_query()
    print()

    print("4. 测试空 query 列表")
    test_recall_multi_query_empty()
    print()

    print("5. 测试去重逻辑")
    test_recall_multi_query_deduplication()
    print()

    print("=== 全部测试完成 ===")
