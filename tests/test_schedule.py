"""测试日程/待办（v2.62）。

不依赖真实 LLM/调度循环——ScheduleStore 用临时目录隔离，Scheduler 用 mock enqueue
+ 手动调 _maybe_trigger 验证到点/勿扰/单次标done/daily重复。
"""
import asyncio
import sys
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8")

from core.config import load_config
from core.perception.schedule import (
    ScheduleStore, ScheduleItem, Scheduler, parse_trigger_at,
)
from core.tools import registry
from core.tools.schedule_tools import register_schedule_tools


def _store(tmp):
    return ScheduleStore(Path(tmp))


# ── 时间解析 ──────────────────────────────────────────────────
def test_parse_iso():
    assert parse_trigger_at("2026-06-29T19:00:00") == "2026-06-29T19:00:00"
    assert parse_trigger_at("2026-06-29 19:00") == "2026-06-29T19:00:00"
    assert parse_trigger_at("") is None
    assert parse_trigger_at("xxx") is None
    print("✅ ISO 解析 + 空串/非法返回 None")


def test_parse_natural():
    now = datetime(2026, 6, 29, 8, 0)  # 早上8点
    # 明天10点
    assert parse_trigger_at("明天10点", now) == "2026-06-30T10:00:00"
    # 后天9点
    assert parse_trigger_at("后天9点", now) == "2026-07-01T09:00:00"
    # 晚上7点（今天晚上，已过8点? 7点<8点所以是今天晚上19点 > now）
    assert parse_trigger_at("晚上7点", now) == "2026-06-29T19:00:00"
    # 今晚8点半
    assert parse_trigger_at("今晚8点半", now) == "2026-06-29T20:30:00"
    # 下午3点（15点 > 8点，今天）
    assert parse_trigger_at("下午3点", now) == "2026-06-29T15:00:00"
    # 7点（没指上下午，7点<8点已过→明天）
    assert parse_trigger_at("7点", now) == "2026-06-30T07:00:00"
    print("✅ 自然语言解析（明天/后天/晚上/下午/半/已过推明天）")


# ── ScheduleStore CRUD ────────────────────────────────────────
def test_store_crud():
    tmp = tempfile.mkdtemp(prefix="newtouch_sched_")
    store = _store(tmp)
    # add
    item = store.add("吃药", "2026-06-29T19:00:00", repeat="daily", context="7点提醒我吃药")
    assert item and item.id and item.repeat == "daily"
    assert "2026-06-29T19:00:00" == item.trigger_at
    # list
    assert len(store.list_all()) == 1
    # 自然语言 add
    item2 = store.add("开会", "明天10点")
    assert item2.trigger_at.endswith("T10:00:00")
    assert len(store.list_all()) == 2
    # update
    assert store.update(item.id, content="吃药+喝水")
    assert store.get(item.id).content == "吃药+喝水"
    assert not store.update("no-such-id", content="x")
    # mark_done
    assert store.mark_done(item.id)
    assert store.get(item.id).done is True
    assert len(store.list_all(include_done=False)) == 1  # item2 未完成
    # delete
    assert store.delete(item2.id)
    assert not store.get(item2.id)
    assert not store.delete("no-such-id")
    # 持久化：重新加载
    store2 = _store(tmp)
    assert len(store2.list_all()) == 1  # 只剩 item（已done）
    import shutil; shutil.rmtree(tmp, ignore_errors=True)
    print("✅ Store CRUD + 持久化 + include_done 过滤")


def test_invalid_repeat_falls_back():
    """非法 repeat 回退 none。"""
    tmp = tempfile.mkdtemp(prefix="newtouch_sched_r_")
    store = _store(tmp)
    item = store.add("x", "2026-06-29T19:00:00", repeat="weekly")
    assert item.repeat == "none"
    import shutil; shutil.rmtree(tmp, ignore_errors=True)
    print("✅ 非法 repeat 回退 none")


def test_mark_triggered():
    """mark_triggered：单次标 done，daily 只更 last_triggered。"""
    tmp = tempfile.mkdtemp(prefix="newtouch_sched_t_")
    store = _store(tmp)
    once = store.add("单次", "2026-06-29T19:00:00", repeat="none")
    daily = store.add("每天", "2026-06-29T07:00:00", repeat="daily")
    store.mark_triggered(once.id)
    store.mark_triggered(daily.id)
    assert store.get(once.id).done is True
    assert store.get(daily.id).done is False
    assert store.get(daily.id).last_triggered != ""
    import shutil; shutil.rmtree(tmp, ignore_errors=True)
    print("✅ mark_triggered 单次标done/daily只更last_triggered")


# ── Scheduler ─────────────────────────────────────────────────
def test_scheduler_triggers_due():
    """到点且非勿扰 → 触发 + enqueue + 单次标done。"""
    tmp = tempfile.mkdtemp(prefix="newtouch_sched_s1_")
    store = _store(tmp)
    past = (datetime.now() - timedelta(minutes=5)).strftime("%Y-%m-%dT%H:%M:%S")
    item = store.add("过期提醒", past, repeat="none")
    cfg = load_config()
    enqueued = []
    sched = Scheduler(store, lambda e: enqueued.append(e), cfg)
    now = datetime.now()
    # 非勿扰时段手动触发
    in_quiet, _ = sched._in_quiet_hours(now)
    from core.events import Event, EventType, EventPriority
    sched._maybe_trigger(store.get(item.id), now, in_quiet, None, EventType, Event, EventPriority)
    if not in_quiet:
        assert len(enqueued) == 1, "应触发一次"
        assert enqueued[0].type == EventType.SCHEDULE
        assert store.get(item.id).done is True, "单次触发后应标done"
        print("✅ 到点触发 + enqueue + 单次标done")
    else:
        print("⏭️ 当前处勿扰时段，跳过触发断言（勿扰逻辑另测）")
    import shutil; shutil.rmtree(tmp, ignore_errors=True)


def test_scheduler_quiet_defers():
    """勿扰时段 → 不触发（延后）。"""
    tmp = tempfile.mkdtemp(prefix="newtouch_sched_s2_")
    store = _store(tmp)
    past = (datetime.now() - timedelta(minutes=5)).strftime("%Y-%m-%dT%H:%M:%S")
    item = store.add("深夜提醒", past, repeat="none")
    cfg = load_config()
    enqueued = []
    sched = Scheduler(store, lambda e: enqueued.append(e), cfg)
    now = datetime.now()
    from core.events import Event, EventType, EventPriority
    # 模拟勿扰时段
    sched._maybe_trigger(store.get(item.id), now, True, now + timedelta(hours=6),
                          EventType, Event, EventPriority)
    assert len(enqueued) == 0, "勿扰时段不应触发"
    assert store.get(item.id).done is False, "勿扰未触发不应标done"
    print("✅ 勿扰时段延后不触发")
    import shutil; shutil.rmtree(tmp, ignore_errors=True)


def test_scheduler_daily_no_repeat_same_day():
    """daily 今天已触发过 → 不重复触发。"""
    tmp = tempfile.mkdtemp(prefix="newtouch_sched_s3_")
    store = _store(tmp)
    past = (datetime.now() - timedelta(minutes=5)).strftime("%Y-%m-%dT%H:%M:%S")
    item = store.add("每天吃药", past, repeat="daily")
    # 手动设 last_triggered 为今天
    it = store.get(item.id)
    it.last_triggered = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    store._save([it])
    cfg = load_config()
    enqueued = []
    sched = Scheduler(store, lambda e: enqueued.append(e), cfg)
    now = datetime.now()
    from core.events import Event, EventType, EventPriority
    in_quiet, _ = sched._in_quiet_hours(now)
    sched._maybe_trigger(store.get(item.id), now, in_quiet, None,
                          EventType, Event, EventPriority)
    assert len(enqueued) == 0, "daily 今天已触发不应重复"
    print("✅ daily 当天不重复触发")
    import shutil; shutil.rmtree(tmp, ignore_errors=True)


# ── 工具注册 + 权限 ───────────────────────────────────────────
def _cleanup_tools():
    for n in ("add_schedule", "list_schedules", "mark_done", "update_schedule", "delete_schedule"):
        if n in registry._REGISTRY:
            del registry._REGISTRY[n]


def test_tools_registered():
    tmp = tempfile.mkdtemp(prefix="newtouch_sched_tools_")
    store = _store(tmp)
    cfg = load_config()
    cfg.set("ai_permissions.allow_manage_schedules", True)
    register_schedule_tools(store, cfg)
    names = {s["name"] for s in registry.get_schemas()}
    for n in ("add_schedule", "list_schedules", "mark_done", "update_schedule", "delete_schedule"):
        assert n in names, f"{n} 应注册"
    print("✅ 5 个日程工具已注册")
    _cleanup_tools()
    import shutil; shutil.rmtree(tmp, ignore_errors=True)


def test_tools_permission_denied():
    """权限关 → 工具调用被拒。"""
    tmp = tempfile.mkdtemp(prefix="newtouch_sched_perm_")
    store = _store(tmp)
    cfg = load_config()
    cfg.set("ai_permissions.allow_manage_schedules", False)
    register_schedule_tools(store, cfg)
    out = asyncio.run(registry.call("add_schedule", content="x", trigger_at="2026-06-29T19:00:00"))
    assert "拒绝" in out or "未获授权" in out, f"应被拒，实际：{out}"
    print("✅ 权限关时工具调用被拒")
    _cleanup_tools()
    import shutil; shutil.rmtree(tmp, ignore_errors=True)


def test_tool_add_then_list():
    """工具 add + list 串通。"""
    tmp = tempfile.mkdtemp(prefix="newtouch_sched_e2e_")
    store = _store(tmp)
    cfg = load_config()
    cfg.set("ai_permissions.allow_manage_schedules", True)
    register_schedule_tools(store, cfg)
    r = asyncio.run(registry.call("add_schedule", content="吃药", trigger_at="明天10点", repeat="daily"))
    assert "OK" in r, f"add 应成功，实际：{r}"
    r2 = asyncio.run(registry.call("list_schedules"))
    assert "吃药" in r2, f"list 应含刚加的，实际：{r2}"
    print("✅ 工具 add+list 串通")
    _cleanup_tools()
    import shutil; shutil.rmtree(tmp, ignore_errors=True)


# ── catalog ───────────────────────────────────────────────────
def test_catalog_has_schedule_tools():
    from core.tools.catalog import TOOL_CATALOG
    names = {t["name"] for t in TOOL_CATALOG}
    for n in ("add_schedule", "list_schedules", "mark_done", "update_schedule", "delete_schedule"):
        assert n in names, f"catalog 应含 {n}"
    print("✅ catalog 含 5 个日程工具")


if __name__ == "__main__":
    print("=== 测试日程/待办 ===\n")
    test_parse_iso()
    test_parse_natural()
    test_store_crud()
    test_invalid_repeat_falls_back()
    test_mark_triggered()
    test_scheduler_triggers_due()
    test_scheduler_quiet_defers()
    test_scheduler_daily_no_repeat_same_day()
    test_tools_registered()
    test_tools_permission_denied()
    test_tool_add_then_list()
    test_catalog_has_schedule_tools()
    _cleanup_tools()
    print("\n=== 全部测试通过 ===")
