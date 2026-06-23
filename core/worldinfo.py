"""世界书 (World Info / Lorebook) — 复刻 SillyTavern 激活算法 (架构文档 6.2/6.3)。

按需注入的知识条目：只在相关关键词出现时才激活，平时不占 context。
存储: data/worldbooks/{name}.json，结构 {name, entries: [...]}。
兼容导入 ST 的 json（字段映射 keys→key, secondary_keys→keysecondary, insertion_order→order）。
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path


# selectiveLogic 枚举
AND_ANY = "AND_ANY"    # 主key命中后, 任一次key命中即通过
AND_ALL = "AND_ALL"    # 全部次key命中才通过
NOT_ANY = "NOT_ANY"    # 任一次key命中则否决
NOT_ALL = "NOT_ALL"    # 全部次key命中才否决


@dataclass
class WorldInfoEntry:
    uid: int = 0
    key: list[str] = field(default_factory=list)          # 主关键词
    keysecondary: list[str] = field(default_factory=list) # 次关键词
    content: str = ""
    constant: bool = False        # 常驻: 跳过关键词匹配直接激活
    selective: bool = False       # 启用次关键词二次判断
    selectiveLogic: str = AND_ANY
    order: int = 100              # 注入排序
    position: str = "after"       # before | after | depth
    depth: int = 4               # position=depth 时插入对话历史的深度
    probability: int = 100        # 命中后激活概率 0-100
    sticky: int = 0              # 激活后强制保持的轮数
    cooldown: int = 0            # 激活后冷却轮数
    delay: int = 0               # 会话前 N 轮禁止激活
    disable: bool = False

    # 运行时状态 (不持久化)
    _sticky_left: int = 0
    _cooldown_left: int = 0

    @classmethod
    def from_dict(cls, d: dict) -> "WorldInfoEntry":
        # 兼容 ST 字段名
        m = dict(d)
        if "keys" in m and "key" not in m:
            m["key"] = m.pop("keys")
        if "secondary_keys" in m and "keysecondary" not in m:
            m["keysecondary"] = m.pop("secondary_keys")
        if "insertion_order" in m and "order" not in m:
            m["order"] = m.pop("insertion_order")
        known = {f for f in cls.__dataclass_fields__ if not f.startswith("_")}
        return cls(**{k: v for k, v in m.items() if k in known})


class WorldBook:
    def __init__(self, entries: list[WorldInfoEntry], name: str = ""):
        self.name = name
        self.entries = entries

    @classmethod
    def load(cls, path: Path) -> "WorldBook":
        data = json.loads(path.read_text(encoding="utf-8"))
        raw = data.get("entries", [])
        # ST 用 dict{uid: entry}，我们也兼容 list
        items = raw.values() if isinstance(raw, dict) else raw
        entries = [WorldInfoEntry.from_dict(e) for e in items]
        return cls(entries, name=data.get("name", path.stem))


def _match_keys(text: str, keys: list[str], case_sensitive: bool = False) -> bool:
    """任一 key 出现在 text 中即命中。"""
    hay = text if case_sensitive else text.lower()
    for k in keys:
        if not k:
            continue
        needle = k if case_sensitive else k.lower()
        if needle in hay:
            return True
    return False


def _secondary_ok(text: str, entry: WorldInfoEntry) -> bool:
    """根据 selectiveLogic 判断次关键词。"""
    if not entry.selective or not entry.keysecondary:
        return True
    hay = text.lower()
    hits = [k for k in entry.keysecondary if k and k.lower() in hay]
    logic = entry.selectiveLogic
    if logic == AND_ANY:
        return len(hits) > 0
    if logic == AND_ALL:
        return len(hits) == len([k for k in entry.keysecondary if k])
    if logic == NOT_ANY:
        return len(hits) == 0
    if logic == NOT_ALL:
        return len(hits) < len([k for k in entry.keysecondary if k])
    return True


# 简单的 token 估算 (中文按字符, 够预算控制用)
def _est_tokens(s: str) -> int:
    return len(s)


def activate(
    book: WorldBook,
    scan_text: str,
    budget_chars: int = 2000,
    max_recursion: int = 3,
    round_no: int = 0,
) -> dict:
    """世界书激活, 返回按位置分组的内容。

    参数:
      scan_text   — 扫描缓冲区(最近对话 + 角色描述 + 用户人设 + scenario)
      budget_chars— 内容总字符预算
      round_no    — 当前对话轮次(用于 delay/sticky/cooldown 时序)
    返回:
      {"before": [...], "after": [...], "depth": [(depth, content), ...]}
    """
    activated: list[WorldInfoEntry] = []
    buffer = scan_text
    used = 0

    # 先处理时序状态递减
    for e in book.entries:
        if e._cooldown_left > 0:
            e._cooldown_left -= 1

    def _try_activate(entry: WorldInfoEntry, is_recursion: bool) -> bool:
        nonlocal used
        if entry.disable or entry in activated:
            return False
        # delay: 会话前 N 轮禁止
        if entry.delay and round_no < entry.delay:
            return False
        # cooldown
        if entry._cooldown_left > 0 and entry._sticky_left <= 0:
            return False
        # sticky 生效中 → 强制激活(跳过关键词)
        forced = entry._sticky_left > 0
        if not forced and not entry.constant:
            if not _match_keys(buffer, entry.key):
                return False
            if not _secondary_ok(buffer, entry):
                return False
        # 概率
        if not forced and entry.probability < 100:
            # 用 uid+round 做确定性伪随机(避免 Math.random, 可复现)
            seed = (entry.uid * 2654435761 + round_no * 40503) % 100
            if seed >= entry.probability:
                return False
        # 预算
        cost = _est_tokens(entry.content)
        if used + cost > budget_chars:
            return False
        used += cost
        activated.append(entry)
        # 设置时序状态
        if entry.sticky and entry._sticky_left <= 0:
            entry._sticky_left = entry.sticky
        elif entry._sticky_left > 0:
            entry._sticky_left -= 1
        if entry.cooldown:
            entry._cooldown_left = entry.cooldown
        return True

    # 初始扫描
    for e in book.entries:
        _try_activate(e, is_recursion=False)

    # 递归: 已激活内容加入缓冲, 再扫
    for _ in range(max_recursion):
        new_text = "\n".join(e.content for e in activated)
        buffer = scan_text + "\n" + new_text
        before = len(activated)
        for e in book.entries:
            _try_activate(e, is_recursion=True)
        if len(activated) == before:
            break

    # 分组输出, 按 order 排序
    activated.sort(key=lambda e: e.order)
    result: dict = {"before": [], "after": [], "depth": []}
    for e in activated:
        if e.position == "before":
            result["before"].append(e.content)
        elif e.position == "depth":
            result["depth"].append((e.depth, e.content))
        else:
            result["after"].append(e.content)
    return result
