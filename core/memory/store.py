"""记忆存储 (架构文档 5.9 / 7.3)。mem0 本地接入。

每条记忆带"当时的情绪快照"(metadata)，recall 时一起返回 → ta "回忆起那时的情绪"。
按角色名隔离 (user_id=角色名)。
记忆全在本地 (SQLite/向量库)，只有 recall 命中的少量片段进 prompt。

注意: mem0 默认用 OpenAI embedding。可在 config.memory 配置 provider/base_url/key。
memory.enabled=false 或 mem0 不可用时，降级为空操作（不报错、不阻塞）。

要点抽取的三种模式（按 config 决定，互斥优先级 infer > extract > 原始摘要）：
- memory.infer=true：用 mem0 原生抽取（内部用 response_format=json_object，需 LLM 支持 JSON 模式；
  火山 coding 端点的 ark-code-latest 默认调度模型不支持，需控制台绑定支持 JSON 的模型）。
- memory.extract=true（默认）：store 自己用「续写模式」调 LLM 抽取要点——预填充 assistant 的 `{`
  引导模型续写 JSON，不依赖 response_format，火山 coding 端点直接可用、零额外消耗。
- 两者都关：直接存 orchestrator 拼好的原始对话摘要，不调 LLM。
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime

from ..config import Config
from ..logger import get_logger

log = get_logger("memory")


def _rel_time_desc(ts_str: str, now: datetime | None = None) -> str:
    """把 ISO 时间戳转成"记忆有多久"的口语描述，用于 recall 注入。

    粒度按天/小时（长期记忆通常跨天）。无 ts/解析失败/不足1小时返回空串。
    """
    if not ts_str:
        return ""
    try:
        dt = datetime.fromisoformat(ts_str)
    except (ValueError, TypeError):
        return ""
    delta = ((now or datetime.now()) - dt).total_seconds()
    if delta < 3600:
        return ""
    if delta < 86400:
        return f"约{int(delta // 3600)}小时前"
    d = int(delta // 86400)
    if d < 30:
        return f"约{d}天前"
    if d < 365:
        return f"约{d // 30}个月前"
    return f"约{d // 365}年前"


@dataclass
class MemoryEntry:
    content: str
    emotion_at_that_time: dict
    timestamp: str
    tags: list[str]


class MemoryStore:
    def __init__(self, config: Config):
        self._cfg = config
        self._enabled = config.get("memory.enabled", False)
        self._agent = config.get("character.name", "默认")
        self._mem = None
        self._extract_client = None  # 续写抽取用的同步 OpenAI client，懒加载
        if self._enabled:
            self._init_mem0()

    def _init_mem0(self) -> None:
        try:
            from mem0 import Memory
        except ImportError:
            log.warning("mem0 未安装，记忆功能禁用")
            self._enabled = False
            return
        try:
            db_path = str(self._cfg.project_root / self._cfg.get("memory.db_path", "data/memory_db"))
            # 最小本地配置：向量库用 chroma 本地持久化
            mem_cfg = {
                "vector_store": {
                    "provider": "chroma",
                    "config": {"path": db_path},
                },
            }
            # LLM/embedder: 复用主 LLM 的 key/base_url（需 OpenAI 兼容）
            llm_key = self._cfg.get("memory.api_key") or self._cfg.get("modules.llm.api_key")
            llm_base = self._cfg.get("memory.base_url") or self._cfg.get("modules.llm.base_url")
            if llm_key:
                mem_cfg["llm"] = {"provider": "openai", "config": {
                    "model": self._cfg.get("memory.llm_model", "gpt-4o-mini"),
                    "api_key": llm_key,
                    "openai_base_url": llm_base,
                }}
                mem_cfg["embedder"] = {"provider": "openai", "config": {
                    "model": self._cfg.get("memory.embedding_model", "text-embedding-3-small"),
                    "api_key": self._cfg.get("memory.embed_api_key") or llm_key,
                    "openai_base_url": self._cfg.get("memory.embed_base_url") or llm_base,
                }}
            self._mem = Memory.from_config(mem_cfg)
            log.info("mem0 已就绪")
        except Exception as e:  # noqa: BLE001
            log.error("mem0 初始化失败，记忆禁用: %s", e)
            self._enabled = False

    def _get_extract_client(self):
        """续写抽取用的同步 OpenAI 兼容 client（复用记忆/主脑的 key+base_url）。"""
        if self._extract_client is not None:
            return self._extract_client
        try:
            from openai import OpenAI
        except ImportError:
            return None
        key = self._cfg.get("memory.api_key") or self._cfg.get("modules.llm.api_key")
        base = self._cfg.get("memory.base_url") or self._cfg.get("modules.llm.base_url") or None
        if not key:
            return None
        self._extract_client = OpenAI(api_key=key, base_url=base)
        return self._extract_client

    def _extract_facts(self, content: str) -> list[str]:
        """用续写模式让 LLM 把一轮对话抽成要点列表。失败则返回 [原始 content] 兜底。

        预填充 assistant 的 `{` 引导模型续写 JSON，不依赖 response_format——
        火山 coding 端点的 ark-code-latest 直接可用（实测）。
        """
        client = self._get_extract_client()
        if client is None:
            return [content]
        model = self._cfg.get("memory.llm_model") or self._cfg.get("modules.llm.model", "ark-code-latest")
        prompt = (
            "从下面这轮对话里提取值得长期记住的事实（对方的偏好/习惯/重要信息/计划/承诺、"
            "以及我对ta说过的关键内容等），忽略纯寒暄和无信息量的话。"
            "用JSON返回，格式 {\"facts\": [\"一句完整中文事实\", ...]}，"
            "每条事实独立、自包含、不超过30字。没有值得记的就返回 {\"facts\": []}。\n\n"
            f"对话：\n{content}"
        )
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": "{"},  # 预填充，引导续写 JSON
                ],
                max_tokens=400,
                temperature=0.3,
            )
            raw = "{" + (resp.choices[0].message.content or "")
            facts = self._parse_facts(raw)
            return facts if facts else []  # 空列表=本轮没值得记的，不兜底存摘要
        except Exception as e:  # noqa: BLE001
            log.warning("抽取失败，降级存原文: %s", e)
            return [content]

    @staticmethod
    def _parse_facts(raw: str) -> list[str]:
        """从（可能不完整的）JSON 文本里抠出 facts 列表，容错。"""
        try:
            obj = json.loads(raw)
            facts = obj.get("facts", []) if isinstance(obj, dict) else []
        except (json.JSONDecodeError, ValueError):
            # 续写被 max_tokens 截断等情况：正则兜底抠字符串数组元素
            m = re.search(r'"facts"\s*:\s*\[(.*?)\]', raw, re.S)
            facts = re.findall(r'"([^"]+)"', m.group(1)) if m else []
        return [f.strip() for f in facts if isinstance(f, str) and f.strip()]

    def add(self, content: str, emotion_snapshot: dict, tags: list[str]) -> None:
        if not self._enabled or not self._mem:
            return
        try:
            # 抽取模式决定存什么（见模块 docstring）
            infer = self._cfg.get("memory.infer", False)
            extract = self._cfg.get("memory.extract", True)
            if infer:
                contents = [content]  # mem0 自己抽取
            elif extract:
                contents = self._extract_facts(content)  # 续写抽取的要点列表
            else:
                contents = [content]  # 原始摘要
            if not contents:
                return  # 抽取判定本轮无可记内容
            # chroma metadata 只接受标量(str/int/float/bool/list/None)，不接受嵌套 dict。
            # 把情绪各维拍平成 emo_* 标量、tags 存逗号串；recall 时按此格式还原。
            meta = {f"emo_{k}": v for k, v in (emotion_snapshot or {}).items()
                    if isinstance(v, (int, float, str, bool))}
            meta["tags"] = ",".join(tags) if tags else ""
            meta["ts"] = datetime.now().isoformat(timespec="seconds")  # 记忆形成时间，recall 时还原"多久前"
            for c in contents:
                self._mem.add(c, user_id=self._agent, infer=infer, metadata=meta)
        except Exception as e:  # noqa: BLE001
            log.error("add 失败: %s", e)

    def delete_user(self, name: str) -> None:
        """删除某角色（user_id=目录名）的全部 mem0 记忆。用于删角色时清理。"""
        if not self._enabled or not self._mem:
            return
        try:
            self._mem.delete_all(user_id=name)
            log.info("已清理角色记忆: %s", name)
        except Exception as e:  # noqa: BLE001
            log.error("delete_user 失败: %s", e)

    def generate_recall_queries(self, user_name: str, context: str = "") -> list[str]:
        """用 LLM 生成精准的记忆检索 query（主动回想路线 B）。

        Args:
            user_name: 用户名字
            context: 当前情境（可选），如"心跳·孤独感高"、"刚看到用户在加班"等

        Returns:
            1-3 个检索问题的列表，失败返回空列表（调用方兜底用默认 query）
        """
        if not self._enabled:
            return []

        client = self._get_extract_client()
        if client is None:
            return []

        model = self._cfg.get("memory.llm_model") or self._cfg.get("modules.llm.model", "ark-code-latest")

        ctx_hint = f"（当前情境：{context}）" if context else ""
        prompt = f"""
你想主动找 {user_name} 说点什么。
在开口前，你想先回忆一些关于 ta 的事情。{ctx_hint}

生成 1-3 个检索问题（每行一个），用于在你的记忆中搜索相关内容。
问题应该具体、有针对性，能帮助你回忆起有用的信息。

示例：
- {user_name}最近在忙什么工作？
- {user_name}有什么兴趣爱好？
- 我们上次聊天说了什么？
- {user_name}最近有什么烦恼或困扰吗？

只输出问题列表，每行一个，不要编号，不要其他解释：
""".strip()

        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=150,
                temperature=0.7,
            )
            raw = (resp.choices[0].message.content or "").strip()
            if not raw:
                return []

            # 解析输出：每行一个问题，过滤空行和编号
            lines = raw.split("\n")
            queries = []
            for line in lines:
                line = line.strip()
                # 移除可能的编号（1. / - / • 等）
                line = re.sub(r"^[\d\-•\*]+[\.\)]\s*", "", line)
                if line and len(line) > 5:  # 至少 5 个字符才算有效问题
                    queries.append(line)

            return queries[:3]  # 最多返回 3 个

        except Exception as e:  # noqa: BLE001
            log.warning("生成 recall query 失败: %s", e)
            return []

    def recall(self, query: str, limit: int = 3) -> list[str]:
        """语义检索，返回拼好的记忆字符串列表（带情绪标签）。"""
        if not self._enabled or not self._mem:
            return []
        try:
            # mem0 2.x: search 用 filters={'user_id':...} + top_k，不再接受 user_id=/limit=
            res = self._mem.search(query, filters={"user_id": self._agent}, top_k=limit)
            items = res.get("results", res) if isinstance(res, dict) else res
            out = []
            for it in items:
                text = it.get("memory") or it.get("text") or ""
                meta = it.get("metadata") or {}
                # add 时把情绪拍平成 emo_* 标量（见 add）
                val = meta.get("emo_valence", 0) or 0
                mood = "(当时心情不错)" if val > 0.2 else "(当时有点低落)" if val < -0.2 else ""
                when = _rel_time_desc(meta.get("ts", ""))
                tag = " ".join(t for t in (when and f"({when})", mood) if t)
                if tag:
                    text = f"{text} {tag}".strip()
                if text:
                    out.append(text)
            return out
        except Exception as e:  # noqa: BLE001
            log.error("recall 失败: %s", e)
            return []

    def recall_multi_query(self, queries: list[str], limit_per_query: int = 2) -> list[str]:
        """多个 query 并行检索，合并去重后返回。

        Args:
            queries: 检索问题列表
            limit_per_query: 每个问题最多返回几条记忆

        Returns:
            合并去重后的记忆列表
        """
        if not queries:
            return []

        all_memories = []
        seen = set()

        for q in queries:
            memories = self.recall(q, limit=limit_per_query)
            for mem in memories:
                # 简单去重：用记忆文本的前 50 字符作为指纹
                fingerprint = mem[:50]
                if fingerprint not in seen:
                    seen.add(fingerprint)
                    all_memories.append(mem)

        return all_memories
