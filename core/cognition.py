"""认知引擎 (架构文档 5.7)。支持 anthropic / openai 兼容接口 (DeepSeek 等)。

阶段1: 反应路径，流式输出文本。
切换 provider 只需改 config.yaml 的 modules.llm 节，代码不动。
"""
from __future__ import annotations

import json
import re
from typing import AsyncIterator

from .character import (
    CharacterCard,
    build_proactive_prompt,
    build_reactive_prompt,
)
from .config import Config
from .tools import registry
from .logger import get_logger

log = get_logger("cognition")


def _parse_emotion_delta(raw) -> dict:
    """解析 LLM 输出的情绪增量 dict，只保留五维已知键 + 数值，钳到 [-1,1]。

    容错：非 dict、非数值、未知键一律丢弃，返回干净的 {维度: float}。
    """
    if not isinstance(raw, dict):
        return {}
    _KEYS = ("valence", "arousal", "attachment", "worry", "loneliness")
    out = {}
    for k in _KEYS:
        v = raw.get(k)
        if isinstance(v, bool):  # bool 是 int 子类，先排除
            continue
        if isinstance(v, (int, float)):
            out[k] = max(-1.0, min(1.0, float(v)))
    return out


# 决策标记：括号支持半/全角 [ ［ 【，关键词覆盖简体/繁体/日文汉字。
# 模型在非中文回复语言下偶尔把标记写成日文汉字（決定/沈黙/開口/觀察），必须一并识别，
# 否则标记识别不到、整段（往往只有标记本身）被当成 thought 落进意识流（见 problem 2）。
_BRK_OPEN = r"[\[［【]"
_BRK_CLOSE = r"[\]］】]"
_ACTION_PATTERNS = {
    "speak": rf"{_BRK_OPEN}\s*[决決]定\s*[：:]\s*(?:开口|開口|说|說|发言|發言){_BRK_CLOSE}",
    "silent": rf"{_BRK_OPEN}\s*[决決]定\s*[：:]\s*(?:沉默|沈黙|不说|不說|保持安静|保持安靜){_BRK_CLOSE}",
    "look": rf"{_BRK_OPEN}\s*[决決]定\s*[：:]\s*(?:看看|看一眼|观察|觀察)(?:他|她|ta)?{_BRK_CLOSE}",
    "retry_later": rf"{_BRK_OPEN}\s*[决決]定\s*[：:]\s*(?:稍后|稍後|晚点|晚點|改天|再试|再試)(?:再说|再說|提醒|提)?{_BRK_CLOSE}",
}
# retry_later 的延后分钟数：决策标记后跟「(10分钟)」「10min」「10分钟后」等
_RETRY_MINUTES_RE = re.compile(r"(\d{1,3})\s*(?:分钟|分鐘|min|分钟後|分钟后)")
# 兜底：任意形态的决策标记（用于把残留标记从 thought 里清掉，含其后紧跟的翻译括号）。
_ANY_DECISION_MARKER = re.compile(
    rf"{_BRK_OPEN}\s*[决決]定\s*[：:][^\]］】]*{_BRK_CLOSE}\s*(?:[（(][^）)]*[）)])?"
)


def _strip_decision_markers(text: str) -> str:
    """清掉文本里残留的决策标记（含其后的翻译括号），用于净化 thought。

    模型若只吐一个光秃秃的标记、前面没有独白，净化后 thought 为空（而非把标记当独白）。
    """
    return _ANY_DECISION_MARKER.sub("", text or "").strip()


def _parse_monologue_cot(raw: str) -> dict:
    """解析思维链（CoT）格式的内心独白。

    格式：
    [自由思考过程...]
    [决定：开口/沉默/看看他]
    "要说的话"（如果是开口）

    失败兜底为沉默。决策标记关键词兼容简体/繁体/日文汉字 + 全/半角括号。
    """
    default = {"thought": "", "action": "silent", "text": "", "emotion": "", "face": "", "emotion_delta": {}}
    if not raw or not raw.strip():
        return default

    raw = raw.strip()

    # 立绘表情标记 [face:得意]（CoT 用方括号格式，可能出现在决策行末尾或 thought 里）
    face = ""
    face_match = re.search(r'\[face\s*:\s*([\w]+)\s*\]', raw, re.IGNORECASE)
    if face_match:
        face = face_match.group(1)
        raw = re.sub(r'\[face\s*:\s*[\w]+\s*\]', '', raw, flags=re.IGNORECASE).strip()

    action = "silent"  # 默认沉默
    decision_match = None
    for act, pattern in _ACTION_PATTERNS.items():
        m = re.search(pattern, raw, re.IGNORECASE)
        if m:
            action = act
            decision_match = m
            break

    # 提取思考过程（决策标记之前的部分）
    if decision_match:
        thought = raw[:decision_match.start()].strip()
        remaining = raw[decision_match.end():].strip()
    else:
        # 没有决策标记：整段当思考过程，默认沉默
        thought = raw
        remaining = ""

    # 净化 thought：剥掉任何残留的决策标记（防止标记本身被当成内心独白落进意识流）
    thought = _strip_decision_markers(thought)

    # 提取要说的话（引号内的内容）
    text = ""
    if action == "speak" and remaining:
        # 提取引号内容（支持中英文引号）
        quote_match = re.search(r'["""](.*?)["""]', remaining, re.DOTALL)
        if quote_match:
            text = quote_match.group(1).strip()
        else:
            # 没有引号，把剩余部分当作要说的话
            text = remaining

    # 情绪标签提取（可选，从思考链或文本末尾提取 <emo:xxx>）
    emotion = ""
    emo_match = re.search(r'<emo:(\w+)>', text or thought)
    if emo_match:
        emotion = emo_match.group(1).lower()
        # 从文本中移除情绪标签
        if text:
            text = re.sub(r'<emo:\w+>', '', text).strip()

    # retry_later：从决策标记后/思考里提取延后分钟数（默认 10）
    retry_minutes = 10
    if action == "retry_later":
        rm = _RETRY_MINUTES_RE.search(remaining or thought or raw)
        if rm:
            try:
                retry_minutes = max(1, min(int(rm.group(1)), 120))
            except (TypeError, ValueError):
                retry_minutes = 10

    return {
        "thought": thought,
        "action": action,
        "text": text,
        "emotion": emotion,
        "face": face,
        "emotion_delta": {},  # CoT 模式下情绪增量单独判断
        "retry_minutes": retry_minutes,
    }


def _parse_monologue(raw: str, use_cot: bool = False) -> dict:
    """解析内心独白：支持结构化 JSON 和思维链 CoT 两种格式。

    Args:
        raw: LLM 原始输出
        use_cot: 是否使用 CoT 解析（True=思维链，False=JSON）

    Returns:
        {thought, action, text, emotion, face, emotion_delta}
    """
    if use_cot:
        return _parse_monologue_cot(raw)

    # 原有 JSON 解析逻辑（完全不变）
    default = {"thought": "", "action": "silent", "text": "", "emotion": "", "face": "", "emotion_delta": {}}
    if not raw or not raw.strip():
        return default
    # 提取第一个 JSON 对象（容忍模型外面套了 ```json 或解释文字）
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if not m:
        # 没有 JSON：把整段当作想说的话（兼容模型不守格式）
        return {"thought": "", "action": "speak", "text": raw.strip(), "emotion": "", "face": "", "emotion_delta": {}}
    try:
        data = json.loads(m.group(0))
    except (json.JSONDecodeError, ValueError):
        return {"thought": "", "action": "speak", "text": raw.strip(), "emotion": "", "face": "", "emotion_delta": {}}
    action = str(data.get("action", "silent")).strip().lower()
    text = str(data.get("text", "")).strip()
    thought = str(data.get("thought", "")).strip()
    emotion = str(data.get("emotion", "")).strip().lower()
    face = str(data.get("face", "")).strip()
    delta = _parse_emotion_delta(data.get("emotion_delta"))
    if action == "look":
        return {"thought": thought, "action": "look", "text": "", "emotion": emotion, "face": face, "emotion_delta": delta}
    if action == "retry_later":
        retry_minutes = 10
        try:
            retry_minutes = max(1, min(int(data.get("retry_minutes", 10)), 120))
        except (TypeError, ValueError):
            retry_minutes = 10
        return {"thought": thought, "action": "retry_later", "text": "", "emotion": emotion,
                "face": face, "emotion_delta": delta, "retry_minutes": retry_minutes}
    if action != "speak" or not text:
        return {"thought": thought, "action": "silent", "text": "", "emotion": emotion, "face": face, "emotion_delta": delta}
    return {"thought": thought, "action": "speak", "text": text, "emotion": emotion, "face": face, "emotion_delta": delta}


def _parse_motion_map(raw: str, emotions: list[str]) -> dict[str, str]:
    """从 LLM 输出解析 {情绪: 抖动动作}。

    动作归一为 bounce/jump/shake/none（接受英文及常见中文别名），只保留 emotions
    列表里的情绪。找不到 JSON 或解析失败返回 {}（调用方用代码默认兜底）。
    """
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if not m:
        return {}
    try:
        data = json.loads(m.group(0))
    except (json.JSONDecodeError, ValueError):
        return {}
    valid = {"bounce", "jump", "shake", "none"}
    out: dict[str, str] = {}
    for emo, motion in data.items():
        if emo in emotions:
            n = _normalize_motion(motion)
            if n:
                out[str(emo)] = n
    return out


# 抖动动作别名 -> 规范名（LLM 可能返回中文，统一归一）
_MOTION_ALIASES = {
    "bounce": "bounce", "jump": "jump", "shake": "shake", "none": "none",
    "上下跳": "bounce", "跳": "bounce", "跳动": "bounce", "弹跳": "bounce",
    "猛跳": "jump", "跳一下": "jump", "猛跳一下": "jump",
    "左右晃": "shake", "晃": "shake", "抖动": "shake", "摇晃": "shake",
    "不动": "none", "无": "none", "静止": "none", "默认": "none",
}


def _normalize_motion(v) -> str | None:
    """把 LLM 返回的动作值归一为 bounce/jump/shake/none，无法识别返回 None。"""
    s = str(v).strip()
    if s in _MOTION_ALIASES:
        return _MOTION_ALIASES[s]
    sl = s.lower()
    return _MOTION_ALIASES.get(sl)


class Cognition:
    def __init__(self, config: Config):
        self._cfg = config
        self._model = config.get("modules.llm.model", "claude-opus-4-8")
        self._temperature = config.get("modules.llm.temperature", 0.85)
        self._max_tokens = config.get("modules.llm.max_tokens", 1024)

        provider = config.get("modules.llm.provider", "anthropic")
        api_key = config.get("modules.llm.api_key") or None
        base_url = config.get("modules.llm.base_url") or None  # OpenAI 兼容接口

        if provider == "anthropic":
            from anthropic import AsyncAnthropic
            self._client = AsyncAnthropic(api_key=api_key)
            self._backend = "anthropic"
        else:
            # openai / deepseek / 任何 OpenAI 兼容服务
            from openai import AsyncOpenAI
            self._client = AsyncOpenAI(api_key=api_key or "sk-placeholder", base_url=base_url)
            self._backend = "openai"

    @property
    def client(self):
        """底层 LLM 客户端 (供 Classifier 等复用)。"""
        return self._client

    @property
    def backend(self) -> str:
        """"anthropic" | "openai"。"""
        return self._backend

    @property
    def model(self) -> str:
        return self._model

    async def react_stream(
        self,
        card: CharacterCard,
        user_name: str,
        user_text: str,
        chat_history: list[dict],
        user_persona: str = "",
        world_before: list[str] | None = None,
        world_after: list[str] | None = None,
        emotion_summary: str = "",
        memories: list[str] | None = None,
        preset: dict | None = None,
        reply_lang: str = "zh",
        translation_lang: str = "",
        voice_emotions: list[str] | None = None,
        earlier_summary: str = "",
        time_context: str = "",
        face_emotions: list[str] | None = None,
    ) -> AsyncIterator[str]:
        """反应路径: 流式产出回复文本片段。anthropic 后端支持工具调用。"""
        system_prompt, messages = build_reactive_prompt(
            card, user_name, user_text, chat_history,
            user_persona=user_persona,
            world_before=world_before,
            world_after=world_after,
            emotion_summary=emotion_summary,
            memories=memories,
            preset=preset,
            reply_lang=reply_lang,
            translation_lang=translation_lang,
            voice_emotions=voice_emotions,
            earlier_summary=earlier_summary,
            time_context=time_context,
            face_emotions=face_emotions,
        )
        log.info("[LLM] 反应路径调用 %s/%s（用户：%s）",
                 self._backend, self._model, (user_text or "")[:40])
        log.debug("[LLM] 反应路径 system 含face指令=%s 含emo指令=%s face档=%s",
                  "立绘表情标签" in system_prompt, "语气标签" in system_prompt,
                  face_emotions)
        if self._backend == "anthropic":
            async for text in self._anthropic_stream(system_prompt, messages):
                yield text
        else:
            # OpenAI 兼容：system 作为第一条消息 + 工具循环
            async for text in self._openai_stream(system_prompt, messages):
                yield text

    async def _openai_stream(
        self, system_prompt: str, messages: list[dict]
    ) -> AsyncIterator[str]:
        """OpenAI 兼容流式 + 工具循环：有 tool_calls 则执行后再续流。"""
        # 转换工具 schema：Anthropic input_schema → OpenAI parameters（registry 已在模块顶部导入）
        tools_anthropic = registry.get_schemas()
        tools_openai = None
        if tools_anthropic:
            tools_openai = [
                {
                    "type": "function",
                    "function": {
                        "name": t["name"],
                        "description": t["description"],
                        "parameters": t["input_schema"],  # Anthropic input_schema == OpenAI parameters
                    },
                }
                for t in tools_anthropic
            ]

        oai_messages = [{"role": "system", "content": system_prompt}] + list(messages)
        if tools_openai:
            log.info("[LLM] 本次提供给 LLM 的工具: %s", [t["function"]["name"] for t in tools_openai])
        max_rounds = 5  # 防死循环
        for _ in range(max_rounds):
            kwargs = dict(
                model=self._model,
                messages=oai_messages,
                temperature=self._temperature,
                max_tokens=self._max_tokens,
                stream=True,
            )
            if tools_openai:
                kwargs["tools"] = tools_openai

            collected_text = []
            tool_calls = []
            stream = await self._client.chat.completions.create(**kwargs)
            async for chunk in stream:
                delta = chunk.choices[0].delta
                # 流式输出文本
                if delta.content:
                    collected_text.append(delta.content)
                    yield delta.content
                # 收集 tool_calls（可能分多个 chunk）
                if delta.tool_calls:
                    for tc_delta in delta.tool_calls:
                        idx = tc_delta.index
                        # 扩展列表
                        while len(tool_calls) <= idx:
                            tool_calls.append({"id": "", "name": "", "arguments": ""})
                        if tc_delta.id:
                            tool_calls[idx]["id"] = tc_delta.id
                        if tc_delta.function.name:
                            tool_calls[idx]["name"] = tc_delta.function.name
                        if tc_delta.function.arguments:
                            tool_calls[idx]["arguments"] += tc_delta.function.arguments

            # 没有工具调用 → 结束
            if not tool_calls:
                full = "".join(collected_text)
                log.debug("[LLM] 反应路径完整输出: %s", full[:500])
                return

            # 有工具调用 → 执行 + 回灌
            import json
            assistant_msg = {
                "role": "assistant",
                "content": "".join(collected_text) or None,
                "tool_calls": [
                    {
                        "id": tc["id"],
                        "type": "function",
                        "function": {"name": tc["name"], "arguments": tc["arguments"]},
                    }
                    for tc in tool_calls
                ],
            }
            oai_messages.append(assistant_msg)

            for tc in tool_calls:
                try:
                    args = json.loads(tc["arguments"]) if tc["arguments"] else {}
                    log.info("[LLM] 调用工具 %s（%s）", tc["name"], json.dumps(args, ensure_ascii=False)[:80])
                    result = await registry.call(tc["name"], **args)
                    log.info("[LLM] 工具 %s 返回: %s", tc["name"], str(result)[:200])
                except Exception as e:  # noqa: BLE001
                    result = f"工具调用失败: {e}"
                    log.warning("[LLM] 工具 %s 调用失败: %s", tc["name"], e)
                oai_messages.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": str(result),
                })

    async def _anthropic_stream(
        self, system_prompt: str, messages: list[dict]
    ) -> AsyncIterator[str]:
        """anthropic 流式 + 工具循环：有 tool_use 则执行后再续流。"""
        tools = registry.get_schemas()
        msgs = list(messages)
        while True:
            kwargs = dict(model=self._model, max_tokens=self._max_tokens,
                          temperature=self._temperature, system=system_prompt,
                          messages=msgs)
            if tools:
                kwargs["tools"] = tools
            async with self._client.messages.stream(**kwargs) as stream:
                async for text in stream.text_stream:
                    yield text
                final = await stream.get_final_message()

            tool_uses = [b for b in final.content if b.type == "tool_use"]
            if not tool_uses:
                return
            msgs.append({"role": "assistant", "content": final.content})
            results = []
            for tu in tool_uses:
                try:
                    import json as _json
                    log.info("[LLM] 调用工具 %s（%s）", tu.name,
                             _json.dumps(tu.input or {}, ensure_ascii=False)[:80])
                    out = await registry.call(tu.name, **(tu.input or {}))
                    log.info("[LLM] 工具 %s 返回: %s", tu.name, str(out)[:200])
                except Exception as e:  # noqa: BLE001
                    out = f"工具调用失败: {e}"
                    log.warning("[LLM] 工具 %s 调用失败: %s", tu.name, e)
                results.append({"type": "tool_result",
                                "tool_use_id": tu.id, "content": str(out)})
            msgs.append({"role": "user", "content": results})

    async def proactive_think(
        self,
        card: CharacterCard,
        user_name: str,
        trigger_reason: str,
        emotion_summary: str,
        chat_history: list[dict],
        elapsed_desc: str,
        can_look: bool = False,
        preset: dict | None = None,
        reply_lang: str = "zh",
        translation_lang: str = "",
        voice_emotions: list[str] | None = None,
        memories: list[str] | None = None,
        earlier_summary: str = "",
        time_context: str = "",
        think_seed: str = "",
        recent_inner: list[str] | None = None,
        face_emotions: list[str] | None = None,
    ) -> dict:
        """主动路径内心独白: 返回 {thought, action, text, emotion, face}，action 可为 speak/silent/look。

        支持工具调用：如果 LLM 调用工具，执行后回灌结果，让 LLM 基于工具结果决定要不要说话。
        """
        # 读取 CoT 开关
        use_cot = self._cfg.get("token_intensive.proactive_cot", False)

        system_prompt, messages = build_proactive_prompt(
            card, user_name, trigger_reason, emotion_summary, chat_history, elapsed_desc,
            can_look=can_look, preset=preset, reply_lang=reply_lang,
            translation_lang=translation_lang,
            voice_emotions=voice_emotions,
            memories=memories,
            earlier_summary=earlier_summary,
            time_context=time_context,
            think_seed=think_seed,
            recent_inner=recent_inner,
            use_cot=use_cot,
            face_emotions=face_emotions,
        )

        # 工具循环（主动路径：非流式，返回结构化 JSON 或思维链）
        tools = registry.get_schemas()
        log.info("[LLM] 主动路径调用 %s/%s（触发：%s，CoT=%s）",
                 self._backend, self._model, trigger_reason, use_cot)
        max_rounds = 5  # 防死循环

        if self._backend == "anthropic":
            return await self._proactive_anthropic(system_prompt, messages, tools, max_rounds, use_cot)
        else:
            return await self._proactive_openai(system_prompt, messages, tools, max_rounds, use_cot)

    async def _proactive_anthropic(
        self, system_prompt: str, messages: list[dict], tools: list[dict], max_rounds: int, use_cot: bool = False
    ) -> dict:
        """Anthropic 主动路径工具循环：非流式，返回结构化 JSON 或思维链。"""
        msgs = list(messages)
        for _ in range(max_rounds):
            kwargs = dict(
                model=self._model,
                max_tokens=self._max_tokens,
                temperature=self._temperature,
                system=system_prompt,
                messages=msgs,
            )
            if tools:
                kwargs["tools"] = tools

            resp = await self._client.messages.create(**kwargs)

            # 检查是否有工具调用
            tool_uses = [b for b in resp.content if b.type == "tool_use"]
            if not tool_uses:
                # 没有工具调用，解析最终响应
                text = "".join(b.text for b in resp.content if hasattr(b, "text"))
                result = _parse_monologue(text, use_cot=use_cot)
                log.debug("[LLM] 主动路径完整输出: %s", text[:500])
                log.debug("[LLM] 主动路径解析: action=%s emotion=%s face=%s text=%r",
                          result.get("action"), result.get("emotion"), result.get("face"),
                          (result.get("text") or "")[:80])
                return result

            # 有工具调用：执行工具
            msgs.append({"role": "assistant", "content": resp.content})
            results = []
            for tu in tool_uses:
                try:
                    import json as _json
                    log.info("[LLM] 调用工具 %s（%s）", tu.name,
                             _json.dumps(tu.input or {}, ensure_ascii=False)[:80])
                    out = await registry.call(tu.name, **(tu.input or {}))
                    log.info("[LLM] 工具 %s 返回: %s", tu.name, str(out)[:200])
                except Exception as e:  # noqa: BLE001
                    out = f"工具调用失败: {e}"
                    log.warning("[LLM] 工具 %s 调用失败: %s", tu.name, e)
                results.append({
                    "type": "tool_result",
                    "tool_use_id": tu.id,
                    "content": str(out)
                })
            msgs.append({"role": "user", "content": results})
            # 循环继续，让 LLM 基于工具结果生成最终响应

        # 超出最大轮次，返回 silent
        log.warning("[LLM] 主动路径工具循环超限（%d 轮），返回 silent", max_rounds)
        return {"thought": "工具循环超限", "action": "silent", "text": "", "emotion": "", "emotion_delta": {}}

    async def _proactive_openai(
        self, system_prompt: str, messages: list[dict], tools: list[dict], max_rounds: int, use_cot: bool = False
    ) -> dict:
        """OpenAI 兼容主动路径工具循环：非流式，返回结构化 JSON 或思维链。"""
        # 转换工具 schema
        tools_openai = None
        if tools:
            tools_openai = [
                {
                    "type": "function",
                    "function": {
                        "name": t["name"],
                        "description": t["description"],
                        "parameters": t["input_schema"],
                    },
                }
                for t in tools
            ]

        oai_messages = [{"role": "system", "content": system_prompt}] + list(messages)

        for _ in range(max_rounds):
            kwargs = dict(
                model=self._model,
                messages=oai_messages,
                temperature=self._temperature,
                max_tokens=self._max_tokens,
            )
            if tools_openai:
                kwargs["tools"] = tools_openai

            resp = await self._client.chat.completions.create(**kwargs)
            choice = resp.choices[0]

            # 检查是否有工具调用
            if not choice.message.tool_calls:
                # 没有工具调用，解析最终响应
                text = choice.message.content or ""
                result = _parse_monologue(text, use_cot=use_cot)
                log.debug("[LLM] 主动路径完整输出: %s", text[:500])
                log.debug("[LLM] 主动路径解析: action=%s emotion=%s face=%s text=%r",
                          result.get("action"), result.get("emotion"), result.get("face"),
                          (result.get("text") or "")[:80])
                return result

            # 有工具调用：执行工具
            oai_messages.append({
                "role": "assistant",
                "content": choice.message.content,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.function.name, "arguments": tc.function.arguments}
                    }
                    for tc in choice.message.tool_calls
                ]
            })

            for tc in choice.message.tool_calls:
                try:
                    import json
                    args = json.loads(tc.function.arguments)
                    log.info("[LLM] 调用工具 %s（%s）", tc.function.name,
                             json.dumps(args, ensure_ascii=False)[:80])
                    out = await registry.call(tc.function.name, **args)
                    log.info("[LLM] 工具 %s 返回: %s", tc.function.name, str(out)[:200])
                except Exception as e:  # noqa: BLE001
                    out = f"工具调用失败: {e}"
                    log.warning("[LLM] 工具 %s 调用失败: %s", tc.function.name, e)
                oai_messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": str(out)
                })
            # 循环继续，让 LLM 基于工具结果生成最终响应

        # 超出最大轮次，返回 silent
        log.warning("[LLM] 主动路径工具循环超限（%d 轮），返回 silent", max_rounds)
        return {"thought": "工具循环超限", "action": "silent", "text": "", "emotion": "", "emotion_delta": {}}

    async def _complete(self, system_prompt: str, messages: list[dict]) -> str:
        """非流式补全，两种后端统一接口。"""
        log.debug("[LLM] 补全调用 %s/%s（%d 条消息）",
                  self._backend, self._model, len(messages))
        if self._backend == "anthropic":
            resp = await self._client.messages.create(
                model=self._model,
                max_tokens=self._max_tokens,
                temperature=self._temperature,
                system=system_prompt,
                messages=messages,
            )
            return "".join(b.text for b in resp.content if b.type == "text")
        else:
            oai_messages = [{"role": "system", "content": system_prompt}] + messages
            resp = await self._client.chat.completions.create(
                model=self._model,
                messages=oai_messages,
                temperature=self._temperature,
                max_tokens=self._max_tokens,
            )
            return resp.choices[0].message.content or ""

    async def fix_reply(self, text: str, issues: list[str]) -> str:
        """回复修正器（v2.68）：在原回复基础上做最小修正，**不是重新生成**。

        审查发现的问题（issues）作为约束喂给 LLM，要求保留原意/口吻/语言格式
        （若有「日文+中文翻译括号」格式须保留）、长度适中，只做最小改动。
        失败返回 ""（调用方回退用原回复，不阻断对话）。
        """
        sys_prompt = (
            "你是回复修正器。下面是 AI 给用户的回复，审查发现以下问题：\n"
            + "\n".join(f"- {i}" for i in issues) + "\n"
            "请在原回复基础上做最小修正：保留原意、口吻、语言格式"
            "（若有日文+中文翻译括号请保留）、长度适中。"
            "不要重新生成、不要扩写、不要加解释或前缀。直接输出修正后的完整回复。"
        )
        msg = [{"role": "user", "content": text}]
        try:
            out = await self._complete(sys_prompt, msg)
            return (out or "").strip()
        except Exception as e:  # noqa: BLE001
            log.warning("fix_reply 失败: %s", e)
            return ""

    async def judge_emotion_delta(self, user_text: str, reply: str) -> dict:
        """反应路径：判断这轮对话对 ta 情绪的影响，返回 delta dict（失败返回 {}）。

        由 orchestrator fire-and-forget 调用，不阻塞对话主流程。轻量请求
        （max_tokens 小、低温），只要 LLM 给出五维增量的小 JSON。
        解析复用 _parse_emotion_delta（只保留已知维度、数值、钳 [-1,1]）。
        """
        sys_prompt = (
            "你是情绪评估器。根据用户这轮说的话和AI的回应，判断这轮对话让AI的情绪"
            "如何**变化**（增量，不是绝对值）。五个维度：\n"
            "valence 愉悦度[-1~1]：聊得开心↑、被骂/难过↓\n"
            "arousal 唤醒度[-1~1]：兴奋/激动↑、平静↓\n"
            "attachment 依恋度[-1~1]：亲密温暖的互动↑\n"
            "worry 担心[-1~1]：用户说累了/不舒服/出事↑、报平安↓\n"
            "loneliness 孤独感[-1~1]：通常这轮在互动故应↓\n"
            "只输出有**明显**变化的维度，增量幅度一般 0.05~0.3。没明显变化就输出 {}。"
            "严格只输出 JSON，如 {\"valence\":0.15,\"attachment\":0.1}，不要任何其他文字。"
        )
        messages = [{"role": "user",
                     "content": f"用户说：{user_text}\nAI回应：{reply}"}]
        try:
            log.debug("[LLM] 情绪判断调用 %s/%s（用户：%s）",
                      self._backend, self._model, (user_text or "")[:30])
            # 轻量：临时压低 max_tokens / temperature（_complete 用实例值，故这里直调）
            if self._backend == "anthropic":
                resp = await self._client.messages.create(
                    model=self._model, max_tokens=80, temperature=0.3,
                    system=sys_prompt, messages=messages,
                )
                raw = "".join(b.text for b in resp.content if b.type == "text")
            else:
                resp = await self._client.chat.completions.create(
                    model=self._model,
                    messages=[{"role": "system", "content": sys_prompt}] + messages,
                    temperature=0.3, max_tokens=80,
                )
                raw = resp.choices[0].message.content or ""
        except Exception as e:  # noqa: BLE001
            log.warning("judge_emotion_delta 失败: %s", e)
            return {}
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if not m:
            return {}
        try:
            data = json.loads(m.group(0))
        except (json.JSONDecodeError, ValueError):
            return {}
        return _parse_emotion_delta(data)

    async def generate_motion_map(self, emotions: list[str]) -> dict[str, str]:
        """给每个表情分配一个立绘抖动动作（bounce/jump/shake/none），LLM 生成。

        立绘库管理页「大模型生成抖动动作」用：把角色立绘库的表情列表给 LLM，
        让它按情绪语义分配合适的抖动。失败返回 {}（调用方用代码默认兜底）。
        """
        if not emotions:
            return {}
        sys_prompt = (
            "你是立绘动作设计师。给每个表情分配一个抖动动作，让立绘切换到该表情时动一下。\n"
            "动作类型：\n"
            "- bounce：上下跳动几下（开心/得意/兴奋/愉快/期待 等积极活跃情绪）\n"
            "- jump：猛跳一下（惊讶/吃惊/震惊/吓一跳 等强烈突发情绪）\n"
            "- shake：左右晃动（生气/愤怒/烦躁/不满 等负面激动情绪）\n"
            "- none：不动（平静/中性/思考/发呆/睡觉 等安静情绪）\n"
            "只输出 JSON，键为表情名原样，值为上述四种之一。"
            "如 {\"得意\":\"bounce\",\"惊讶\":\"jump\",\"neutral\":\"none\"}。"
        )
        msg = [{"role": "user", "content": "表情列表：" + ", ".join(emotions)}]
        try:
            raw = await self._complete(sys_prompt, msg)
        except Exception as e:  # noqa: BLE001
            log.warning("generate_motion_map 失败: %s", e)
            return {}
        log.debug("[LLM] generate_motion_map 原始输出: %s", (raw or "")[:200])
        out = _parse_motion_map(raw, emotions)
        if not out:
            log.warning("generate_motion_map 解析为空，原始输出: %s", (raw or "")[:200])
        return out

    async def summarize_history(self, batch_text: str, prev_summary: str = "") -> str:
        """把"旧摘要 + 这批要移出窗口的对话"融合成一段简洁的早先聊天摘要。

        用于短期窗口增量压缩：窗口溢出一批时，把最旧那批浓缩进摘要保留脉络，
        而非直接丢弃。失败抛异常由调用方降级处理（保留旧摘要 = 等于直接丢）。
        """
        sys_prompt = (
            "你是对话记录的摘要器。把【已有摘要】和【新增对话】融合成一段连贯的中文摘要，"
            "概括到目前为止聊过的主要话题、对方提到的重要信息/情绪/事件，"
            "保留时间脉络感，去掉寒暄和无信息量的内容。直接输出摘要正文，不超过200字，"
            "不要加'摘要：'之类前缀。"
        )
        parts = []
        if prev_summary:
            parts.append(f"【已有摘要】\n{prev_summary}")
        parts.append(f"【新增对话】\n{batch_text}")
        messages = [{"role": "user", "content": "\n\n".join(parts)}]
        text = await self._complete(sys_prompt, messages)
        return (text or "").strip()
