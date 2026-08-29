"""角色与 prompt 组装 (架构文档 6.5)。阶段1: 最简版，只用角色卡，不含世界书/记忆/情绪。"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from .logger import get_logger

log = get_logger("worldinfo")


DEFAULT_REPLY_STYLE = "保持口语化、简短，一两句话即可。不要用书面语或列表。"


@dataclass
class CharacterCard:
    name: str
    description: str = ""
    personality: str = ""
    scenario: str = ""
    first_mes: str = ""
    mes_example: str = ""
    system_prompt: str = ""
    post_history_instructions: str = ""
    alternate_greetings: list = field(default_factory=list)
    extensions: dict = field(default_factory=dict)

    @classmethod
    def load(cls, path: Path) -> "CharacterCard":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        # 兼容 ST 角色卡的 data 嵌套结构
        if "data" in data and isinstance(data["data"], dict):
            data = data["data"]
        known = cls.__dataclass_fields__.keys()
        return cls(**{k: v for k, v in data.items() if k in known})


def _subst(text: str, char_name: str, user_name: str) -> str:
    return (text or "").replace("{{char}}", char_name).replace("{{user}}", user_name)


def load_preset(config, char_name: str | None = None) -> dict:
    """加载当前预设（提示词预设）。不存在或未配置时返回空 dict（不影响组装）。

    预设字段（均可选）：
      - system_prompt: 覆盖/补充角色 system（拼在最前，角色卡 system_prompt 之前）
      - jailbreak:     破限/越狱提示（拼在 system 最末尾，最强位置）
      - post_history:  历史后指令（在角色卡 post_history_instructions 之后）
      - temperature / max_tokens: 采样参数（cognition 读取，可选）
    """
    import json
    # 预设总开关：character.preset_enabled=false 时不加载（即便选了 current_preset）
    if not config.get("character.preset_enabled", True):
        return {}
    name = config.get("character.current_preset") or ""
    if not name:
        return {}
    p = config.project_root / "data" / "presets" / f"{name}.json"
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, ValueError, OSError):
        return {}


_LANG_NAMES = {
    "zh": "中文", "en": "English", "ja": "日本語", "ko": "한국어",
    "fr": "français", "de": "Deutsch", "es": "español", "ru": "русский",
}


def _lang_instruction(reply_lang: str, translation_lang: str = "") -> str:
    """生成语言约束指令。

    Args:
        reply_lang: 回复使用的语言（zh/en/ja等）
        translation_lang: 翻译目标语言，为空则不翻译

    Returns:
        语言约束指令字符串
    """
    lang = reply_lang.strip().lower()
    trans = translation_lang.strip().lower()

    # 没有指定回复语言或回复语言是中文
    if lang in ("zh", ""):
        return "请始终用中文回复。"

    reply_name = _LANG_NAMES.get(lang, lang)

    # 需要翻译
    if trans and trans != lang:
        trans_name = _LANG_NAMES.get(trans, trans)
        return (
            f"【语言要求·必须严格遵守】\n"
            f"1. 用{reply_name}回复，**每一句话**后面都要紧跟{trans_name}翻译，格式：{reply_name}原文（{trans_name}翻译）。\n"
            f"2. 翻译用全角括号（）包裹，且**每句话都要翻译，不能漏**——哪怕只有一句也要附翻译。\n"
            f"3. 不要照搬/复述历史里出现过的回复，每次都要重新生成，并确保带上翻译。\n"
            f"4. 历史消息里的 `[X分钟前]` 是系统加的时间标记，仅供你参考时间，**绝不要**在你的回复里模仿这个格式或照抄它。\n"
            f"示例：おはよう（早上好）"
        )

    # 不需要翻译
    return f"请始终用{reply_name}回复。"


def _emotion_instruction(voice_emotions: list[str] | None) -> str:
    """情绪标签指令（方案3 库驱动）：可选情绪来自语音库实际的键。

    要求每次回复以 <emo:情绪> 开头，仅控制 TTS 语气、不会被读出、不进聊天记录。
    空列表则返回空串（无语音库时退化为无标签纯文本）。
    另提醒：用户消息里可能出现 <|情绪|> 这类语音识别附带的标签，供你判断用户情绪，
    但你**不要**模仿这种格式（你只用 <emo:…>）。
    """
    if not voice_emotions:
        return ""
    opts = " / ".join(voice_emotions)
    return (
        f"【语气标签】你的语音能表达这些情绪：{opts}。\n"
        f"**每次回复的第一件事就是输出 `<emo:情绪>` 标签**——必须是回复最开头的第一个字符，"
        f"不能先说别的话再补标签。从上面选一个最贴合当下心情的，例如 `<emo:happy>`。"
        f"漏标或标在非开头位置是严重错误。标签只控制说话语气，不会被读出来也不显示，放心标。\n"
        f"注意：用户的话里有时会带 `<|愉快|>`、`<|HAPPY|>` 这类语音识别附带的情绪标记，"
        f"那是给你参考用户情绪的，你**不要**模仿这种写法，你只用开头的 `<emo:…>`。"
    )


def _face_instruction(face_emotions: list[str] | None) -> str:
    """立绘表情标签指令（库驱动）：可选表情来自立绘库 sprites.json 实际的键。

    与 <emo:>（语音语气）独立——立绘表情和说话语气是两个维度，可不同。
    要求回复开头以 <face:表情> 标注本次立绘表情，仅驱动立绘换图，不朗读、不显示。
    空列表则返回空串（无立绘库时退化，不要求 face 标签）。
    """
    if not face_emotions:
        return ""
    opts = " / ".join(face_emotions)
    return (
        f"【立绘表情标签】你能展现这些立绘表情：{opts}。\n"
        f"**每次回复必须在最开头输出 `<face:表情>` 标签**（和 `<emo:>` 一起放最开头，"
        f"顺序随意如 `<emo:happy><face:得意>`），不能漏、不能放非开头位置。"
        f"从上面选一个最贴合的，例如 `<face:得意>`。它独立于 `<emo:…>`（语气），"
        f"只决定画面上你的神态，不会被读出来也不显示给用户。"
    )


def _tool_guidance() -> str:
    """动态生成工具使用引导（反应路径 system 末尾）。

    从 registry 读当前已注册工具，告诉 LLM：①有哪些工具可用 ②何时该调
    ③关键——不要凭对话历史里出现过的视觉 caption/记忆/旧信息编"当前"事实，
    那可能已过时，要看现在画面必须调 look、要查实时信息必须调对应工具。

    无工具注册时返回空串（不污染 prompt）。
    """
    try:
        from .tools import registry
        schemas = registry.get_schemas()
    except Exception:  # noqa: BLE001
        return ""
    if not schemas:
        return ""
    names = "、".join(s["name"] for s in schemas)
    has_look = any(s["name"] == "look" for s in schemas)
    look_clause = (
        "用户要你「看看」「看一下」现在的画面/姿势/样子→**必须调 look 工具**抓当前帧，"
        "绝不凭对话历史里「系统自动抓取到画面」的内容编——那是被动抓取的、可能已过时"
        "（人可能已移动/换装/离开）。即使用户没明说\"调用工具\"，只要意图是看现在，就调 look。"
        if has_look else ""
    )
    return (
        "# 工具使用\n"
        f"你可用工具：{names}。需要时**主动调用**，不要硬编结论。\n"
        f"{look_clause}\n"
        "问天气/新闻/实时信息→调 get_weather/web_search，不要凭记忆编。"
        "只有工具返回的结果才是当前真实信息，历史里的都是过去的。"
    )


def _build_system(card: CharacterCard, user_name: str,
                  user_persona: str = "",
                  world_before: list[str] | None = None,
                  world_after: list[str] | None = None,
                  preset: dict | None = None,
                  reply_lang: str = "zh",
                  translation_lang: str = "",
                  voice_emotions: list[str] | None = None,
                  lang_instruction: bool = True,
                  face_emotions: list[str] | None = None,
                  tool_guidance: bool = False) -> str:
    """组装角色系统提示。世界书 before 在角色定义前, after 在其后 (架构文档 6.5)。

    预设（提示词预设）叠加位置：
      - preset.system_prompt → 拼在最前（角色定义之前的全局指令）
      - preset.jailbreak     → 拼在 system 最末尾（破限，最强位置）
      - preset.post_history  → 在角色卡 post_history_instructions 之后

    lang_instruction=False 时**不**拼那条"用X语回复+附翻译"的一刀切语言指令。
    主动路径用它关掉——主动路径输出含 thought/决策标记/text 多种角色，
    语言规则改由 monologue 指令按字段精确控制（见 _proactive_lang_rules），
    否则这条一刀切指令会污染 thought（把内心独白也写成回复语言+翻译），
    并让决策标记被翻译（如 `[決定：沈黙]（决定：沉默）`）。
    """
    cn, un = card.name, user_name
    preset = preset or {}
    sys_parts: list[str] = []
    # 预设全局 system（最前）
    if preset.get("system_prompt"):
        sys_parts.append(_subst(preset["system_prompt"], cn, un))
    if card.system_prompt:
        sys_parts.append(_subst(card.system_prompt, cn, un))
    elif not preset.get("system_prompt"):
        sys_parts.append(f"你是{cn}，正在和{un}相处。始终以{cn}的身份、口吻自然地表达。")
    # 世界书 before
    for c in (world_before or []):
        sys_parts.append(_subst(c, cn, un))
    # 用户人设
    if user_persona:
        sys_parts.append(f"# 关于{un}\n{_subst(user_persona, cn, un)}")
    if card.description:
        sys_parts.append(f"# 角色设定\n{_subst(card.description, cn, un)}")
    if card.personality:
        sys_parts.append(f"# 性格\n{_subst(card.personality, cn, un)}")
    if card.scenario:
        sys_parts.append(f"# 当前情境\n{_subst(card.scenario, cn, un)}")
    # 世界书 after
    for c in (world_after or []):
        sys_parts.append(_subst(c, cn, un))
    if card.post_history_instructions:
        sys_parts.append(_subst(card.post_history_instructions, cn, un))
    # 预设历史后指令
    if preset.get("post_history"):
        sys_parts.append(_subst(preset["post_history"], cn, un))
    # 所有角色统一的默认回复风格。角色卡/预设里若已包含同一句则不重复注入。
    if not any(DEFAULT_REPLY_STYLE in part for part in sys_parts):
        sys_parts.append(DEFAULT_REPLY_STYLE)
    # 预设破限（拼最末，最强位置）
    if preset.get("jailbreak"):
        sys_parts.append(_subst(preset["jailbreak"], cn, un))
    # 语言约束（最末，覆盖角色卡/预设里可能有的语言要求）
    if lang_instruction:
        sys_parts.append(_lang_instruction(reply_lang, translation_lang))
    # 情绪语气标签指令（库驱动，空库则不加）
    emo_instr = _emotion_instruction(voice_emotions)
    if emo_instr:
        sys_parts.append(emo_instr)
    # 立绘表情标签指令（库驱动，空库则不加；与 emo 独立）
    face_instr = _face_instruction(face_emotions)
    if face_instr:
        sys_parts.append(face_instr)
    # 工具使用引导（仅反应路径：反应路径走 tool_use，主动路径用 action 机制不需要）
    if tool_guidance:
        tg = _tool_guidance()
        if tg:
            sys_parts.append(tg)
    return "\n\n".join(sys_parts)


def _proactive_lang_rules(reply_lang: str, translation_lang: str, thought_lang: str) -> str:
    """主动路径按字段的语言规则（thought / 决策标记 / text 各自的语言）。

    取代主动路径里的一刀切 _lang_instruction：
      - thought（内心独白）：用 thought_lang（= 翻译语言，没有则回复语言），**不附翻译**。
      - 决策标记 [决定：开口/沉默/看看他]：**固定中文控制记号，原样照抄、绝不翻译**
        （否则非中文回复语言下会被写成日文等、并附翻译括号，导致解析失配）。
      - text（speak 对外说的话）：用回复语言；翻译语言与回复语言不同时附「原文（翻译）」。
    """
    tl = (thought_lang or "zh").strip().lower()
    rl = (reply_lang or "zh").strip().lower()
    trans = (translation_lang or "").strip().lower()
    thought_name = _LANG_NAMES.get(tl, tl) if tl not in ("zh", "") else "中文"
    reply_name = _LANG_NAMES.get(rl, rl) if rl not in ("zh", "") else "中文"

    lines = [
        "【语言规则（请严格区分以下三部分，各用各的语言）】",
        f"1. 你的内心独白/思考过程：一律用{thought_name}书写，**不要**附任何翻译。",
        "2. 决策标记 `[决定：开口]`/`[决定：沉默]`/`[决定：看看他]`："
        "这是固定的**中文**控制记号，**原样照抄**，无论你用什么语言思考都不要翻译它、不要改写它。",
    ]
    if rl in ("zh", ""):
        lines.append("3. 你决定开口要说的话：用中文。")
    elif trans and trans != rl:
        trans_name = _LANG_NAMES.get(trans, trans)
        lines.append(
            f"3. 你决定开口要说的话（text 字段 / [决定：开口] 后引号内）：\n"
            f"   - 用{reply_name}说，**每一句**后面紧跟{trans_name}翻译，全角括号（）包裹，"
            f"格式：{reply_name}原文（{trans_name}翻译）。\n"
            f"   - **每句话都必须带翻译，绝对不能漏**——哪怕只说一句也要附翻译。"
            f"漏翻译是严重错误。\n"
            f"   - 不要照搬/复述历史里出现过的回复，每次重新生成并带翻译。\n"
            f"   - 示例：おかえり！（欢迎回家！）今日もお疲れさま。（今天也辛苦啦。）"
        )
    else:
        lines.append(f"3. 你决定开口要说的话：用{reply_name}。")
    # 防历史标记泄漏 + 防照搬历史
    lines.append(
        "4. 历史消息里的 `[X分钟前]` 是系统时间标记，仅供参考时间，"
        "**绝不要**在你要说的话或内心独白里模仿这个格式或照抄它；也不要照搬历史里出现过的回复。"
    )
    return "\n".join(lines)


# ─────────────────── 短期历史时间标注 ──────────────────────────────

def _rel_time(ts_str: str, now: datetime) -> str:
    """把 ISO 时间戳字符串转成口语相对时间（"刚刚"/"5分钟前"/"2小时前"/"3天前"）。

    无 ts 或解析失败返回空串（退化为不标注时间，兼容旧数据）。
    距今 < 2分钟不标注（太近加标记是噪声）。
    """
    if not ts_str:
        return ""
    try:
        dt = datetime.fromisoformat(ts_str)
    except (ValueError, TypeError):
        return ""
    delta = (now - dt).total_seconds()
    if delta < 120:
        return ""  # 两分钟以内不标
    if delta < 3600:
        return f"{int(delta // 60)}分钟前"
    if delta < 86400:
        h = int(delta // 3600)
        return f"{h}小时前"
    d = int(delta // 86400)
    return f"{d}天前"


def _history_to_messages(chat_history: list[dict], now: datetime | None = None) -> list[dict]:
    """把内部带 ts 的短期窗口转成 LLM API 格式 [{role, content}]。

    较旧消息在 content 前加 [X分钟前] 标记，让 AI 感知时间流逝。
    最近消息（< 2分钟）不加标记，避免噪声。
    无 ts 字段的旧数据不加标记，退化为现行为。
    """
    _now = now or datetime.now()
    out = []
    for m in chat_history:
        content = m.get("content", "")
        rel = _rel_time(m.get("ts", ""), _now)
        if rel:
            content = f"[{rel}] {content}"
        out.append({"role": m["role"], "content": content})
    return out


def build_reactive_prompt(
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
    reaction_hint: str = "",
) -> tuple[str, list[dict]]:
    """反应路径: 返回 (system_prompt, messages)。

    组装顺序 (架构文档 6.5): system → world_before → 用户人设 → 角色定义 →
    world_after → [情绪/记忆注入] → 对话历史 → 本轮输入 → 破限(在 system 末尾)。
    """
    system_prompt = _build_system(card, user_name, user_persona, world_before, world_after,
                                  preset, reply_lang, translation_lang, voice_emotions,
                                  face_emotions=face_emotions, tool_guidance=True)
    messages = _history_to_messages(chat_history)
    # 情绪 + 相关记忆作为 system 消息注入在历史前 (比历史更"新鲜")
    inject = []
    if earlier_summary:
        inject.append(f"[早先聊天摘要] {earlier_summary}")
    if time_context:
        inject.append(f"[当前时间] 现在是{time_context}。")
    if reaction_hint:
        inject.append(f"[本轮反应要求] {reaction_hint}")
    if emotion_summary:
        inject.append(f"[你此刻的情绪] {emotion_summary}")
    if memories:
        inject.append("[相关记忆]\n" + "\n".join(f"- {m}" for m in memories))
    if inject:
        messages.append({"role": "system", "content": "\n".join(inject)})
    messages.append({"role": "user", "content": user_text})
    return system_prompt, messages


def build_proactive_prompt(
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
    use_cot: bool = False,
    face_emotions: list[str] | None = None,
) -> tuple[str, list[dict]]:
    """主动路径: 内心独白 (架构文档 4.3 第3层)。

    丰富度增强：注入长期记忆 + 早先摘要 + 时间情境 + 思考切入点，
    并放开「只能想沉默和心情」的约束（禁编造画面，但允许回忆/好奇/联想），
    让内心活动有具体素材、不再千篇一律。

    支持两种输出格式：
    - use_cot=False: 结构化 JSON（默认）
    - use_cot=True: 思维链（CoT）自由推理

    语言策略：
    - thought 字段用翻译语言（方便查看 consciousness.jsonl）
    - text 字段用回复语言+翻译（对外说话）
    """
    # thought 用翻译语言（如果有），否则用回复语言
    # 这样 consciousness.jsonl 里的内心活动能直接看懂
    thought_lang = translation_lang if translation_lang else reply_lang

    # 主动路径 system **不**拼一刀切语言指令（lang_instruction=False）：
    # 它说"每句话用X语并附翻译"，会污染 thought（内心独白也变回复语言+翻译）、
    # 并让决策标记被翻译（如 `[決定：沈黙]（决定：沉默）`）。
    # 语言规则改由下方 monologue 里的 _proactive_lang_rules 按字段精确控制。
    system_prompt = _build_system(card, user_name, preset=preset, reply_lang=reply_lang,
                                  translation_lang=translation_lang, lang_instruction=False,
                                  voice_emotions=voice_emotions, face_emotions=face_emotions)

    lang_rules = _proactive_lang_rules(reply_lang, translation_lang, thought_lang)

    if use_cot:
        # 思维链模式：自由思考
        monologue = _build_proactive_cot_instruction(
            trigger_reason, emotion_summary, chat_history, elapsed_desc, memories,
            earlier_summary, time_context, think_seed, recent_inner, can_look, lang_rules,
            face_emotions, voice_emotions
        )
    else:
        # JSON 模式：原有逻辑（完全不变）
        monologue = _build_proactive_json_instruction(
            user_name, trigger_reason, emotion_summary, chat_history, elapsed_desc, voice_emotions,
            memories, earlier_summary, time_context, think_seed, recent_inner, can_look, lang_rules,
            face_emotions
        )

    messages = _history_to_messages(chat_history)
    messages.append({"role": "user", "content": monologue})
    return system_prompt, messages


def _build_proactive_json_instruction(
    user_name: str, trigger_reason: str, emotion_summary: str, chat_history: list[dict], elapsed_desc: str,
    voice_emotions: list[str] | None, memories: list[str] | None, earlier_summary: str,
    time_context: str, think_seed: str, recent_inner: list[str] | None, can_look: bool,
    lang_rules: str = "",
    face_emotions: list[str] | None = None,
) -> str:
    """构建 JSON 格式的内心独白指令（原有逻辑）。

    Args:
        lang_rules: 按字段的语言规则（thought/决策标记/text 各自语言），由
            _proactive_lang_rules 生成，拼在指令末尾。
        face_emotions: 立绘库可用表情档（库驱动），非空时要求输出 face 字段。
    """
    # 情绪字段说明：主动路径用 JSON 的 emotion 字段（非 <emo:> 前缀）传语气给 TTS
    emo_opts = " / ".join(voice_emotions) if voice_emotions else ""
    emo_field = (f', "emotion": "从 [{emo_opts}] 选一个最贴合的说话语气"'
                 if emo_opts else "")
    # 立绘表情字段：与 emotion（语气）独立，决定画面神态。空立绘库时不要求。
    face_opts = " / ".join(face_emotions) if face_emotions else ""
    face_field = (f', "face": "必填，从 [{face_opts}] 选一个最贴合此刻神态的立绘表情"'
                  if face_opts else "")
    # 情绪状态增量：这次内心活动让你内在情绪如何变化（喂给 apply_delta，影响主动行为/记忆）。
    # 区别于上面的 emotion（那是说话语气/TTS），这里是心理状态五维的细微增量。
    delta_field = (
        ', "emotion_delta": {可选，只写有明显变化的维度，每个是小幅增量浮点数，范围约 -0.3~0.3：'
        '"valence":愉悦度变化(开心+/难过-), "arousal":唤醒度变化(兴奋+/平静-), '
        '"attachment":依恋度变化(更亲近+), "worry":担心变化(更担心+/放心-)}'
    )

    # 可联想的素材：长期记忆 + 早先聊天摘要
    material = []
    if earlier_summary:
        material.append(f"【早先你们聊过】{earlier_summary}")
    if memories:
        material.append("【你能想起关于ta的事】\n" + "\n".join(f"- {m}" for m in memories))
    # 近期内心活动（未必说出口）：让这次思考延续上几次的思绪，而非每次从零重想。
    # 来源是最近几次主动思考的结局摘要（想了没说/看了一眼/开口说了什么）。
    if recent_inner:
        material.append("【你最近的内心活动（按时间顺序，未必都说出口了）】\n"
                        + "\n".join(f"- {s}" for s in recent_inner))
    material_block = ("\n\n" + "\n\n".join(material)) if material else ""

    ctx_line = f"现在是{time_context}。" if time_context else ""
    seed_line = f"一个可以由头：{think_seed}。" if think_seed else ""

    lang_block = f"\n\n{lang_rules}" if lang_rules else ""

    monologue = (
        f"[系统提示·内心独白]\n"
        f"以下不是{user_name}的消息，而是你自己的一次内心活动。\n"
        f"{ctx_line}距上次互动已过去 {elapsed_desc}。触发原因：{trigger_reason}。\n"
        f"你此刻的情绪：{emotion_summary}。"
        f"{material_block}\n\n"
        f"你可以自由地想：回忆你们聊过的事、好奇ta最近怎么样、由某条记忆联想开去、"
        f"冒出一个想分享的念头、或单纯想念ta。{seed_line}\n"
        f"唯一的约束：你现在**看不到也听不到**{user_name}此刻的样子，"
        f"所以**不要编造你正在观察到的画面**（例如「看到你打哈欠」「你好像在忙」这种当下感知）——"
        f"那是幻觉。但回忆过去、联想、好奇都没问题，那是正常的思绪。\n\n"
        f"**要不要开口，还要看对方此刻能不能收到。** 如果从近期对话看，"
        f"{user_name}明确表示暂时联系不上、要离开一段时间、或最近一直没回你"
        f"（你多次主动开口都没等到回应），那此刻开口对方也收不到——选 silent。"
        f"此时 thought 写你此刻真实的内心活动（比如想念、猜测对方在做什么、"
        f"决定等ta回来再说），**不要**把本来想说出口的话写进 thought。\n\n"
        f"**不要只凭时间或空场景猜测对方回来了。** 如果画面里没有出现人，"
        f"只有过道、门、家具、光线变化或时间推测，就不要说「欢迎回来/おかえり」。"
        f"但如果画面已经明确出现正在进门、刚到家或带着归来迹象的人，"
        f"可以结合你们的相处背景自然地欢迎，不必先追问或核验对方身份；"
        f"证据仍然太弱时，可以再看一眼或选 silent。\n\n"
        f"请先在心里真实地想一想（thought **必须**写出有具体内容的内心活动，至少一两句，"
        f"别只说'好久没聊有点想ta'，更不要把 thought 留空或只写一个决定），"
        f"再决定要不要开口。{lang_block}\n\n"
        f"**不要重复**：参考你最近的内心活动和近期对话——如果最近已经主动开口说过类似的话"
        f"（哪怕措辞不同、意思相近），这次换种说法或换个话题，不要把同一句关心再说一遍。\n\n"
        f"严格输出如下 JSON（不要多余文字）：\n"
        + (
            f'{{"thought": "你此刻真实、具体的内心想法（不能为空）", '
            f'"action": "speak、silent 或 look(若好奇想看看他在干嘛)", '
            f'"text": "若 speak 则写你要说的话(用平时口吻)，其他情况留空"{emo_field}{face_field}{delta_field}}}'
            if can_look else
            f'{{"thought": "你此刻真实、具体的内心想法（不能为空）", '
            f'"action": "speak 或 silent", '
            f'"text": "若 speak 则写你要说的话(用平时口吻)，若 silent 则留空"{emo_field}{face_field}{delta_field}}}'
        )
    )
    return monologue


def _build_proactive_cot_instruction(
    trigger_reason: str, emotion_summary: str, chat_history: list[dict], elapsed_desc: str,
    memories: list[str] | None, earlier_summary: str, time_context: str, think_seed: str,
    recent_inner: list[str] | None, can_look: bool, lang_rules: str = "",
    face_emotions: list[str] | None = None,
    voice_emotions: list[str] | None = None,
) -> str:
    """构建思维链（CoT）格式的内心独白指令。

    Args:
        lang_rules: 按字段的语言规则（思考/决策标记/要说的话各自语言），由
            _proactive_lang_rules 生成，拼在指令末尾。
        face_emotions: 立绘库可用表情档（库驱动），非空时要求决策行附 [face:表情]。
        voice_emotions: 语音库可用语气档（库驱动），非空时要求 speak 时话里带 <emo:语气>。
    """
    # 可联想的素材：长期记忆 + 早先聊天摘要
    material = []
    if earlier_summary:
        material.append(f"【早先你们聊过】{earlier_summary}")
    if memories:
        material.append("【你能想起关于ta的事】\n" + "\n".join(f"- {m}" for m in memories))
    # 近期内心活动
    if recent_inner:
        material.append("【你最近的内心活动（按时间顺序，未必都说出口了）】\n"
                        + "\n".join(f"- {s}" for s in recent_inner))
    material_block = ("\n\n" + "\n\n".join(material)) if material else ""

    ctx_line = f"现在是{time_context}。" if time_context else ""
    seed_line = f"一个可以由头：{think_seed}。" if think_seed else ""

    lang_block = f"\n\n{lang_rules}" if lang_rules else ""

    # 立绘表情标记说明（库驱动，空立绘库则不要求）
    face_opts = " / ".join(face_emotions) if face_emotions else ""
    face_block = (
        f"\n\n你在画面上的立绘表情用 `[face:表情]` 标注（从 [{face_opts}] 选一个最贴合此刻神态的），"
        f"与说话语气独立。**每次独白都必须带 `[face:表情]`，不能漏**。"
        f"**speak 时**把 `[face:表情]` 放在决策标记同一行末尾，"
        f"如 `[决定：开口] \"要说的话\" [face:得意]`；silent 时也标，"
        f"如 `[决定：沉默] [face:思考]`。"
        if face_opts else ""
    )
    # 语音语气标签说明（库驱动，空语音库则不要求）
    emo_opts = " / ".join(voice_emotions) if voice_emotions else ""
    emo_block = (
        f"\n\n你说话的语气用 `<emo:语气>` 标注（从 [{emo_opts}] 选一个最贴合的），"
        f"决定参考音色，与立绘表情独立。**speak 时必须带 `<emo:语气>`，不能漏**，"
        f"放在要说的话开头，如 `[决定：开口] <emo:happy>\"要说的话\" [face:得意]`。"
        f"silent 时不用带 emo。"
        if emo_opts else ""
    )

    monologue = (
        f"[系统提示·内心独白]\n"
        f"以下不是用户的消息，而是你自己的一次内心活动。\n"
        f"{ctx_line}距上次互动已过去 {elapsed_desc}。触发原因：{trigger_reason}。\n"
        f"你此刻的情绪：{emotion_summary}。"
        f"{material_block}\n\n"
        f"请自由地、真实地思考：\n"
        f"- 回忆你们聊过的事\n"
        f"- 好奇ta最近怎么样\n"
        f"- 从某条记忆联想开去\n"
        f"- 冒出一个想分享的念头\n"
        f"- 或单纯想念ta\n"
        f"{seed_line}"
        f"{lang_block}\n\n"
        f"**唯一的约束**：你现在看不到也听不到用户此刻的样子，"
        f"所以不要编造你正在观察到的画面（如「看到你打哈欠」「你好像在忙」）——"
        f"那是幻觉。但回忆过去、联想、好奇都可以，那是正常的思绪。\n\n"
        f"**要不要开口，还要看对方此刻能不能收到。** 如果从近期对话看，"
        f"用户明确表示暂时联系不上、要离开一段时间、或最近一直没回你"
        f"（你多次主动开口都没等到回应），那此刻开口对方也收不到——用 `[决定：沉默]`。"
        f"此时思考过程写你此刻真实的内心活动（比如想念、猜测对方在做什么、"
        f"决定等ta回来再说），**不要**把本来想说出口的话写进思考过程。\n\n"
        f"**不要只凭时间或空场景猜测用户回来了。** 如果画面里没有出现人，"
        f"只有过道、门、家具、光线变化或时间推测，就不要说「欢迎回来/おかえり」。"
        f"但如果画面已经明确出现正在进门、刚到家或带着归来迹象的人，"
        f"可以结合你们的相处背景自然地欢迎，不必先追问或核验对方身份；"
        f"证据仍然太弱时，可以再看一眼或用 `[决定：沉默]`。\n\n"
        f"**重要**：决策标记之前**必须**先写出至少一两句真实、具体的思考过程，"
        f"不能一上来就直接给决策标记（那样意识流里就只剩一个空决定，没有任何内心活动）。\n\n"
        f"**不要重复**：参考【你最近的内心活动】和近期对话——如果你最近已经主动开口说过"
        f"类似的话（哪怕措辞不同、意思相近），这次就**换种说法或换个话题**，不要把同一句"
        f"关心再说一遍。重复说一样的话会让对方觉得敷衍。\n\n"
        + (f"{face_block.lstrip()}\n\n" if face_block else "")
        + (f"{emo_block.lstrip()}\n\n" if emo_block else "")
        + f"思考完毕后，另起一行，用以下格式之一表达你的决定：\n"
        + (
            f"[决定：开口] \"要说的话\"\n"
            f"[决定：沉默]\n"
            f"[决定：看看他]  （如果好奇想看看ta在干嘛）\n\n"
            if can_look else
            f"[决定：开口] \"要说的话\"\n"
            f"[决定：沉默]\n\n"
        )
        + f"示例：\n"
        f"嗯...现在都晚上10点了。\n"
        f"他上次说那个项目这周要交，应该快截止了吧。\n"
        f"这个点还没睡，会不会在加班？\n"
        f"我记得他是那种会熬夜赶工的性格...\n"
        f"要不要问一句呢？\n"
        f"算了，还是问一下吧。\n"
        f"[决定：开口] \"项目进展怎么样了？还在忙吗？\""
    )
    return monologue


# ─────────────────── 世界书管理 (架构文档 6.7) ──────────────────────────────

class WorldInfoManager:
    """加载并合并角色绑定/用户人设绑定/全局世界书，运行激活算法。

    绑定来源 (合并去重):
      - 角色卡 extensions.world      → 外部世界书文件名
      - 角色卡 character_book        → 内嵌世界书 (随卡走)
      - 用户人设 bound_worldinfo     → 专属外部世界书
      - config global_worldbooks[]   → 全局世界书 (多角色共享)
    """

    def __init__(self, config, card: "CharacterCard"):
        from .worldinfo import WorldBook, WorldInfoEntry  # 延迟导入避免环依赖
        self._books = []
        wb_dir = config.project_root / "data" / "worldbooks"

        def _load_external(name: str):
            if not name:
                return
            p = wb_dir / f"{name}.json"
            if p.exists():
                try:
                    self._books.append(WorldBook.load(p))
                except Exception as e:  # noqa: BLE001
                    log.warning("加载 %s 失败: %s", name, e)

        # 角色内嵌世界书
        cb = card.extensions.get("character_book") if card.extensions else None
        if cb and isinstance(cb, dict):
            items = cb.get("entries", [])
            items = items.values() if isinstance(items, dict) else items
            entries = [WorldInfoEntry.from_dict(e) for e in items]
            self._books.append(WorldBook(entries, name="character_book"))
        # 角色外部世界书引用
        if card.extensions:
            _load_external(card.extensions.get("world", ""))
            for wb_name in (card.extensions.get("world_additional") or []):
                _load_external(wb_name)
        # 用户人设绑定
        _load_external(config.get("user_persona.bound_worldinfo") or "")
        # 全局世界书
        for name in (config.get("global_worldbooks") or []):
            _load_external(name)

        self._budget = config.get("worldinfo.budget_chars", 2000)

    def has_books(self) -> bool:
        return bool(self._books)

    def activate(self, scan_text: str, round_no: int = 0) -> dict:
        """合并所有书的激活结果。"""
        from .worldinfo import activate as _activate, WorldBook
        if not self._books:
            return {"before": [], "after": [], "depth": []}
        # 合并所有书的条目到一本临时书，统一激活+预算
        all_entries = []
        for b in self._books:
            all_entries.extend(b.entries)
        merged = WorldBook(all_entries, name="merged")
        return _activate(merged, scan_text, budget_chars=self._budget, round_no=round_no)
