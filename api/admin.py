"""管理平台后端 (架构文档 8.1/8.2/8.4)。

FastAPI，仅监听 loopback 127.0.0.1:8080。
启动：python -m api.admin
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import httpx
import yaml
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from core.logger import get_logger, read_logs

log = get_logger("admin")

_ROOT = Path(__file__).resolve().parent.parent
_DATA = _ROOT / "data"
_CONFIG_PATH = _DATA / "config.yaml"
_UI_DIR = _ROOT / "ui"
_MASK = "********"
_SECRET_HINTS = ("api_key", "key", "token", "secret", "password")

# 当前运行中的 orchestrator enqueue（由 main.py 注入，用于聊天接口）
_enqueue_fn = None
# 角色切换回调（由 main.py 注入，用于切角色时通知 orchestrator）
_switch_char_fn = None
# 删除角色记忆回调（由 main.py 注入，删角色时清 mem0）
_delete_memory_fn = None
# 运行中的内存 Config 对象（由 main.py 注入）。改配置后调 .reload() 把磁盘新值
# 灌回这个对象，运行中的"现读"模块下次 get() 即拿到新值，无需重启。
_live_config = None
# 热更钩子 {"gatekeeper": fn, "vision": fn, ...}，每个 fn(cfg) 把新配置同步到
# 构造时缓存了值的活模块实例上。reload 后逐个触发。
_refreshers: dict = {}


def set_enqueue(fn) -> None:
    global _enqueue_fn
    _enqueue_fn = fn


def set_switch_character(fn) -> None:
    global _switch_char_fn
    _switch_char_fn = fn


_reload_card_fn = None


def set_reload_card(fn) -> None:
    """注册角色卡热重载回调（orchestrator.reload_card）。"""
    global _reload_card_fn
    _reload_card_fn = fn


def set_delete_memory(fn) -> None:
    global _delete_memory_fn
    _delete_memory_fn = fn


def set_live_config(cfg, refreshers: dict | None = None) -> None:
    """注入运行中的内存 Config + 热更钩子。put_config 保存后用它们让改动即时生效。"""
    global _live_config, _refreshers
    _live_config = cfg
    _refreshers = refreshers or {}


def _char_dir(char_name: str | None = None) -> Path:
    cfg = _load_raw_config()
    name = char_name or (cfg.get("character") or {}).get("name", "默认")
    return _DATA / "characters" / name


def _is_secret(dotted: str) -> bool:
    leaf = dotted.rsplit(".", 1)[-1].lower()
    return any(h in leaf for h in _SECRET_HINTS)


def _load_raw_config() -> dict:
    if not _CONFIG_PATH.exists():
        return {}
    with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _save_raw_config(data: dict) -> None:
    with open(_CONFIG_PATH, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False, default_flow_style=False)


def _mask_secrets(node: Any, path: str = "") -> Any:
    if isinstance(node, dict):
        return {k: _mask_secrets(v, f"{path}.{k}" if path else k) for k, v in node.items()}
    if isinstance(node, list):
        return [_mask_secrets(v, path) for v in node]
    if _is_secret(path) and node not in (None, "", []):
        return _MASK
    return node


def _merge_unmask(new: Any, old: Any, path: str = "") -> Any:
    if isinstance(new, dict):
        merged = {}
        for k, v in new.items():
            ov = old.get(k) if isinstance(old, dict) else None
            merged[k] = _merge_unmask(v, ov, f"{path}.{k}" if path else k)
        return merged
    if _is_secret(path) and new == _MASK:
        return old
    return new


def _read_jsonl(path: Path, limit: int = 300) -> list[dict]:
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    out = []
    for ln in lines[-limit:]:
        try:
            out.append(json.loads(ln))
        except (json.JSONDecodeError, ValueError):
            continue
    return out


def _safe_name(name: str) -> str:
    if "/" in name or "\\" in name or ".." in name or not name:
        raise HTTPException(400, "非法名称")
    return name


app = FastAPI(title="newTouch 管理平台", docs_url=None, redoc_url=None)


# ── 配置读写 ──────────────────────────────────────────────────
@app.get("/api/config")
def get_config() -> dict:
    return _mask_secrets(_load_raw_config())


@app.get("/api/config/schema")
def get_config_schema() -> list:
    """结构化配置表单 schema，供前端生成控件。"""
    return [
        {"section": "LLM", "key": "modules.llm", "fields": [
            {"k": "provider", "label": "Provider", "type": "select",
             "options": ["anthropic", "deepseek", "openai"], "restart": True},
            {"k": "model", "label": "模型名", "type": "text", "restart": True},
            {"k": "api_key", "label": "API Key", "type": "password", "restart": True},
            {"k": "base_url", "label": "Base URL", "type": "text", "restart": True},
            {"k": "temperature", "label": "Temperature", "type": "number", "min": 0, "max": 2, "step": 0.05, "restart": True},
            {"k": "max_tokens", "label": "Max Tokens", "type": "number", "min": 256, "max": 8192, "restart": True},
        ]},
        {"section": "TTS", "key": "modules.tts", "fields": [
            {"k": "enabled", "label": "启用 TTS", "type": "toggle"},
            {"k": "endpoint", "label": "GPT-SoVITS 地址", "type": "text"},
            {"k": "ref_audio_path", "label": "参考音频路径", "type": "text"},
            {"k": "prompt_text", "label": "参考文本（参考音频对应文字）", "type": "text"},
            {"k": "prompt_lang", "label": "参考音频语言", "type": "select",
             "options": ["zh", "en", "ja", "ko", "yue", "auto"]},
            {"k": "text_lang", "label": "生成文本语言", "type": "select",
             "options": ["zh", "en", "ja", "ko", "yue", "auto"]},
            {"k": "text_split_method", "label": "切分方式", "type": "select",
             "options": [
                 {"value": "cut0", "label": "cut0 - 不切（推荐，已预切句）"},
                 {"value": "cut1", "label": "cut1 - 凑四句一切"},
                 {"value": "cut2", "label": "cut2 - 凑50字一切"},
                 {"value": "cut3", "label": "cut3 - 按中文句号切"},
                 {"value": "cut4", "label": "cut4 - 按英文句号切"},
                 {"value": "cut5", "label": "cut5 - 按标点符号切"},
             ],
             "help": "newTouch 已在外部按标点预切句再逐句发，推荐 cut0 不让服务端二次切"},
            {"k": "speed_factor", "label": "语速", "type": "number", "min": 0.5, "max": 2, "step": 0.1},
            {"k": "top_k", "label": "Top K", "type": "number", "min": 1, "max": 20},
            {"k": "top_p", "label": "Top P", "type": "number", "min": 0.1, "max": 1, "step": 0.05},
            {"k": "temperature", "label": "Temperature", "type": "number", "min": 0.1, "max": 2, "step": 0.05},
            {"k": "repetition_penalty", "label": "重复惩罚", "type": "number", "min": 1, "max": 2, "step": 0.05},
            {"k": "batch_size", "label": "Batch Size", "type": "number", "min": 1, "max": 10},
            {"k": "fragment_interval", "label": "片段间隔(s)", "type": "number", "min": 0, "max": 1, "step": 0.05},
        ]},
        {"section": "STT", "key": "modules.stt", "fields": [
            {"k": "provider", "label": "STT 引擎", "type": "select",
             "options": [
                 {"value": "faster-whisper", "label": "faster-whisper（多语言通用）"},
                 {"value": "funasr", "label": "FunASR SenseVoice（中文专优，更快）"},
             ], "restart": True,
             "help": "faster-whisper 多语言成熟；funasr(SenseVoiceSmall) 中文识别更准更快、小智同款，首次需下模型"},
            {"k": "model", "label": "Whisper 模型大小", "type": "select",
             "options": ["tiny", "base", "small", "medium", "large-v3"], "restart": True,
             "help": "仅 faster-whisper 用：模型越大越准越慢，CPU 建议 small"},
            {"k": "funasr_model", "label": "FunASR 模型", "type": "text", "restart": True,
             "help": "仅 funasr 用：默认 iic/SenseVoiceSmall"},
            {"k": "device", "label": "设备", "type": "select", "options": ["cpu", "cuda"], "restart": True},
            {"k": "language", "label": "识别语言", "type": "select",
             "options": ["zh", "en", "ja", "ko", "yue", "auto"], "restart": True,
             "help": "auto 自动检测；指定语言更准更快"},
            {"k": "use_itn", "label": "逆文本规整(ITN)", "type": "toggle", "restart": True,
             "help": "仅 funasr 用：开启则输出带标点/数字规整"},
            {"k": "keep_emotion_tags", "label": "保留情绪标签", "type": "toggle", "restart": True,
             "help": "仅 funasr 用：保留 SenseVoice 的 <|情绪|> 标签让主脑感知用户情绪；关则洗成纯文本"},
            {"k": "hf_endpoint", "label": "HF 镜像", "type": "text", "restart": True,
             "help": "模型下载镜像，国内填 https://hf-mirror.com"},
            {"k": "model_cache_dir", "label": "模型缓存目录", "type": "text", "restart": True,
             "help": "STT 模型缓存位置（相对项目根，默认 data/stt_cache）。ModelScope 在此建 models/{model_id}/"},
        ]},
        {"section": "VLM（视觉）", "key": "modules.vlm", "fields": [
            {"k": "provider", "label": "Provider", "type": "select",
             "options": ["anthropic", "openai"], "restart": True},
            {"k": "model", "label": "模型名", "type": "text", "restart": True},
            {"k": "api_key", "label": "API Key", "type": "password", "restart": True},
            {"k": "base_url", "label": "Base URL", "type": "text", "restart": True},
            {"k": "caption_max_tokens", "label": "描述最大Token", "type": "number",
             "min": 64, "max": 2048, "step": 64},
        ]},
        {"section": "感知层", "key": "perception", "fields": [
            {"k": "audio.enabled", "label": "启用麦克风", "type": "toggle", "restart": True},
            {"k": "vision.enabled", "label": "启用摄像头", "type": "toggle"},
            {"k": "vision.camera_index", "label": "摄像头编号", "type": "number", "min": 0, "max": 9, "restart": True},
            {"k": "vision.frame_diff_threshold", "label": "帧差阈值（普通变化）", "type": "number",
             "min": 0.05, "max": 0.5, "step": 0.05,
             "help": "灰度差超此值才调用 VLM，过低会频繁调用（默认 0.15）"},
            {"k": "vision.significant_threshold", "label": "帧差阈值（显著变化）", "type": "number",
             "min": 0.1, "max": 1.0, "step": 0.05,
             "help": "超此值触发 LLM 智能判断是否开口，低于此值只记录 caption（默认 0.30）"},
            {"k": "vision.min_caption_interval_s", "label": "VLM 调用最小间隔(秒)", "type": "number",
             "min": 5, "max": 60,
             "help": "两次 VLM caption 的最小间隔，节省调用（默认 10）"},
            {"k": "vision.min_check_interval_s", "label": "视觉判断最小间隔(秒)", "type": "number",
             "min": 10, "max": 300,
             "help": "LLM 判断是否开口的最小间隔，防刷屏（默认 60）"},
        ]},
        {"section": "主动行为", "key": "proactive", "fields": [
            {"k": "enabled", "label": "主动发言", "type": "toggle"},
            {"k": "min_interval_seconds", "label": "最小间隔(秒)", "type": "number", "min": 30},
            {"k": "hourly_cap", "label": "每小时上限", "type": "number", "min": 1, "max": 20},
            {"k": "quiet_start", "label": "勿扰开始", "type": "text"},
            {"k": "quiet_end", "label": "勿扰结束", "type": "text"},
            {"k": "loneliness_threshold", "label": "孤独感阈值", "type": "number",
             "min": 0, "max": 1, "step": 0.05},
            {"k": "clinginess", "label": "黏人度（全局默认）", "type": "number",
             "min": 0, "max": 1, "step": 0.05,
             "help": "被冷落时该催还是该退。0=识趣退避 0.5=中性 1=焦虑追问。角色卡可单独覆盖"},
            {"k": "ignored_grace_seconds", "label": "未回应宽限(秒)", "type": "number",
             "min": 30, "step": 30,
             "help": "主动说完至少等这么久没回应才算\"被无视\"（触发担心递增/拉长冷却）。不应短于心跳间隔"},
            {"k": "recent_inner_max", "label": "近期内心活动条数", "type": "number",
             "min": 0, "max": 20,
             "help": "每次主动思考能看到自己最近几次想了什么(说了/没说/看了)，使思绪连贯不每次从零重想。0=关闭"},
        ]},
        {"section": "记忆（mem0）", "key": "memory", "fields": [
            {"k": "chat_history_window", "label": "短期对话窗口（条）", "type": "number",
             "min": 10, "max": 200, "step": 10,
             "help": "user+assistant 各算一条，即 N/2 轮对话。同时决定重启时回填多少条（续聊）"},
            {"k": "compress_enabled", "label": "短期窗口增量压缩", "type": "toggle",
             "help": "开：窗口溢出时把最旧一批 LLM 浓缩成「早先聊天摘要」保留脉络；关：直接丢最旧"},
            {"k": "compress_batch_size", "label": "压缩批大小（条）", "type": "number",
             "min": 4, "max": 40, "step": 2,
             "help": "每溢出这么多条才压一次（非每轮，省 LLM 调用）。越大压得越少越省、但摘要颗粒越粗"},
            {"k": "enabled", "label": "启用 mem0 记忆", "type": "toggle"},
            {"k": "infer", "label": "LLM 抽取记忆要点", "type": "toggle",
             "help": "开：mem0 原生抽取+去重（需 LLM 支持 JSON 模式；火山 coding 端点须在控制台绑定支持 JSON 的模型）。关时看 extract 开关"},
            {"k": "extract", "label": "续写模式抽取（推荐）", "type": "toggle",
             "help": "infer 关时生效：用续写模式让 LLM 把对话提炼成要点再存，不依赖 JSON 模式，火山 coding 端点直接可用。两者都关则存原始摘要不调 LLM"},
            {"k": "llm_model", "label": "记忆抽取 LLM 模型", "type": "text",
             "help": "mem0 用它从对话里抽取记忆要点。留空走 gpt-4o-mini。火山填模型名或 endpoint id"},
            {"k": "api_key", "label": "记忆 LLM Key", "type": "password",
             "help": "留空则复用主 LLM 的 Key"},
            {"k": "base_url", "label": "记忆 LLM Base URL", "type": "text",
             "help": "留空则复用主 LLM 的 Base URL"},
            {"k": "embedding_model", "label": "Embedding 模型", "type": "text",
             "help": "向量化模型。火山如 doubao-embedding-text-240715 或对应 endpoint id"},
            {"k": "embed_api_key", "label": "Embedding Key", "type": "password",
             "help": "留空则复用记忆 LLM Key"},
            {"k": "embed_base_url", "label": "Embedding Base URL", "type": "text",
             "help": "留空则复用记忆 LLM Base URL。注意火山 embedding 在 /api/v3，非 /api/coding/v3"},
            {"k": "reactive_auto_recall", "label": "反应路径自动注入记忆", "type": "toggle",
             "help": "开：每句用户发言自动 recall 注入 system（现状）；关：不自动注入，靠 LLM 调 memory_search 工具。改完即生效"},
            {"k": "tool_enabled", "label": "memory_search 工具", "type": "toggle", "restart": True,
             "help": "注册 memory_search 工具给 LLM 自主调用（觉得自动注入不够时可再查）。与上面独立，可叠加/互替。改后需重启"},
        ]},
        {"section": "角色 / 预设（切换即时生效）", "key": "character", "fields": [
            {"k": "name", "label": "当前角色", "type": "char_select"},
            {"k": "preset_enabled", "label": "启用预设", "type": "preset_toggle"},
            {"k": "current_preset", "label": "当前预设", "type": "preset_select"},
        ]},
        {"section": "用户人设", "key": "user_persona", "fields": [
            {"k": "name", "label": "你的名字", "type": "text"},
            {"k": "description", "label": "你的描述", "type": "textarea"},
        ]},
        {"section": "Token 密集型优化（可选）", "key": "token_intensive", "fields": [
            {"k": "_info", "label": "说明", "type": "info",
             "help": "这些优化显著提升体验，但会增加 API 调用成本。详见 docs/Token密集型优化方案.md。每实现一个功能会添加对应开关。"},
            {"k": "proactive_cot", "label": "主动思考链（CoT）", "type": "toggle",
             "help": "完整推理过程替代结构化 JSON，思考更深入更像人。成本：每次主动思考 200→1000 tokens"},
            {"k": "memory_query_generation", "label": "记忆检索 LLM 生成 query", "type": "toggle",
             "help": "主动思考时 LLM 生成精准检索问题（主动回想路线 B）。成本：每次主动思考 +100 tokens"},
        ]},
        {"section": "立绘浮窗", "key": "sprite", "fields": [
            {"k": "enabled", "label": "启用桌面立绘浮窗", "type": "toggle", "restart": True,
             "help": "开则启动 PyQt6 透明置顶浮窗，按 <face:表情> 标签切换差分立绘。需先 pip install PyQt6。立绘库放 data/characters/{角色}/sprites/sprites.json。改后需重启"},
            {"k": "host", "label": "广播地址", "type": "text", "restart": True,
             "help": "浮窗通信地址，默认 127.0.0.1（本机）"},
            {"k": "port", "label": "广播端口", "type": "number", "min": 1024, "max": 65535, "restart": True,
             "help": "主程序 TCP server 端口，浮窗 client 连此端口收表情。改后需重启"},
        ]},
        {"section": "日志", "key": "logging", "fields": [
            {"k": "enabled", "label": "写日志文件", "type": "toggle",
             "help": "开启则写 data/logs/system.log（按日轮转）；关闭只输出到控制台。保存后下次写日志即生效。"},
            {"k": "level", "label": "日志级别", "type": "select",
             "options": ["DEBUG", "INFO", "WARNING", "ERROR"],
             "help": "DEBUG 最详细（含 LLM/工具细节），ERROR 只记错误。保存即生效。"},
            {"k": "dir", "label": "日志目录", "type": "text",
             "help": "相对项目根。主文件 system.log。改后新日志写到新目录（需重启旧的才释放）。"},
            {"k": "backup_count", "label": "保留天数", "type": "number", "min": 1, "max": 90, "step": 1,
             "help": "按日轮转保留多少天的历史日志文件。"},
        ]},
    ]


class ConfigUpdate(BaseModel):
    config: dict


@app.put("/api/config")
def put_config(body: ConfigUpdate) -> dict:
    old = _load_raw_config()
    old_char = (old.get("character") or {}).get("name")
    merged = _merge_unmask(body.config, old)
    _save_raw_config(merged)
    # 把磁盘新值灌回运行中的内存 Config，"现读"模块下次 get() 即生效（无需重启）。
    if _live_config is not None:
        try:
            _live_config.reload()
        except Exception as e:  # noqa: BLE001
            log.error("配置热重载失败: %s", e)
        # 触发热更钩子：把新值同步到构造时缓存了值的活模块（gatekeeper/vision 等）
        for name, fn in _refreshers.items():
            try:
                fn(_live_config)
            except Exception as e:  # noqa: BLE001
                log.warning("刷新 %s 失败: %s", name, e)
    # 检测角色名变化 → 通知 orchestrator 热切换（switch_character 读的是已 reload 的 cfg）
    new_char = (merged.get("character") or {}).get("name")
    if new_char and new_char != old_char and _switch_char_fn:
        _switch_char_fn(new_char)
    return {"ok": True}


# ── 权限 ──────────────────────────────────────────────────────
class PermissionUpdate(BaseModel):
    permissions: dict


@app.put("/api/permissions")
def put_permissions(body: PermissionUpdate) -> dict:
    cfg = _load_raw_config()
    cfg.setdefault("ai_permissions", {})
    for k, v in body.permissions.items():
        cfg["ai_permissions"][k] = bool(v)
    _save_raw_config(cfg)
    # 权限是 self_config 工具每次现读的（ai_permissions.*），热重载即生效，无需重启
    if _live_config is not None:
        try:
            _live_config.reload()
        except Exception as e:  # noqa: BLE001
            log.error("权限热重载失败: %s", e)
    return {"ok": True, "permissions": cfg["ai_permissions"]}


# ── 观测面板（路径已按角色隔离）──────────────────────────────────
@app.get("/api/observe/consciousness")
def observe_consciousness(limit: int = 200, action: str | None = None,
                          char: str | None = None) -> list[dict]:
    items = _read_jsonl(_char_dir(char) / "consciousness.jsonl", limit=limit)
    if action == "monologue":
        # 仅内心独白：有 thought 的条目（过滤掉心跳/旁听等无独白的噪音）
        items = [i for i in items if (i.get("thought") or "").strip()]
    elif action:
        items = [i for i in items if i.get("action") == action]
    return items


@app.get("/api/observe/emotion")
def observe_emotion(limit: int = 200, char: str | None = None) -> list[dict]:
    items = _read_jsonl(_char_dir(char) / "consciousness.jsonl", limit=limit)
    return [{"ts": i.get("ts"), **(i.get("emotion") or {})}
            for i in items if i.get("emotion")]


@app.get("/api/observe/audit")
def observe_audit(limit: int = 200, char: str | None = None) -> list[dict]:
    return _read_jsonl(_char_dir(char) / "audit.log", limit=limit)


@app.get("/api/logs")
def get_logs(level: str | None = None, module: str | None = None,
             limit: int = 300, since: str | None = None) -> list[dict]:
    """读取系统日志（newest-first）。level=级别以上，module=子串，since=ISO 前缀。"""
    if _live_config is None:
        return []
    limit = max(1, min(int(limit), 2000))
    try:
        return read_logs(_live_config, level=level, module=module, limit=limit, since=since)
    except Exception as e:  # noqa: BLE001
        log.error("读取日志失败: %s", e)
        return []


@app.get("/api/observe/state")
def observe_state(char: str | None = None) -> dict:
    p = _char_dir(char) / "state.json"
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, ValueError):
        return {}


# ── 聊天记录 + 发送消息 ────────────────────────────────────────
def _chat_history_data(limit: int = 100, char: str | None = None) -> list[dict]:
    """读聊天历史原始数据（返回 list，供路由包装 / 测试 / 内部复用）。"""
    return _read_jsonl(_char_dir(char) / "chat_history.jsonl", limit=limit)


@app.get("/api/chat/history")
def get_chat_history(limit: int = 100, char: str | None = None):
    """聊天历史：user 发言 + assistant speak（含主动发言）。

    显式禁缓存：这是高频轮询端点，浏览器若缓存 GET 会导致 TTS 播放期间
    新写入的 assistant 行刷不出来（启发式缓存命中旧响应）。
    返回 JSONResponse 以带缓存头；纯数据读取见 _chat_history_data()。
    """
    from fastapi.responses import JSONResponse
    return JSONResponse(
        content=_chat_history_data(limit=limit, char=char),
        headers={"Cache-Control": "no-store, no-cache, must-revalidate",
                 "Pragma": "no-cache", "Expires": "0"},
    )


class ChatSend(BaseModel):
    text: str


@app.post("/api/chat/send")
async def send_chat(body: ChatSend) -> dict:
    """管理平台文本聊天：把消息投入 orchestrator 队列（等同于文本输入）。"""
    if not _enqueue_fn:
        raise HTTPException(503, "orchestrator 未连接，请先启动 main.py")
    from core.events import Event, EventType, EventPriority
    await _enqueue_fn(Event(
        priority=EventPriority.URGENT,
        type=EventType.USER_SPEECH,
        payload={"text": body.text.strip()},
    ))
    return {"ok": True}


# ── 语音库（多文件 + 角色绑定）────────────────────────────────
_VOICES_DIR = _DATA / "voices"
_VOICE_LIB_RESERVED = frozenset(("gpt_weights", "sovits_weights"))


def _voice_lib_path(name: str) -> Path:
    return _VOICES_DIR / f"{_safe_name(name)}.json"


def _read_voice_lib(name: str) -> dict:
    p = _voice_lib_path(name)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, ValueError, OSError):
        return {}


@app.get("/api/voice-libs")
def list_voice_libs() -> list[str]:
    """列出所有语音库名（文件名去 .json 后缀，按字母排序）。"""
    if not _VOICES_DIR.exists():
        return []
    return sorted(p.stem for p in _VOICES_DIR.iterdir()
                  if p.is_file() and p.suffix == ".json")


@app.get("/api/voice-libs/{name}")
def get_voice_lib(name: str) -> dict:
    data = _read_voice_lib(name)
    if not data and not _voice_lib_path(name).exists():
        raise HTTPException(404, "语音库不存在")
    return data


@app.post("/api/voice-libs/{name}")
def create_voice_lib(name: str) -> dict:
    p = _voice_lib_path(name)
    if p.exists():
        raise HTTPException(400, "语音库已存在")
    _VOICES_DIR.mkdir(parents=True, exist_ok=True)
    p.write_text(
        json.dumps({"gpt_weights": "", "sovits_weights": "", "emotions": {}},
                   ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return {"ok": True}


class VoiceLibUpdate(BaseModel):
    lib: dict


@app.put("/api/voice-libs/{name}")
def put_voice_lib(name: str, body: VoiceLibUpdate) -> dict:
    _VOICES_DIR.mkdir(parents=True, exist_ok=True)
    _voice_lib_path(name).write_text(
        json.dumps(body.lib, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return {"ok": True}


@app.delete("/api/voice-libs/{name}")
def delete_voice_lib(name: str) -> dict:
    if _safe_name(name) == "library":
        raise HTTPException(400, "library 是兜底库，不允许删除")
    p = _voice_lib_path(name)
    if not p.exists():
        raise HTTPException(404, "语音库不存在")
    p.unlink()
    return {"ok": True}


class ModelSwitchBody(BaseModel):
    gpt_weights: str = ""
    sovits_weights: str = ""


@app.post("/api/voices/switch-model")
async def switch_voice_model(body: ModelSwitchBody) -> dict:
    """转发 GPT-SoVITS 切模型接口（/set_gpt_weights, /set_sovits_weights）。"""
    raw = _load_raw_config()
    endpoint = ((raw.get("modules") or {}).get("tts") or {}).get(
        "endpoint", "http://127.0.0.1:9880"
    )
    results = {}
    try:
        async with httpx.AsyncClient(timeout=8.0) as client:
            if body.gpt_weights:
                r = await client.get(f"{endpoint}/set_gpt_weights",
                                     params={"weights_path": body.gpt_weights})
                results["gpt"] = "ok" if r.status_code == 200 else r.text[:200]
            if body.sovits_weights:
                r = await client.get(f"{endpoint}/set_sovits_weights",
                                     params={"weights_path": body.sovits_weights})
                results["sovits"] = "ok" if r.status_code == 200 else r.text[:200]
    except Exception as e:  # noqa: BLE001
        results["error"] = str(e)
    return results


# ── 立绘库（按角色存 data/characters/{角色}/sprites/）─────────
def _sprites_dir(char_id: str) -> Path:
    """角色立绘目录 data/characters/{char}/sprites/（自动创建）。"""
    d = _DATA / "characters" / _safe_name(char_id) / "sprites"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _sprites_json_path(char_id: str) -> Path:
    return _sprites_dir(char_id) / "sprites.json"


@app.get("/api/sprites/{char_id}")
def get_sprites(char_id: str) -> dict:
    """读角色立绘库 {emotions: {情绪名: {image: 文件名}}}。不存在返回空。"""
    p = _sprites_json_path(char_id)
    if not p.exists():
        return {"emotions": {}}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return {"emotions": {}}
        if "emotions" not in data:
            data["emotions"] = {}
        return data
    except (json.JSONDecodeError, ValueError, OSError):
        return {"emotions": {}}


class SpriteLibUpdate(BaseModel):
    lib: dict


@app.put("/api/sprites/{char_id}")
def put_sprites(char_id: str, body: SpriteLibUpdate) -> dict:
    """保存角色立绘库 sprites.json。"""
    _sprites_json_path(char_id).write_text(
        json.dumps(body.lib, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return {"ok": True}


@app.post("/api/sprites/{char_id}/image")
async def upload_sprite_image(char_id: str, file: UploadFile = File(...)) -> dict:
    """上传一张立绘图片到角色 sprites/ 目录，返回 {filename}。"""
    # 防路径穿越：原始 filename 含路径分隔符/.. 直接拒（不靠 Path.name 静默改名）
    raw_name = file.filename or "sprite.png"
    if "/" in raw_name or "\\" in raw_name or ".." in raw_name:
        raise HTTPException(400, "非法文件名")
    fname = _safe_name(raw_name)
    data = await file.read()
    if not data:
        raise HTTPException(400, "空文件")
    (_sprites_dir(char_id) / fname).write_bytes(data)
    return {"filename": fname}


@app.delete("/api/sprites/{char_id}/image/{filename}")
def delete_sprite_image(char_id: str, filename: str) -> dict:
    """删除角色立绘目录里的一张图片。"""
    p = _sprites_dir(char_id) / _safe_name(filename)
    if p.exists():
        p.unlink()
    return {"ok": True}


@app.get("/api/sprites/{char_id}/image/{filename}")
def get_sprite_image(char_id: str, filename: str):
    """返回立绘图片（前端预览用）。"""
    p = _sprites_dir(char_id) / _safe_name(filename)
    if not p.exists():
        raise HTTPException(404, "图片不存在")
    return FileResponse(str(p))


# ── 角色卡 ────────────────────────────────────────────────────
@app.get("/api/characters")
def list_characters() -> list[dict]:
    """返回 [{id, name}]：id=目录名（内部隔离键，建后不变），name=card.json 展示名。"""
    d = _DATA / "characters"
    if not d.exists():
        return []
    out = []
    for p in d.iterdir():
        if not (p.is_dir() and (p / "card.json").exists()):
            continue
        try:
            card = json.loads((p / "card.json").read_text(encoding="utf-8"))
            disp = card.get("name") or p.name
        except (json.JSONDecodeError, OSError):
            disp = p.name
        out.append({"id": p.name, "name": disp})
    return out


@app.get("/api/characters/{name}")
def get_character(name: str) -> dict:
    p = _DATA / "characters" / _safe_name(name) / "card.json"
    if not p.exists():
        raise HTTPException(404, "角色卡不存在")
    return json.loads(p.read_text(encoding="utf-8"))


async def _infer_clinginess(char_name: str) -> float:
    """用主脑 LLM 从角色名推断黏人度 ∈ [0,1]。失败默认 0.5。

    只发一次轻量请求（<20 token 输出），不阻塞 UI。
    如果角色名看不出性格（如"测试"），模型应该回 0.5。
    """
    cfg = _load_raw_config()
    llm = (cfg.get("modules") or {}).get("llm") or {}
    api_key = llm.get("api_key") or ""
    base_url = llm.get("base_url") or ""
    model = llm.get("model") or ""
    if not api_key or not base_url or not model:
        return 0.5
    prompt = (
        "你是一个角色性格分析器。给定角色名称，推断它的「黏人度」(clinginess)："
        "被忽略时倾向焦虑追问还是识趣退避。\n"
        "输出一个 0 到 1 之间的数字（保留两位小数），含义：\n"
        "0=极度独立，被冷落会安静退到一边\n"
        "0.5=中性\n"
        "1=极度黏人，越没人理越焦虑追问\n"
        "只输出数字，不要解释。如果角色名看不出性格倾向，输出 0.5。\n\n"
        f"角色名：{char_name}"
    )
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(
                f"{base_url.rstrip('/')}/chat/completions",
                headers={"Authorization": f"Bearer {api_key}",
                         "Content-Type": "application/json"},
                json={"model": model, "max_tokens": 10, "temperature": 0.3,
                      "messages": [{"role": "user", "content": prompt}]},
            )
            resp.raise_for_status()
            text = resp.json()["choices"][0]["message"]["content"].strip()
            val = float(text)
            return max(0.0, min(1.0, round(val, 2)))
    except Exception:  # noqa: BLE001
        return 0.5


@app.post("/api/characters/{name}")
async def create_character(name: str) -> dict:
    """新建角色：建目录 + 写一张最小 card.json + LLM 推断 clinginess。"""
    d = _DATA / "characters" / _safe_name(name)
    card_path = d / "card.json"
    if card_path.exists():
        raise HTTPException(400, "角色已存在")
    d.mkdir(parents=True, exist_ok=True)
    clinginess = await _infer_clinginess(name)
    card = {
        "name": name, "description": "", "personality": "",
        "scenario": "", "first_mes": "", "mes_example": "",
        "system_prompt": "", "post_history_instructions": "",
        "extensions": {"clinginess": clinginess},
    }
    card_path.write_text(json.dumps(card, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"ok": True, "clinginess": clinginess}


class RenameCharacter(BaseModel):
    new_name: str


@app.post("/api/characters/{name}/rename")
def rename_character(name: str, body: RenameCharacter) -> dict:
    """重命名角色展示名：只改 card.json 的 name 字段，目录名（内部隔离键）永不变。"""
    card_path = _DATA / "characters" / _safe_name(name) / "card.json"
    if not card_path.exists():
        raise HTTPException(404, "角色不存在")
    card = json.loads(card_path.read_text(encoding="utf-8"))
    card["name"] = body.new_name
    card_path.write_text(json.dumps(card, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"ok": True, "new_name": body.new_name}


class CardUpdate(BaseModel):
    card: dict


@app.put("/api/characters/{name}")
def put_character(name: str, body: CardUpdate) -> dict:
    d = _DATA / "characters" / _safe_name(name)
    d.mkdir(parents=True, exist_ok=True)
    (d / "card.json").write_text(
        json.dumps(body.card, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    # 热重载：如果保存的是当前角色，通知 orchestrator 重新加载角色卡
    cfg = _load_raw_config()
    current_char = (cfg.get("character") or {}).get("name", "默认")
    if _safe_name(name) == current_char and _reload_card_fn:
        _reload_card_fn()
    return {"ok": True}


@app.delete("/api/characters/{name}")
def delete_character(name: str) -> dict:
    """删除角色：整个目录（state/意识流/聊天历史）+ 该角色的 mem0 长期记忆。

    不允许删当前正在使用的角色，也不允许删到一个不剩。
    """
    import shutil
    safe = _safe_name(name)
    d = _DATA / "characters" / safe
    if not d.exists():
        raise HTTPException(404, "角色不存在")
    cfg = _load_raw_config()
    if (cfg.get("character") or {}).get("name") == name:
        raise HTTPException(400, "不能删除当前正在使用的角色，请先切换到其他角色")
    others = [p for p in (_DATA / "characters").iterdir()
              if p.is_dir() and (p / "card.json").exists() and p.name != safe]
    if not others:
        raise HTTPException(400, "至少保留一个角色，不能删除最后一个")
    shutil.rmtree(d)
    # 清理该角色的 mem0 长期记忆（user_id=目录名，存全局 memory_db，不在角色目录下）
    if _delete_memory_fn:
        try:
            _delete_memory_fn(name)
        except Exception as e:  # noqa: BLE001
            log.error("mem0 记忆清理失败: %s", e)
    return {"ok": True}


# ── 世界书 ────────────────────────────────────────────────────
def _wb_path(name: str) -> Path:
    return _DATA / "worldbooks" / f"{_safe_name(name)}.json"


def _default_wb() -> dict:
    return {"name": "", "entries": []}


def _default_entry(uid: int) -> dict:
    return {
        "uid": uid, "key": [], "keysecondary": [], "content": "",
        "constant": False, "selective": False, "selectiveLogic": "AND_ANY",
        "order": 100, "position": "after", "probability": 100,
        "sticky": 0, "cooldown": 0, "delay": 0, "disable": False,
    }


@app.get("/api/worldbooks")
def list_worldbooks() -> list[str]:
    d = _DATA / "worldbooks"
    return [p.stem for p in d.glob("*.json")] if d.exists() else []


@app.get("/api/worldbooks/{name}")
def get_worldbook(name: str) -> dict:
    p = _wb_path(name)
    if not p.exists():
        raise HTTPException(404, f"世界书 {name} 不存在")
    return json.loads(p.read_text(encoding="utf-8"))


class WBUpdate(BaseModel):
    doc: dict


@app.put("/api/worldbooks/{name}")
def put_worldbook(name: str, body: WBUpdate) -> dict:
    p = _wb_path(name)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(body.doc, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"ok": True}


@app.post("/api/worldbooks/{name}")
def create_worldbook(name: str) -> dict:
    p = _wb_path(name)
    if p.exists():
        raise HTTPException(400, "世界书已存在")
    p.parent.mkdir(parents=True, exist_ok=True)
    wb = _default_wb()
    wb["name"] = name
    p.write_text(json.dumps(wb, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"ok": True}


@app.delete("/api/worldbooks/{name}")
def delete_worldbook(name: str) -> dict:
    p = _wb_path(name)
    if not p.exists():
        raise HTTPException(404, "不存在")
    p.unlink()
    return {"ok": True}


# ── 预设（通用） ──────────────────────────────────────────────
@app.get("/api/presets")
def list_presets() -> list[str]:
    d = _DATA / "presets"
    return [p.stem for p in d.glob("*.json")] if d.exists() else []


@app.get("/api/presets/{name}")
def get_preset(name: str) -> dict:
    p = _DATA / "presets" / f"{_safe_name(name)}.json"
    if not p.exists():
        raise HTTPException(404, "预设不存在")
    return json.loads(p.read_text(encoding="utf-8"))


class JsonDoc(BaseModel):
    doc: dict


@app.put("/api/presets/{name}")
def put_preset(name: str, body: JsonDoc) -> dict:
    d = _DATA / "presets"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{_safe_name(name)}.json").write_text(
        json.dumps(body.doc, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return {"ok": True}


@app.post("/api/presets/{name}")
def create_preset(name: str) -> dict:
    p = _DATA / "presets" / f"{_safe_name(name)}.json"
    if p.exists():
        raise HTTPException(400, "预设已存在")
    p.parent.mkdir(parents=True, exist_ok=True)
    default = {
        "name": name,
        "system_prompt": "",   # 全局 system 前置（覆盖/补充角色定义）
        "jailbreak": "",       # 破限指令（拼在 system 末尾，最强位置）
        "post_history": "",    # 历史后指令（拼在角色卡 post_history_instructions 之后）
    }
    p.write_text(json.dumps(default, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"ok": True}


@app.delete("/api/presets/{name}")
def delete_preset(name: str) -> dict:
    p = _DATA / "presets" / f"{_safe_name(name)}.json"
    if not p.exists():
        raise HTTPException(404, "预设不存在")
    p.unlink()
    return {"ok": True}


# ── 静态前端 ──────────────────────────────────────────────────
@app.get("/")
def index() -> FileResponse:
    return FileResponse(str(_UI_DIR / "index.html"))


if _UI_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(_UI_DIR)), name="static")


def run(enqueue=None, port: int = 8080) -> None:
    import uvicorn
    if enqueue:
        set_enqueue(enqueue)
    log.info("管理平台启动: http://127.0.0.1:%s", port)
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")


if __name__ == "__main__":
    run()
