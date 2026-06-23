# newTouch

PC 常驻的自主 AI 陪伴体：会主动思考、有情绪、能看见环境、语音对话、在受限范围内自我配置。

设计文档见 `docs/`：
- `架构设计文档.md` — 完整架构（14 章）
- `决策记录.md` — 关键决策理由（ADR）
- `进度.md` — 开发进度与下一步（新 session 接手先读这个）

## 当前状态

**阶段 1/2/3/4 全部完成，核心功能全部可用。**

| 功能 | 状态 |
|---|---|
| 语音对话（STT + GPT-SoVITS TTS） | ✅ 联调通过 |
| STT 引擎可切换（faster-whisper / FunASR SenseVoice） | ✅ |
| 主动思考 + 情绪 + 沉默闸门 | ✅ |
| 视觉（摄像头 + VLM caption 注入） | ✅ |
| 工具调用（天气、时间） | ✅ |
| 对话对象分流（唤醒词/窗口/LLM） | ✅ |
| 人设系统 + 世界书（ST 兼容） | ✅ |
| 长期记忆（mem0 + chroma，语义召回） | ✅ 联调通过 |
| 自我配置工具（白名单权限） | ✅ |
| 管理平台（Vue 3，`http://127.0.0.1:8080`） | ✅ |
| 重启续聊（短期窗口自动回填） | ✅ |
| 配置保存即热重载（多数配置改完无需重启） | ✅ |

## 快速开始

> 推荐 **Python 3.11 / 3.12**（依赖兼容性最稳）。3.14 也能跑，但 funasr 的
> `editdistance` 依赖在 3.14 无预编译 wheel，需 C++ 编译器或 `pip install funasr --no-deps`
> 后手动补依赖。3.11 下直接 `pip install -r requirements.txt` 基本一把过。

```powershell
# 1. 安装依赖（建议在 Python 3.11 虚拟环境中）
pip install -r requirements.txt
# 只用 faster-whisper、不用 FunASR 的话，可跳过 funasr/modelscope/transformers

# 2. （可选）安装 mem0 NLP 增强，提升记忆检索质量
pip install mem0ai[nlp]              # 安装 spaCy 库
python -m spacy download en_core_web_sm  # 下载英文模型（~12MB）
# 注：Python 3.14 下 spaCy 暂无预编译包，可跳过此步

# 3. 配置 .env（填入 key）
# ARK_API_KEY=你的火山方舟 key

# 4. 启动
python main.py
```

管理平台：启动后访问 `http://127.0.0.1:8080`，可在线配置所有参数、查看意识流/情绪/聊天记录。多数配置**保存即生效**，无需重启；标「需重启」徽标的字段（LLM/VLM/STT/记忆等连接类）改完需重启 `main.py`。

## 关键配置（data/config.yaml）

| 配置项 | 说明 |
|---|---|
| `modules.llm` | 主脑 LLM（当前：火山 ark-code-latest） |
| `modules.tts.enabled` | GPT-SoVITS 语音开关，false 则打印文本 |
| `modules.tts.ref_audio_path` | 参考音频路径 |
| `modules.stt.provider` | STT 引擎：`faster-whisper`（多语言）/ `funasr`（SenseVoice 中文专优、更快） |
| `modules.stt.model_cache_dir` | STT 模型缓存目录（默认 `data/stt_cache`，相对项目根） |
| `perception.vision.enabled` | 摄像头视觉开关 |
| `perception.audio.enabled` | 麦克风开关，false 则用管理平台文字输入 |
| `memory.enabled` | 长期记忆开关（mem0 + chroma） |
| `memory.chat_history_window` | 短期对话窗口条数，同时决定重启回填多少条 |
| `proactive.enabled` | 主动发言开关 |

## 目录结构

```
core/
├─ config.py          配置加载（+.env + ${ENV} 替换）
├─ events.py          事件定义 + 优先级队列
├─ orchestrator.py    编排器：事件分发 + 反应/主动两条路径
├─ character.py       角色卡 + prompt 组装（世界书/记忆/情绪注入）
├─ cognition.py       认知引擎（anthropic / openai 兼容，流式）
├─ state.py           情绪状态（五维 + 衰减 + 持久化）
├─ gatekeeper.py      主动发言闸门（勿扰/间隔/每小时上限/backoff）
├─ heartbeat.py       心跳（随机间隔触发主动路径）
├─ consciousness.py   意识流日志（consciousness.jsonl）
├─ worldinfo.py       世界书激活（ST 兼容算法）
├─ perception/
│   ├─ audio_in.py    输入源（TextInput / MicInput + VAD）
│   ├─ stt.py         STT 引擎抽象（faster-whisper / FunASR 可切换）
│   └─ vision.py      摄像头抓帧 + VLM caption + 主动 look_now
├─ action/
│   └─ speak.py       流式分句 → GPT-SoVITS → 播放 + barge-in
├─ memory/
│   └─ store.py       长期记忆（mem0 + chroma，续写抽取 / infer）
└─ tools/
    ├─ registry.py    工具注册表
    ├─ external.py    外部工具（天气/时间）
    └─ self_config.py 自我配置工具（白名单 + 审计）
api/
└─ admin.py           管理平台后端（FastAPI）
ui/
└─ index.html         管理平台前端（Vue 3 单页）
data/
├─ config.yaml        主配置
├─ characters/        角色数据（card/state/意识流/聊天历史，按角色隔离）
├─ worldbooks/        世界书 JSON
├─ presets/           提示词预设
├─ memory_db/         本地向量库（chroma）
└─ stt_cache/         STT 模型缓存（FunASR SenseVoice 等，自动下载）
main.py               启动入口
```

## 路线图

- 阶段 1：语音对话闭环（反应路径）✅
- 阶段 2：主动路径 + 情绪 + 沉默闸门 ✅
- 阶段 3：视觉 + 工具 + 对话对象分流 ✅
- 阶段 4：人设系统 + 记忆 + 自我配置 + 管理平台 ✅
- 最后：定时提醒/待办（融入主动思考）
