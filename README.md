# newTouch

newTouch 是一个运行在 PC 上的自主 AI 陪伴体。它不只是等待提问，还会结合角色设定、情绪、记忆、时间、视觉和日程主动思考，并通过语音、管理平台或桌面立绘与你互动。

当前版本：**v2.70**

## 主要能力

| 能力 | 说明 |
|---|---|
| 对话 | 支持文本与麦克风输入，LLM 回复可流式送入 GPT-SoVITS |
| 主动行为 | 心跳驱动内心独白，结合孤独感、勿扰时段、频率限制和未回应反馈决定是否开口 |
| 情绪 | 五维情绪状态、自然衰减、LLM 情绪增量、语音语气和立绘表情联动 |
| 视觉 | 摄像头帧差检测、VLM 画面描述、显著变化主动判断、单次主动复查 |
| 记忆 | 短期窗口与增量摘要、mem0 长期记忆、自动召回和 `memory_search` 工具 |
| 人设 | SillyTavern 角色卡、提示词预设、用户人设和世界书激活 |
| 工具 | 天气、联网搜索、视觉查看、记忆检索、日程管理和受控自我配置 |
| 日程 | 单次/每日提醒，到点后以角色身份走主动思考路径提醒 |
| 回复审查 | 可选机械审查；发现循环、过长或系统标记泄漏时由 LLM 最小修正 |
| 管理平台 | 配置、聊天、意识流、情绪、日志、工具、权限、角色、世界书、日程和素材管理 |
| 桌面形态 | PyQt6 立绘浮窗、台词气泡、表情切换、情绪抖动和系统托盘 |

## 运行结构

```text
麦克风 / 文本 / 管理平台 / 日程 / 摄像头
                    │
                    ▼
            优先级事件队列
                    │
                    ▼
          Orchestrator 编排器
          ├─ 反应路径：用户发言 → LLM → 工具 → 回复
          ├─ 主动路径：心跳/日程 → 内心独白 → speak/silent/look
          └─ 视觉路径：帧差 → VLM caption → 判断/复查 → speak/silent
                    │
          ┌─────────┼─────────┐
          ▼         ▼         ▼
       GPT-SoVITS  长期记忆   管理平台/立绘
```

## 环境要求

- Windows 为主要运行环境。
- 推荐 Python **3.11 或 3.12**。
- LLM 至少需要一个 Anthropic 或 OpenAI 兼容接口。
- 语音输出需要单独运行的 GPT-SoVITS 服务。
- 视觉需要摄像头和支持图片输入的 VLM；主脑 LLM 与 VLM 可以使用不同服务。
- 桌面立绘和系统托盘需要 PyQt6；未安装时主程序会自动跳过该子系统。

Python 3.14 可以运行核心功能，但 FunASR 的部分依赖可能没有预编译 wheel，需要本地 C++ 编译环境或手动处理依赖。

## 快速开始

### 1. 创建环境并安装依赖

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

如果只使用文本输入、关闭语音、视觉、记忆和立绘，可以按实际需要精简依赖。

### 2. 初始化本地配置

运行时配置、角色数据、日志、记忆库和素材包含隐私或密钥，均不会被 Git 提交。首次运行需要自行创建：

```powershell
Copy-Item .env.example .env
Copy-Item data/config.example.yaml data/config.yaml
```

在 `.env` 中填写 API Key，并在 `data/config.yaml` 中引用，例如：

```yaml
modules:
  llm:
    provider: deepseek
    model: deepseek-chat
    api_key: ${DEEPSEEK_API_KEY}
    base_url: https://api.deepseek.com/v1
```

VLM、长期记忆和 Tavily 等服务如使用不同密钥，也应分别配置对应环境变量。

### 3. 准备角色卡

`character.name` 必须对应以下文件：

```text
data/characters/<角色名>/card.json
```

最小角色卡示例：

```json
{
  "name": "默认",
  "description": "PC 中的 AI 陪伴者",
  "personality": "自然、体贴，有自己的想法",
  "scenario": "与你共同生活在日常环境中",
  "first_mes": "你好，我在这里。",
  "system_prompt": "始终保持角色身份，自然地与用户相处。",
  "extensions": {}
}
```

将配置同步为：

```yaml
character:
  name: 默认
```

也可以先放入已有的 SillyTavern JSON 角色卡，启动后再通过管理平台编辑和切换。

### 4. 选择输入与输出方式

最容易验证的是文本模式：

```yaml
perception:
  audio:
    enabled: false

modules:
  tts:
    enabled: false
```

此时终端会显示 `你 >` 输入提示，输入 `/quit` 退出。

启用语音输出前，需要先手动启动 GPT-SoVITS API 服务，并确认配置的端点可访问：

```yaml
modules:
  tts:
    enabled: true
    endpoint: http://127.0.0.1:9880
```

启用麦克风后，本地 STT 可选择：

- `faster-whisper`：多语言通用，CPU `int8` 可用。
- `funasr`：SenseVoice 中文场景更快，并可保留语音情绪标签。

### 5. 启动

```powershell
python main.py
```

默认管理平台地址：<http://127.0.0.1:8080>

如果安装了 PyQt6，系统托盘会随主程序启动。`sprite.enabled` 仅决定启动时是否显示立绘，关闭时仍可从托盘唤出。

## 常用配置

配置文件为 `data/config.yaml`，管理平台也可编辑大部分字段。

| 配置组 | 用途 |
|---|---|
| `modules.llm` | 主脑模型、接口和采样参数 |
| `modules.vlm` | 视觉模型与 caption 输出参数 |
| `modules.stt` / `modules.vad` | 语音识别和人声检测 |
| `modules.tts` | GPT-SoVITS、回复语言、参考音频和合成参数 |
| `perception.audio` | 麦克风、对话窗口与回声冷却 |
| `perception.vision` | 摄像头、帧差阈值、caption 与主动查看冷却 |
| `proactive` | 心跳、主动频率、勿扰、黏人度、未回应和去重 |
| `memory` | mem0、embedding、短期窗口、摘要与自动召回 |
| `tools` | 各 LLM 工具的注册开关；多数改动需要重启 |
| `ai_permissions` | AI 修改配置、视觉和日程的运行时权限 |
| `schedule` | 日程扫描与提醒行为 |
| `reply_review` | 回复审查开关与最大长度 |
| `sprite` | 立绘浮窗、气泡、表情恢复和抖动参数 |
| `logging` | 系统日志级别、目录和保留天数 |

配置保存后的生效方式取决于字段：

- TTS 参数、角色人设、部分主动参数和视觉开关等会在使用时重新读取。
- LLM/VLM/STT/记忆客户端、摄像头编号和工具注册等构造期配置需要重启。
- 管理平台中标记“需重启”的字段应以该提示为准。

## 工具与权限

`tools.*` 决定工具是否注册给 LLM，`ai_permissions.*` 决定有副作用的操作是否允许执行，两者是不同层级。

例如，AI 要自行创建提醒，需要同时满足：

```yaml
tools:
  add_schedule: true

ai_permissions:
  allow_manage_schedules: true
```

只读工具如天气、记忆检索和视觉查看通常不需要额外授权，但仍受模块总开关限制。

## 数据目录

```text
core/
├─ orchestrator.py       事件分发与反应/主动/视觉/日程路径
├─ cognition.py          Anthropic/OpenAI 兼容认知与工具循环
├─ character.py          角色卡、语言、情绪和 prompt 组装
├─ state.py              五维情绪状态
├─ gatekeeper.py         主动行为闸门
├─ review.py             回复机械审查
├─ perception/
│  ├─ audio_in.py        文本/麦克风输入、VAD 与对象分类
│  ├─ stt.py             faster-whisper/FunASR 抽象
│  ├─ vision.py          摄像头、帧差、VLM 与 look_now
│  └─ schedule.py        角色日程存储与调度
├─ memory/store.py       mem0 长期记忆
├─ sprite/               立绘素材读取与主进程广播
└─ tools/                天气、搜索、视觉、记忆、日程与自配置工具
api/admin.py             FastAPI 管理平台后端
ui/index.html            Vue 3 单页管理界面
sprite_window.py         PyQt6 立绘浮窗与系统托盘进程
data/                    本地配置、角色、日志、素材和运行时状态
tests/                   功能与回归脚本
main.py                  主程序入口
```

每个角色的数据相互隔离，主要保存在：

```text
data/characters/<角色名>/
├─ card.json
├─ state.json
├─ chat_history.jsonl
├─ consciousness.jsonl
├─ schedules.json
└─ sprites/
   └─ sprites.json
```

## 已知限制

- 当前麦克风采用半双工回声门控：TTS 播放期间会暂停收音，因此暂不支持语音打断播放；文本和管理平台消息可以打断。
- GPT-SoVITS 是外部服务，目前需要单独手动启动。
- DuckDuckGo 在部分网络环境中不可用；联网搜索可切换 Tavily。
- 视觉 caption 可能误判，尤其是光线差或遮挡场景。显著变化路径会允许角色主动复查一次，但仍不等同于身份识别。
- 回复审查开启后需要先缓存完整回复，不再流式出声；发现问题时还会增加一次 LLM 调用。
- 长期记忆需要兼容的 embedding 接口；仅有 DeepSeek 对话接口通常不足以启用完整记忆链路。

## 测试

测试脚本以离线单元和链路回归为主，部分文件会调用真实模型或外部服务，运行前请先阅读文件头说明。

代表性离线回归：

```powershell
python tests/test_vision_significant.py
python tests/test_vision_tools.py
python -m tests.test_proactive_cot
python -m tests.test_recent_inner_time
```

## 文档

本地开发工作区中的 `docs/` 包含架构设计、决策记录、Token 密集型优化方案和详细版本进度。该目录按项目约定不提交 Git，因此 README 保持为仓库内可追踪的项目入口，详细开发记录以本地 `docs/进度.md` 为准。
