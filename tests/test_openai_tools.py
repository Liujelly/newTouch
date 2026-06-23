"""OpenAI 兼容后端的工具循环测试（需真实 API key，仅手动跑）。

验证 cognition._openai_stream 的工具调用流程：
  1. schema 转换（Anthropic input_schema → OpenAI parameters）
  2. 流式输出 + tool_calls 收集
  3. 工具执行 + 结果回灌
  4. 多轮循环直到无工具调用

运行: cd D:\\code\\self\\newTouch && python tests/test_openai_tools.py
前提: config.yaml 的 modules.llm 配置火山或其他 OpenAI 兼容后端 + API key
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.config import load_config
from core.cognition import Cognition
from core.character import CharacterCard
import core.tools.external  # noqa: F401  import 即注册 get_weather 到 registry


async def test_tool_call():
    """模拟一次触发工具的对话。"""
    # 加载配置（load_config 自动找 data/config.yaml + .env）
    cfg = load_config()

    # 注册自我配置工具（get_my_status 等）
    from core.tools.self_config import register_self_config_tools
    from core.state import EmotionState
    register_self_config_tools(cfg, EmotionState())

    # 检查后端类型
    provider = cfg.get("modules.llm.provider", "openai")
    print(f"[测试] LLM Provider: {provider}")
    if provider == "anthropic":
        print("⚠️  当前配置是 anthropic，本测试需要 openai 兼容后端（如火山 ARK）")
        print("    请在 config.yaml 里改成 provider: openai")
        return

    # 创建 Cognition
    cognition = Cognition(cfg)

    # 简化角色卡
    card = CharacterCard(
        name="测试助手",
        description="一个会查天气和时间的助手",
        personality="热情、乐于助人",
        scenario="",
        first_mes="你好",
        system_prompt="你是一个助手，可以查询天气和时间。",
    )

    # 测试 case 1: 查询天气（外部工具）
    print("\n[Case 1] 用户问「北京天气」，预期调用 get_weather 工具")
    print("=" * 60)

    chat_history = []
    user_text = "北京今天天气怎么样？"

    collected = []
    async for chunk in cognition.react_stream(
        card=card,
        user_name="测试用户",
        user_text=user_text,
        chat_history=chat_history,
    ):
        collected.append(chunk)
        print(chunk, end="", flush=True)

    reply = "".join(collected)
    print(f"\n\n✓ 完整回复: {reply}")

    # 简单验证：回复里应该包含天气信息
    if any(w in reply for w in ["°C", "天气", "温度", "降雨", "晴", "雨", "云"]):
        print("✓ 回复包含天气信息，工具调用成功")
    else:
        print("✗ 回复不含天气信息，可能工具未调用")

    # 测试 case 2: 查询自身状态（自我配置工具）
    print("\n\n[Case 2] 用户问「你现在心情怎么样」，预期调用 get_my_status 工具")
    print("=" * 60)

    user_text2 = "你现在心情怎么样？最近主动说话频繁吗？"
    chat_history2 = [
        {"role": "user", "content": user_text},
        {"role": "assistant", "content": reply},
    ]

    collected2 = []
    async for chunk in cognition.react_stream(
        card=card,
        user_name="测试用户",
        user_text=user_text2,
        chat_history=chat_history2,
    ):
        collected2.append(chunk)
        print(chunk, end="", flush=True)

    reply2 = "".join(collected2)
    print(f"\n\n✓ 完整回复: {reply2}")

    print("（get_my_status 工具是否被调用，看终端是否有状态查询痕迹 / 回复是否提到具体情绪状态）")


if __name__ == "__main__":
    print("=" * 60)
    print("OpenAI 兼容后端工具循环测试")
    print("=" * 60)
    print("\n⚠️  本测试会真实调用 LLM API（消耗 token）")
    print("⚠️  需要 config.yaml 配置正确的 API key")
    print("⚠️  需要联网（查天气工具访问 wttr.in）\n")

    try:
        asyncio.run(test_tool_call())
        print("\n" + "=" * 60)
        print("测试完成")
        print("=" * 60)
    except Exception as e:
        print(f"\n✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
