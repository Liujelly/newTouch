"""newTouch 启动入口 (阶段3: 视觉 + 工具 + 对话对象分流)。"""
from __future__ import annotations

import asyncio
import sys

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stdin.reconfigure(encoding="utf-8")

from core.config import load_config
from core.logger import setup_logging, get_logger
from core.character import CharacterCard
from core.cognition import Cognition
from core.state import EmotionState
from core.gatekeeper import GateKeeper
from core.consciousness import ConsciousnessLog
from core.action.speak import Speaker
from core.orchestrator import Orchestrator
from core.perception.audio_in import TextInput, MicInput, Classifier
from core.heartbeat import Heartbeat
from core.tools.external import register_external_tools
from core.tools.self_config import register_self_config_tools


async def main() -> None:
    cfg = load_config()
    setup_logging(cfg)
    log = get_logger("main")

    char_name = cfg.get("character.name", "小触")
    card = CharacterCard.load(
        cfg.project_root / "data" / "characters" / char_name / "card.json"
    )

    char_dir = cfg.char_data_dir()
    state = EmotionState.load(char_dir / "state.json")

    cognition = Cognition(cfg)
    speaker = Speaker(cfg, name=card.name)
    gatekeeper = GateKeeper(cfg)
    consciousness = ConsciousnessLog(char_dir / "consciousness.jsonl")
    ready = asyncio.Event()
    # 托盘「退出」命令置位此事件 -> 打断输入源 await -> 走 finally 清理（同 Ctrl+C 路径）
    shutdown_evt = asyncio.Event()

    # 视觉感知先构造，传给 orchestrator（主动 look_now 需要）
    from core.perception.vision import Vision
    vision = Vision(cfg, None)  # enqueue 先占位，构造后再绑

    orch = Orchestrator(
        config=cfg,
        cognition=cognition,
        speaker=speaker,
        card=card,
        state=state,
        gatekeeper=gatekeeper,
        consciousness=consciousness,
        ready=ready,
        vision=vision,
    )
    # 绑定 enqueue（构造 orch 之后才有）
    vision._enqueue = orch.enqueue

    # 注册 AI 白名单自我配置工具（8.3），绑定 live-refresh 钩子
    def _refresh_gatekeeper(cfg):
        gatekeeper.refresh(cfg)

    def _refresh_vision(cfg):
        vision._enabled = cfg.get("perception.vision.enabled", False)

    def _refresh_orch(cfg):
        orch._user_name = cfg.get("user_persona.name", "你")
        orch._max_history = cfg.get("memory.chat_history_window", 40)
        orch._compress_batch = cfg.get("memory.compress_batch_size", 10)

    def _refresh_logger(cfg):
        # 日志级别/目录/开关：重新跑 setup 即生效（幂等，会替换旧 file handler）
        setup_logging(cfg)

    refreshers = {
        "gatekeeper": _refresh_gatekeeper,
        "vision": _refresh_vision,
        "orch": _refresh_orch,
        "logger": _refresh_logger,
    }
    register_self_config_tools(cfg, state, refreshers={"gatekeeper": _refresh_gatekeeper, "vision": _refresh_vision})

    # 注册外部查询工具 get_weather（开关 tools.get_weather）
    register_external_tools(cfg)

    # 注册记忆检索工具 memory_search（反应路径 LLM 可自主调用，开关 tools.memory_search）
    from core.tools.memory_tools import register_memory_tools
    register_memory_tools(orch._memory, cfg)

    # 注册联网搜索工具 web_search（LLM 查实时信息自主调用，开关 tools.web_search）
    from core.tools.web_search import register_web_search_tools
    register_web_search_tools(cfg)

    # 注册视觉查看工具 look（LLM 反应路径自主看一眼，开关 tools.look）
    from core.tools.vision_tools import register_vision_tools
    register_vision_tools(vision, cfg)

    # 注册日程管理工具 add/list/mark_done/update/delete（开关 tools.*，另受权限页授权）
    from core.perception.schedule import ScheduleStore, Scheduler
    schedule_store = ScheduleStore(cfg.char_data_dir())
    from core.tools.schedule_tools import register_schedule_tools
    register_schedule_tools(schedule_store, cfg)

    # 立绘浮窗 + 系统托盘（方案B：随主程序常驻）。sprite_window 无条件启动（浮窗+托盘），
    # sprite.enabled 控制启动时是否显示浮窗；未装 PyQt6 则跳过降级。
    sprite_proc = None
    import importlib.util
    if importlib.util.find_spec("PyQt6") is not None:
        from core.sprite.broadcaster import FaceBroadcaster
        sprite_host = cfg.get("sprite.host", "127.0.0.1")
        sprite_port = int(cfg.get("sprite.port", 17621))
        broadcaster = FaceBroadcaster(sprite_host, sprite_port)
        asyncio.create_task(broadcaster.start())
        speaker.set_emotion_broadcaster(broadcaster)

        # 浮窗右键输入框发来的聊天消息 → 投入 orchestrator 队列（等同文本输入）
        from core.events import Event, EventType, EventPriority
        async def _on_sprite_chat(text: str) -> None:
            await orch.enqueue(Event(
                priority=EventPriority.URGENT,
                type=EventType.USER_SPEECH,
                payload={"text": text},
            ))
        broadcaster.set_on_chat(_on_sprite_chat)

        # 托盘「退出」命令 -> 置位 shutdown_evt 触发优雅关闭（走下方 finally 清理）
        async def _on_sprite_command(cmd: str) -> None:
            if cmd == "quit":
                log.info("收到托盘退出命令，准备关闭…")
                shutdown_evt.set()
        broadcaster.set_on_command(_on_sprite_command)
        # 拉起独立浮窗进程（detached；主程序退出时 terminate）
        import subprocess
        show_sprite = bool(cfg.get("sprite.enabled", False))
        webui_port = cfg.get("ui.webui_port", 8080)
        sprite_proc = subprocess.Popen(
            [sys.executable, str(cfg.project_root / "sprite_window.py"),
             "--port", str(sprite_port), "--host", sprite_host,
             "--character", char_name,
             "--webui-port", str(webui_port),
             "--show" if show_sprite else "--no-show"],
            cwd=str(cfg.project_root),
        )
        log.info("立绘浮窗/托盘已启动: %s:%s (显示浮窗=%s)",
                 sprite_host, sprite_port, show_sprite)
    else:
        log.warning("未装 PyQt6，跳过立绘浮窗/托盘（pip install PyQt6 后重启启用）")

    vision_task = asyncio.create_task(vision.start())

    # 日程调度器（开关 schedule.enabled，到点投 SCHEDULE 事件走主动路径提醒）
    scheduler = None
    if cfg.get("schedule.enabled", True):
        scheduler = Scheduler(schedule_store, orch.enqueue, cfg)
        scheduler.start()

    orch_task = asyncio.create_task(orch.run())
    heartbeat = Heartbeat(cfg, orch.enqueue)
    hb_task = asyncio.create_task(heartbeat.start())

    # 管理平台（可选，ui.enable_webui 开关）；与主程序共享事件循环，Ctrl+C 一起关
    admin_server = None
    admin_task = None
    if cfg.get("ui.enable_webui", False):
        import uvicorn
        from api import admin as admin_module
        admin_module.set_enqueue(orch.enqueue)
        admin_module.set_switch_character(orch.switch_character)
        admin_module.set_reload_card(orch.reload_card)
        admin_module.set_delete_memory(orch._memory.delete_user)
        admin_module.set_live_config(cfg, refreshers)
        admin_module.set_generate_motion(cognition.generate_motion_map)
        port = cfg.get("ui.webui_port", 8080)
        host = cfg.get("ui.webui_host", "127.0.0.1")
        admin_cfg = uvicorn.Config(admin_module.app, host=host, port=port, log_level="warning")
        admin_server = uvicorn.Server(admin_cfg)
        admin_task = asyncio.create_task(admin_server.serve())
        log.info("管理平台: http://%s:%s", host, port)

    if card.first_mes:
        print(f"\n{card.name} > {card.first_mes}")

    proactive_on = cfg.get("proactive.enabled", False)
    log.info("主动模式: %s", "开" if proactive_on else "关 (proactive.enabled=false)")

    # 输入源: audio.enabled 决定走麦克风还是文本
    audio_on = cfg.get("perception.audio.enabled", False)
    mic = None
    text_in = None
    try:
        if audio_on:
            classifier = Classifier(
                cfg, cognition.client, cognition.backend,
                cognition.model, card.name,
            )
            mic = MicInput(cfg, orch.enqueue, classifier, ready=ready, speaker=speaker)
            orch.bind_history_sink(mic.update_history)  # 旁听上下文同步给分类器
            orch.bind_character_refresh(mic.refresh_character)  # 切换角色时刷新唤醒词
            input_coro = mic.start()
        else:
            text_in = TextInput(cfg, orch.enqueue, ready=ready)
            input_coro = text_in.start()

        # 输入源跑成 task，与 shutdown_evt 并发等待：托盘「退出」置位事件即可打断，
        # 走下方 finally 清理（与 Ctrl+C 同路径）。正常 /quit 或 EOF 时 input_task 先完成。
        input_task = asyncio.create_task(input_coro)
        shutdown_task = asyncio.create_task(shutdown_evt.wait())
        done, pending = await asyncio.wait(
            {input_task, shutdown_task}, return_when=asyncio.FIRST_COMPLETED
        )
        for t in pending:
            t.cancel()
        for t in pending:
            try:
                await t
            except (asyncio.CancelledError, Exception):
                pass
        if mic is not None:
            mic.stop()
        if text_in is not None:
            text_in.stop()
    finally:
        vision.stop()
        heartbeat.stop()
        if scheduler is not None:
            await scheduler.stop()
        await orch.shutdown()
        # 管理平台：设 should_exit 让 uvicorn 优雅自关（走完 lifespan shutdown），
        # 再 await 它正常结束。不要直接 cancel()——那会在 starlette lifespan 的
        # `await receive()` 处抛出 CancelledError 打到 stderr（无害但是噪音 traceback）。
        if admin_server:
            admin_server.should_exit = True
        if admin_task:
            try:
                await asyncio.wait_for(admin_task, timeout=5)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                admin_task.cancel()
        # 其余后台任务硬取消，gather 收掉 CancelledError 不外泄
        vision_task.cancel()
        hb_task.cancel()
        orch_task.cancel()
        await asyncio.gather(vision_task, hb_task, orch_task, return_exceptions=True)
        # 立绘浮窗进程：主程序退出时终止（否则 detached 进程残留）
        if sprite_proc is not None and sprite_proc.poll() is None:
            sprite_proc.terminate()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n再见~")
