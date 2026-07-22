"""测试系统托盘命令通道（v2.65 托盘常驻·方案B）。

不依赖 PyQt6 GUI（托盘 widget / 菜单手动验证），只测可自动化部分：
1. FaceBroadcaster 命令分发：client 发 {"cmd":"quit"} -> on_command 回调触发
2. chat 与 cmd 同连接并存、互不干扰（回归 on_chat 不被影响）
3. 任意 cmd 值透传给回调（主侧自行决定 act on 哪些）
4. FaceReceiver.send_command 发出正确 JSON 行（{"cmd":"..."}\\n）
"""
import asyncio
import json
import socket
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8")

from core.sprite.broadcaster import FaceBroadcaster


async def _cmd_scenario():
    """broadcaster 起 server + mock client 发 cmd/chat，验回调。"""
    import random
    port = random.randint(30000, 39999)
    b = FaceBroadcaster("127.0.0.1", port)
    await b.start()
    assert b._started, "broadcaster 应启动成功"

    cmds: list[str] = []
    chats: list[str] = []

    async def _on_cmd(cmd):
        cmds.append(cmd)

    async def _on_chat(text):
        chats.append(text)

    b.set_on_command(_on_cmd)
    b.set_on_chat(_on_chat)

    sock = socket.create_connection(("127.0.0.1", port), timeout=3)
    await asyncio.sleep(0.3)  # 等 server accept

    # 1) 发退出命令 -> on_command 收到 "quit"
    sock.sendall((json.dumps({"cmd": "quit"}) + "\n").encode("utf-8"))
    await asyncio.sleep(0.3)
    assert cmds == ["quit"], f"on_command 应收到 quit: {cmds}"
    print(f"✅ 命令通道分发: {cmds}")

    # 2) chat 仍正常（cmd 处理不影响 chat 路由）
    sock.sendall((json.dumps({"chat": "在吗"}) + "\n").encode("utf-8"))
    await asyncio.sleep(0.3)
    assert chats == ["在吗"], f"on_chat 应收到: {chats}"
    print(f"✅ chat 与 cmd 共存互不干扰: chat={chats} cmd={cmds}")

    # 3) 任意 cmd 透传（主侧 _on_sprite_command 自行判断 act on 哪些）
    sock.sendall((json.dumps({"cmd": "mute"}) + "\n").encode("utf-8"))
    await asyncio.sleep(0.3)
    assert cmds == ["quit", "mute"], f"未知 cmd 也应透传: {cmds}"
    print(f"✅ 任意 cmd 透传: {cmds}")

    # 4) 非法 JSON 行不崩（被 json 异常吞掉）
    sock.sendall(b"not a json line\n")
    await asyncio.sleep(0.2)
    assert cmds == ["quit", "mute"], f"非法行不应影响: {cmds}"
    print("✅ 非法 JSON 行不崩")

    sock.close()
    await asyncio.sleep(0.2)
    await b.stop()


def test_command_channel():
    asyncio.run(_cmd_scenario())


def test_send_command_format():
    """FaceReceiver.send_command 发出 {"cmd":"..."}\\n；无连接返回 False。"""
    try:
        from sprite_window import FaceReceiver
    except ImportError:
        print("⏭ 跳过 send_command 测试（PyQt6 未装）")
        return

    r = FaceReceiver("127.0.0.1", 1)  # 不连接（连接在 run() 里），仅用 send_command

    # 无连接 / 空命令 -> False
    assert r.send_command("quit") is False, "无连接应返回 False"
    assert r.send_command("") is False, "空命令应返回 False"
    print("✅ send_command 无连接/空命令返回 False")

    # mock socket 捕获发送内容（用真实 threading.Lock 即可）
    class _Sock:
        def __init__(self):
            self.sent = b""

        def sendall(self, data):
            self.sent += data

    sock = _Sock()
    r._sock = sock
    assert r.send_command("quit") is True, "有连接应返回 True"
    line = sock.sent.decode("utf-8")
    assert line.endswith("\n"), f"应以换行结尾: {line!r}"
    assert json.loads(line) == {"cmd": "quit"}, f"JSON 格式错误: {line!r}"
    print(f"✅ send_command 格式正确: {line!r}")


if __name__ == "__main__":
    print("=== 测试系统托盘命令通道 ===\n")
    test_command_channel()
    test_send_command_format()
    print("\n=== 全部测试通过 ===")
