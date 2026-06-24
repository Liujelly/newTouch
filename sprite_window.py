"""立绘浮窗进程（PyQt6 透明置顶桌面浮窗）。

独立进程，主程序通过 TCP socket 推送 face（立绘表情）过来，浮窗按 face 查立绘库换图。
启动：python sprite_window.py --port 17621 --character 小触

特性：
- 透明背景 + 无边框 + 置顶 + 点击穿透（不挡操作）
- QLabel 显示 PNG（保留 alpha 通道透明）
- TCP client 后台线程连主程序，收到 {"face","character"} 换图；断线 3s 重连
- 默认显示 neutral 立绘；无立绘库时显示占位文字

主程序退出 → server 关闭 → client 重连失败 → 浮窗自动退出（通过 stop 信号）。
"""
from __future__ import annotations

import argparse
import json
import socket
import sys
import threading
from pathlib import Path

# 项目根（本文件在根目录）
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from PyQt6.QtCore import Qt, pyqtSignal, QObject
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import QApplication, QLabel, QWidget

from core.config import load_config
from core.sprite.store import load_sprites, image_path


class FaceReceiver(QObject):
    """TCP client 后台线程：连主程序收 face，发 Qt 信号给主线程换图。"""

    face_received = pyqtSignal(str)  # face 字符串

    def __init__(self, host: str, port: int) -> None:
        super().__init__()
        self._host = host
        self._port = port
        self._stop = threading.Event()

    def run(self) -> None:
        buf = ""
        while not self._stop.is_set():
            sock = None
            try:
                sock = socket.create_connection((self._host, self._port), timeout=3)
                sock.settimeout(1.0)  # 便于周期检查 _stop
                buf = ""
                while not self._stop.is_set():
                    try:
                        data = sock.recv(4096)
                    except socket.timeout:
                        continue
                    if not data:
                        break  # server 关闭
                    buf += data.decode("utf-8", errors="replace")
                    while "\n" in buf:
                        line, buf = buf.split("\n", 1)
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            obj = json.loads(line)
                            face = obj.get("face") or "neutral"
                            self.face_received.emit(face)
                        except (json.JSONDecodeError, ValueError):
                            pass
            except (OSError, ConnectionError):
                pass
            finally:
                if sock is not None:
                    try:
                        sock.close()
                    except OSError:
                        pass
            # 断线后等 3s 重连（被 stop 打断则退出）
            self._stop.wait(3.0)

    def stop(self) -> None:
        self._stop.set()


class SpriteWindow(QWidget):
    """透明置顶立绘浮窗。"""

    def __init__(self, config, char_name: str) -> None:
        super().__init__()
        self._cfg = config
        self._char_name = char_name
        self._mapping = load_sprites(config, char_name)

        # 透明 + 无边框 + 置顶
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool  # 不在任务栏显示
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)

        self._label = QLabel(self)
        self._label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # 初始尺寸（无图时占位）
        self.resize(360, 480)
        self._show_face("neutral")

        # 屏幕右下角
        screen = QApplication.primaryScreen().availableGeometry()
        self.move(screen.right() - self.width() - 40, screen.bottom() - self.height() - 40)

        self._make_click_through()

    def _make_click_through(self) -> None:
        """Windows：让窗口鼠标点击穿透（不挡桌面操作）。"""
        if sys.platform != "win32":
            return
        try:
            import ctypes
            GWL_EXSTYLE = -20
            WS_EX_LAYERED = 0x00080000
            WS_EX_TRANSPARENT = 0x00000020
            hwnd = int(self.winId())
            style = ctypes.windll.user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
            ctypes.windll.user32.SetWindowLongW(
                hwnd, GWL_EXSTYLE, style | WS_EX_LAYERED | WS_EX_TRANSPARENT
            )
        except (OSError, AttributeError):
            pass

    def show_face(self, face: str) -> None:
        """Qt 信号槽：收到 face 换图（主线程执行）。"""
        self._show_face(face)

    def _show_face(self, face: str) -> None:
        path = image_path(face, self._mapping)
        if path and Path(path).exists():
            pix = QPixmap(path)
            if not pix.isNull():
                # 按窗口高度等比缩放
                scaled = pix.scaledToHeight(
                    self.height(), Qt.TransformationMode.SmoothTransformation
                )
                self._label.setPixmap(scaled)
                self._label.resize(scaled.size())
                self.resize(scaled.size())
                self._label.move(0, 0)
                return
        # 无图：占位文字
        self._label.setText(f"[{self._char_name}]\n{face}\n（无立绘图）")
        self._label.setStyleSheet("color: rgba(255,255,255,180); font-size: 16px;")
        self._label.resize(360, 480)
        self.resize(360, 480)


def main() -> None:
    parser = argparse.ArgumentParser(description="newTouch 立绘浮窗")
    parser.add_argument("--port", type=int, default=17621)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--character", default="默认")
    args = parser.parse_args()

    app = QApplication(sys.argv)
    config = load_config()
    window = SpriteWindow(config, args.character)
    window.show()

    receiver = FaceReceiver(args.host, args.port)
    receiver.face_received.connect(window.show_face)
    t = threading.Thread(target=receiver.run, daemon=True)
    t.start()

    def on_quit():
        receiver.stop()

    app.aboutToQuit.connect(on_quit)

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
