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

from PyQt6.QtCore import Qt, pyqtSignal, QObject, QTimer
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import QApplication, QLabel, QWidget

from core.config import load_config
from core.sprite.store import load_sprites, image_path


class FaceReceiver(QObject):
    """TCP client 后台线程：连主程序收 face/台词，发 Qt 信号给主线程。"""

    face_received = pyqtSignal(str)  # face 字符串
    text_received = pyqtSignal(str)  # 台词增量 chunk（气泡追加）
    text_end_received = pyqtSignal()  # 一段回复结束（气泡淡出）

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
                            if "face" in obj:
                                self.face_received.emit(obj.get("face") or "neutral")
                            if "text" in obj:
                                txt = obj.get("text") or ""
                                if txt:
                                    self.text_received.emit(txt)
                            if obj.get("text_end"):
                                self.text_end_received.emit()
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

        # 台词气泡：立绘头顶上方，圆角白底黑字，超时自动消失
        self._bubble = QLabel(self)
        self._bubble.setWordWrap(True)
        self._bubble.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._bubble.setStyleSheet(
            "QLabel { background: rgba(255,255,255,235); color: #1e293b;"
            "border-radius: 12px; padding: 8px 12px; font-size: 14px; }"
        )
        self._bubble.hide()
        # 气泡超时定时器（config sprite.bubble_timeout，默认 5s；0=不自动消失）。
        # 回复结束后启动，到时清空文本 + hide。
        self._bubble_timeout = float(config.get("sprite.bubble_timeout", 5))
        self._bubble_timer = QTimer(self)
        self._bubble_timer.setSingleShot(True)
        self._bubble_timer.timeout.connect(self._hide_bubble)

        # 表情恢复定时器：表情切换后过 face_reset_timeout 秒自动回 neutral（情绪过去回归平静）。
        # config sprite.face_reset_timeout（默认 8s；0=不恢复，一直停最后表情）。
        self._face_reset_timeout = float(config.get("sprite.face_reset_timeout", 8))
        self._face_reset_timer = QTimer(self)
        self._face_reset_timer.setSingleShot(True)
        def _on_face_reset():
            self._show_face("neutral")
        self._face_reset_timer.timeout.connect(_on_face_reset)

        # 布局：上方气泡区 + 下方立绘区。立绘按固定高度缩放，气泡在上方独立区不挡脸。
        self._portrait_h = 480  # 立绘固定高度
        self._bubble_area_h = 140  # 气泡区预留高度
        self._bubble_width = 460  # 气泡最大宽度（宽一点少换行，避免纵向细长）
        self._bubble_active = False  # 当前是否在一段回复流式期间
        self._cur_face_label = "neutral"  # 当前显示的表情名（供 text_end 判断要不要恢复）

        # 窗口初始尺寸 = 气泡区 + 立绘区（无气泡时气泡区也预留，避免布局跳动）
        self.resize(self._bubble_width, self._bubble_area_h + self._portrait_h)
        self._show_face("neutral")

        # 屏幕右下角
        screen = QApplication.primaryScreen().availableGeometry()
        self.move(screen.right() - self.width() - 40, screen.bottom() - self.height() - 40)

        # 拖动状态：不点击穿透（始终可拖），代价是浮窗会挡住后面操作
        self._drag_offset = None

    def mousePressEvent(self, event) -> None:
        """按住左键开始拖动窗口。"""
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_offset = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
            event.accept()

    def mouseMoveEvent(self, event) -> None:
        """拖动窗口。"""
        if self._drag_offset is not None and event.buttons() & Qt.MouseButton.LeftButton:
            self.move(event.globalPosition().toPoint() - self._drag_offset)
            event.accept()

    def mouseReleaseEvent(self, event) -> None:
        """松开结束拖动。"""
        self._drag_offset = None
        event.accept()

    def show_face(self, face: str) -> None:
        """Qt 信号槽：收到 face 换图（主线程执行）。

        只切图，不启动恢复定时器——face_reset 从回复结束(show_text_end)算起，
        和气泡淡出同起点，保证表情(8s)比气泡(5s)晚消失。
        新回复开头的 face 停掉上一轮未触发的恢复定时器（这轮表情先显示着）。
        """
        self._show_face(face)
        self._face_reset_timer.stop()  # 新 face 来了，取消上一轮的恢复倒计时

    def show_text(self, text: str) -> None:
        """Qt 信号槽：收到增量 chunk，追加到气泡（流式打字效果），气泡在立绘上方独立区。

        流式期间气泡常驻、不启动超时（等 show_text_end 收尾才淡出）。
        """
        if not text:
            return
        # 一段新回复开始（首次追加）：先清空
        if not self._bubble_active:
            self._bubble.setText("")
            self._bubble_active = True
            self._bubble_timer.stop()  # 取消上一次的淡出
        self._bubble.setText(self._bubble.text() + text)
        self._layout_bubble()
        self._bubble.show()

    def show_text_end(self) -> None:
        """Qt 信号槽：一段回复结束：启动气泡淡出 + 表情恢复定时器（同起点）。

        表情 face_reset_timeout(默认8s) > 气泡 bubble_timeout(默认5s)，故表情比气泡晚消失。
        """
        self._bubble_active = False
        if self._bubble_timeout > 0:
            self._bubble_timer.start(int(self._bubble_timeout * 1000))
        # 表情恢复：当前非 neutral 才启动（neutral 无需恢复）
        if (self._face_reset_timeout > 0 and self._cur_face_label != "neutral"
                and not self._face_reset_timer.isActive()):
            self._face_reset_timer.start(int(self._face_reset_timeout * 1000))

    def _hide_bubble(self) -> None:
        """气泡淡出：清空文本 + 隐藏。"""
        self._bubble.setText("")
        self._bubble.hide()

    def _layout_bubble(self) -> None:
        """气泡定位在立绘上方独立区：水平居中，底部贴立绘顶部（留 6px 间距）。

        气泡底部对齐立绘顶部，字少时气泡往下贴、和立绘近，不留大空；字多时往上长（限气泡区）。
        """
        self._bubble.setMaximumWidth(self._bubble_width)
        self._bubble.adjustSize()
        bw = min(self._bubble.sizeHint().width(), self.width())
        bh = min(self._bubble.sizeHint().height(), self._bubble_area_h - 8)
        self._bubble.resize(bw, bh)
        x = max(0, (self.width() - bw) // 2)
        # 底部贴立绘顶部（立绘 y = _bubble_area_h），留 6px 间距
        y = self._bubble_area_h - bh - 6
        self._bubble.move(x, y)
        self._bubble.raise_()

    def _show_face(self, face: str) -> None:
        self._cur_face_label = face
        path = image_path(face, self._mapping)
        # 立绘区高度固定 _portrait_h，放窗口下方（气泡区之下）
        label_h = self._portrait_h
        if path and Path(path).exists():
            pix = QPixmap(path)
            if not pix.isNull():
                scaled = pix.scaledToHeight(
                    label_h, Qt.TransformationMode.SmoothTransformation
                )
                self._label.setPixmap(scaled)
                self._label.setStyleSheet("")
                self._label.resize(scaled.width(), label_h)
                # 立绘水平居中，垂直在气泡区下方
                self._label.move(max(0, (self.width() - scaled.width()) // 2),
                                 self._bubble_area_h)
                self.resize(max(self.width(), scaled.width()),
                            self._bubble_area_h + label_h)
                return
        # 无图：占位文字
        self._label.setText(f"[{self._char_name}]\n{face}\n（无立绘图）")
        self._label.setStyleSheet("color: rgba(255,255,255,180); font-size: 16px;")
        self._label.resize(self.width(), label_h)
        self._label.move(0, self._bubble_area_h)
        self.resize(self.width(), self._bubble_area_h + label_h)


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
    receiver.text_received.connect(window.show_text)
    receiver.text_end_received.connect(window.show_text_end)
    t = threading.Thread(target=receiver.run, daemon=True)
    t.start()

    def on_quit():
        receiver.stop()

    app.aboutToQuit.connect(on_quit)

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
