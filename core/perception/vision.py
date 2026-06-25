"""视觉感知 (架构文档 5.5 / ADR-009)。

两条触发路径:
  被动: 后台抽帧 → 帧差检测 → 关键帧送 VLM → VISION_CHANGE 事件
  主动: look_now() 工具由内心独白调用 → 立即抓一帧送 VLM

perception.vision.enabled=false 时全部功能禁用 (可由 AI 通过白名单工具切换)。
"""
from __future__ import annotations

import asyncio
import base64
import io
import time
from dataclasses import dataclass
from typing import Callable

import cv2
import numpy as np

from ..config import Config
from ..logger import get_logger

log = get_logger("vision")


@dataclass
class VisionCaption:
    caption: str
    timestamp: float
    frame_diff_score: float


class Vision:
    def __init__(self, config: Config, enqueue: Callable):
        self._cfg = config
        self._enqueue = enqueue
        self._enabled = config.get("perception.vision.enabled", False)
        # 摄像头索引：没配或配了打不开时自动检测。
        # 启动时禁用则延迟到首次启用（start 循环里）再解析，避免没插摄像头时白探测。
        cfg_idx = config.get("perception.vision.camera_index", None)
        self._cam_idx = self._resolve_camera(cfg_idx) if self._enabled else None
        self._diff_threshold = config.get("perception.vision.frame_diff_threshold", 0.15)
        self._min_interval = config.get("perception.vision.min_caption_interval_s", 10)
        self._min_look_interval = config.get("perception.vision.min_look_interval_s", 30)

        # VLM 后端: anthropic 或 openai 兼容(豆包ARK/Qwen-VL等)
        vlm_provider = config.get("modules.vlm.provider", "anthropic")
        api_key = config.get("modules.vlm.api_key") or None
        base_url = config.get("modules.vlm.base_url") or None
        if vlm_provider == "anthropic":
            import anthropic
            self._vlm = anthropic.AsyncAnthropic(api_key=api_key) if api_key else anthropic.AsyncAnthropic()
            self._vlm_backend = "anthropic"
        else:
            from openai import AsyncOpenAI
            self._vlm = AsyncOpenAI(api_key=api_key or "sk-placeholder", base_url=base_url)
            self._vlm_backend = "openai"
        self._vlm_model = config.get("modules.vlm.model", "claude-opus-4-8")
        # caption 输出上限改为现读（见 _caption_max_tokens），管理平台改完下次抓帧即生效。

        self._last_frame: np.ndarray | None = None
        self._last_caption_ts: float = 0.0
        self._last_look_ts: float = 0.0
        self._latest: VisionCaption | None = None
        self._latest_frame_raw: np.ndarray | None = None  # 被动循环缓存的最新原始彩色帧
        self._running = False

    def get_latest_caption(self) -> VisionCaption | None:
        return self._latest

    @staticmethod
    def _probe_cameras(max_index: int = 5) -> list[int]:
        """枚举可用摄像头索引（试着打开 0..max_index-1）。"""
        available = []
        for i in range(max_index):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                ok, _ = cap.read()
                if ok:
                    available.append(i)
            cap.release()
        return available

    def _resolve_camera(self, cfg_idx: int | None) -> int:
        """选定摄像头：配置的索引能用就用它，否则自动挑第一个可用的。"""
        # 配置指定了索引且能打开 → 用它
        if cfg_idx is not None:
            cap = cv2.VideoCapture(cfg_idx)
            if cap.isOpened():
                ok, _ = cap.read()
                cap.release()
                if ok:
                    return cfg_idx
            cap.release()
            log.warning("配置的摄像头 index=%s 打不开，自动检测中…", cfg_idx)
        # 自动检测
        found = self._probe_cameras()
        if found:
            log.info("检测到可用摄像头: %s，使用 index=%s", found, found[0])
            return found[0]
        log.warning("未检测到可用摄像头，视觉将无法抓帧")
        return cfg_idx or 0

    async def look_now(self) -> VisionCaption | None:
        """主动工具：立即抓一帧送 VLM (受 min_look_interval 限制)。

        摄像头同一索引不能被两个 capture 同时独占（Windows MSMF 报 -1072873821）。
        所以被动抽帧循环在跑时，直接取它缓存的最新帧，不另开摄像头；只有被动循环
        没开时才自己临时打开抓一帧。
        """
        if not self._enabled:
            return None
        now = time.time()
        if now - self._last_look_ts < self._min_look_interval:
            return self._latest

        frame = None
        frame_src = ""
        if self._passive_running():
            # 被动循环常开摄像头：复用它缓存的最新帧，避免独占冲突
            frame = self._latest_frame_raw
            frame_src = "缓存帧(被动循环)"
            if frame is None:
                # 循环刚启动还没抓到帧，稍等一下重试
                for _ in range(10):
                    await asyncio.sleep(0.2)
                    if self._latest_frame_raw is not None:
                        frame = self._latest_frame_raw
                        break
        else:
            # 被动循环没在抽帧：自己临时开一次摄像头抓帧
            idx = self._cam_idx
            if idx is None:
                idx = self._resolve_camera(self._cfg.get("perception.vision.camera_index", None))
                self._cam_idx = idx
            cap = cv2.VideoCapture(idx)
            ok, f = cap.read()
            cap.release()
            if ok:
                frame = f
                frame_src = f"临时开摄像头(index={idx})"

        if frame is None:
            log.warning("look_now 抓帧失败（无可用帧）")
            return None
        self._last_look_ts = now
        log.info("look_now 抓帧来源: %s", frame_src or "未知")
        caption = await self._vlm_caption(frame, 0.0)
        if caption:
            log.info("look_now caption: %s", (caption.caption or "")[:120])
        self._latest = caption
        return caption

    async def start(self) -> None:
        """持久化监督循环：常驻运行，按 _enabled 标志启停被动抽帧。

        这样管理平台 / AND AI 工具切换 perception.vision.enabled 时能即时生效——
        关→开会重新打开摄像头开始抽帧，开→关会释放摄像头停止抽帧，都无需重启进程。
        摄像头只在启用期间被打开占用。
        """
        self._running = True
        cap = None
        try:
            while self._running:
                if not self._enabled:
                    # 关闭状态：释放摄像头（让 look_now 临时开/其他程序可用），轻量轮询等待开启
                    if cap is not None:
                        cap.release()
                        cap = None
                        self._latest_frame_raw = None
                        self._last_frame = None
                    await asyncio.sleep(0.5)
                    continue

                # 启用状态：确保摄像头已打开（刚从关切到开时解析索引并打开）
                if cap is None:
                    if self._cam_idx is None:
                        self._cam_idx = self._resolve_camera(
                            self._cfg.get("perception.vision.camera_index", None))
                    cap = cv2.VideoCapture(self._cam_idx)
                    if not cap.isOpened():
                        log.warning("摄像头打开失败，1s 后重试")
                        cap.release()
                        cap = None
                        await asyncio.sleep(1)
                        continue

                ok, frame = cap.read()
                if not ok:
                    await asyncio.sleep(1)
                    continue
                self._latest_frame_raw = frame  # 缓存最新原始帧，供 look_now() 复用

                diff_score = self._frame_diff(frame)
                now = time.time()

                if (diff_score > self._diff_threshold
                        and now - self._last_caption_ts > self._min_interval):
                    self._last_caption_ts = now
                    sig_threshold = self._cfg.get(
                        "perception.vision.significant_threshold", 0.30)
                    is_sig = diff_score >= sig_threshold
                    log.info("帧差 %.3f 超阈值 %.2f → 送 VLM 抓帧（%s）",
                             diff_score, self._diff_threshold,
                             "显著变化" if is_sig else "普通变化")
                    caption = await self._vlm_caption(frame, diff_score)
                    if caption:
                        self._latest = caption
                        from ..events import Event, EventType, EventPriority
                        evt_type = (EventType.VISION_SIGNIFICANT
                                    if is_sig
                                    else EventType.VISION_CHANGE)
                        await self._enqueue(Event(
                            priority=EventPriority.NORMAL,
                            type=evt_type,
                            payload={"caption": caption.caption,
                                     "diff_score": diff_score},
                        ))
                        log.info("VLM 描述（diff %.3f, %s）：%s",
                                 diff_score, evt_type.name,
                                 (caption.caption or "")[:60])

                await asyncio.sleep(1)
        finally:
            if cap is not None:
                cap.release()

    def _passive_running(self) -> bool:
        """被动循环当前是否在真正抽帧（用于 look_now 判断是否复用缓存帧）。"""
        return self._running and self._enabled

    def stop(self) -> None:
        self._running = False

    def _frame_diff(self, frame: np.ndarray) -> float:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self._last_frame is None:
            self._last_frame = gray
            return 0.0
        diff = cv2.absdiff(self._last_frame, gray).astype(float)
        score = diff.mean() / 255.0
        self._last_frame = gray
        return score

    def _caption_max_tokens(self) -> int:
        """caption 输出上限，现读 config（管理平台改完下次抓帧即生效）。
        推理型模型(如豆包 ark-*)会先消耗 reasoning token，上限太小会被思考吃光导致正文为空，
        默认 512；描述越详细需越大。"""
        return self._cfg.get("modules.vlm.caption_max_tokens", 512)

    async def _vlm_caption(self, frame: np.ndarray, diff_score: float) -> VisionCaption | None:
        prompt = (
            "你是一双眼睛，自然地描述此刻看到的画面，像在跟人随口讲你看到了什么，三四句话。\n"
            "如果画面里有人，多留意人：ta 的神态、表情、情绪线索（比如累了、在笑、皱着眉、走神），"
            "正在做的动作和姿势，明显的穿着或外貌特征；再带一句周围环境和氛围。\n"
            "如果没有人，就细致描述场景本身——布局、光线明暗、氛围，以及画面里几样主要物品"
            "（例如「房间里没人，床上堆着没叠的被子，窗帘拉着，光线有点暗」"
            "「书桌前空着，台灯还亮着，摊着几本书和一个没合上的笔记本」）。\n"
            "口语化、具体、有画面感，但不要罗列成清单，也不要写「画面中」「图片里」这种生硬措辞。\n"
            "看不清或不确定的就如实说「看不太清」，不要猜、不要编造没看到的细节。"
        )
        try:
            _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            img_b64 = base64.b64encode(buf.tobytes()).decode()
            if self._vlm_backend == "anthropic":
                resp = await self._vlm.messages.create(
                    model=self._vlm_model,
                    max_tokens=self._caption_max_tokens(),
                    messages=[{"role": "user", "content": [
                        {"type": "image", "source": {"type": "base64",
                                                      "media_type": "image/jpeg",
                                                      "data": img_b64}},
                        {"type": "text", "text": prompt},
                    ]}],
                )
                caption = resp.content[0].text.strip()
            else:
                # openai 兼容(豆包ARK/Qwen-VL): image_url + data URI
                resp = await self._vlm.chat.completions.create(
                    model=self._vlm_model,
                    max_tokens=self._caption_max_tokens(),
                    messages=[{"role": "user", "content": [
                        {"type": "image_url", "image_url": {
                            "url": f"data:image/jpeg;base64,{img_b64}"}},
                        {"type": "text", "text": prompt},
                    ]}],
                )
                caption = (resp.choices[0].message.content or "").strip()
            return VisionCaption(caption=caption, timestamp=time.time(),
                                 frame_diff_score=diff_score)
        except Exception as e:  # noqa: BLE001
            log.error("VLM 调用失败: %s", e)
            return None
