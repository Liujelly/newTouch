"""立绘（sprite）子系统：桌面浮窗按情绪展示差分图。

- broadcaster.py：主程序 TCP server，把当前 face（立绘表情）推给浮窗进程。
- store.py：立绘库加载（情绪名→图片路径），仿语音库 load_voice_library。

设计见 docs/进度.md v2.50。与语音库（<emo:> → TTS 音色）解耦：
<face:> 标签独立驱动立绘，两库各自配置、各自查询、互不绑死。
"""
