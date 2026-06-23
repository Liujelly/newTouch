"""self_config.py 功能测试（独立脚本，无需 pytest）。

覆盖：
  1. get_my_status — 只读，无需权限
  6. set_speaking_frequency — 无权限时拒绝
  7. set_speaking_frequency — 非法档位拒绝
  8. set_speaking_frequency — 有权限 + 合法档位成功 + gatekeeper live-refresh
  9. toggle_vision — 无权限时拒绝
 10. toggle_vision — 有权限时成功 + live-refresh
 11. switch_preset — 预设文件不存在时拒绝
 12. switch_preset — 有权限 + 文件存在时成功
 13. 审计日志逐条写入

（select_voice 已移除——音色=切 GPT-SoVITS 模型，待多模型切换功能再加；
  情绪参考音频由 speak 按 <emo:> 标签自动选，不走 AI 工具。）

运行: cd D:\\code\\self\\newTouch && python tests/test_self_config.py
"""
from __future__ import annotations

import asyncio
import json
import sys
import tempfile

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
from pathlib import Path

# 保证 core.* 可以 import
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.config import Config
from core.state import EmotionState
from core.tools.self_config import SelfConfig, FREQUENCY_PRESETS


# ── 通用 fixture ──────────────────────────────────────────────
def make_env(*, preset: str | None = None,
             perm_freq=False, perm_vision=False, perm_preset=False):
    """在临时目录中搭好 SelfConfig 所需的文件树，返回 (SelfConfig, tmpdir_path)。"""
    tmp = Path(tempfile.mkdtemp())
    (tmp / "data").mkdir()
    (tmp / "data" / "presets").mkdir(parents=True)

    # 如果给了预设就创建文件
    if preset:
        (tmp / "data" / "presets" / f"{preset}.json").write_text(
            json.dumps({"name": preset}), encoding="utf-8"
        )

    raw = {
        "modules": {"tts": {}},
        "proactive": {"min_interval_seconds": 900, "hourly_cap": 5},
        "perception": {"vision": {"enabled": False}},
        "character": {"current_preset": "默认"},
        "ai_permissions": {
            "allow_adjust_frequency": perm_freq,
            "allow_toggle_vision": perm_vision,
            "allow_switch_preset": perm_preset,
        },
    }
    cfg = Config(raw)

    class _PatchedConfig(Config):
        @property
        def project_root(self):
            return tmp

        def char_data_dir(self, char_name=None):
            # 测试里把审计日志放 tmp/data 下
            d = tmp / "data"
            d.mkdir(parents=True, exist_ok=True)
            return d

    pcfg = _PatchedConfig(raw)

    # 刷新钩子: 记录被调用次数和参数
    refresher_calls: dict[str, list] = {"gatekeeper": [], "vision": []}

    def _ref_gk(c):
        refresher_calls["gatekeeper"].append((
            c.get("proactive.min_interval_seconds"),
            c.get("proactive.hourly_cap"),
        ))

    def _ref_vis(c):
        refresher_calls["vision"].append(c.get("perception.vision.enabled"))

    state = EmotionState()
    sc = SelfConfig(pcfg, state, refreshers={"gatekeeper": _ref_gk, "vision": _ref_vis})
    return sc, pcfg, tmp, refresher_calls


# ── 测试辅助 ─────────────────────────────────────────────────
PASS = "OK"
FAIL = "FAIL"
_results: list[tuple[str, bool, str]] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    _results.append((name, cond, detail))
    mark = PASS if cond else FAIL
    print(f"  {mark} {name}" + (f"  [{detail}]" if detail else ""))


async def run_tests():
    print("\n── test_self_config ──────────────────────────────────────")

    # 1. get_my_status — 只读
    sc, cfg, tmp, _ = make_env()
    result = await sc.get_my_status()
    data = json.loads(result)
    check("1. get_my_status 返回情绪/权限", "emotion" in data and "permissions" in data)

    # 6. set_speaking_frequency — 无权限
    sc, cfg, tmp, _ = make_env()
    r = await sc.set_speaking_frequency("quiet")
    check("6. 无权限时拒绝", r.startswith("拒绝"), r[:40])

    # 7. set_speaking_frequency — 非法档位
    sc, cfg, tmp, _ = make_env(perm_freq=True)
    r = await sc.set_speaking_frequency("极少")
    check("7. 非法档位拒绝", r.startswith("拒绝"), r[:40])

    # 8. set_speaking_frequency — 成功 + live-refresh
    sc, cfg, tmp, ref_calls = make_env(perm_freq=True)
    r = await sc.set_speaking_frequency("quiet")
    check("8. 有权限+合法档位成功", r.startswith("OK"), r)
    check("8. config 间隔已更新", cfg.get("proactive.min_interval_seconds") == FREQUENCY_PRESETS["quiet"]["min_interval_seconds"])
    check("8. gatekeeper live-refresh 被调用", len(ref_calls["gatekeeper"]) == 1)
    check("8. refresh 参数正确", ref_calls["gatekeeper"][0] == (
        FREQUENCY_PRESETS["quiet"]["min_interval_seconds"],
        FREQUENCY_PRESETS["quiet"]["hourly_cap"],
    ))

    # 9. toggle_vision — 无权限
    sc, cfg, tmp, _ = make_env()
    r = await sc.toggle_vision(True)
    check("9. toggle_vision 无权限拒绝", r.startswith("拒绝"), r[:40])

    # 10. toggle_vision — 成功 + live-refresh
    sc, cfg, tmp, ref_calls = make_env(perm_vision=True)
    r = await sc.toggle_vision(True)
    check("10. toggle_vision 成功", r.startswith("OK"), r)
    check("10. vision live-refresh 被调用", len(ref_calls["vision"]) == 1)
    check("10. vision refresh 值为 True", ref_calls["vision"][0] is True)

    # 11. switch_preset — 文件不存在
    sc, cfg, tmp, _ = make_env(perm_preset=True)
    r = await sc.switch_preset("不存在预设")
    check("11. 预设文件不存在时拒绝", r.startswith("拒绝"), r[:40])

    # 12. switch_preset — 成功
    sc, cfg, tmp, _ = make_env(perm_preset=True, preset="温柔")
    r = await sc.switch_preset("温柔")
    check("12. switch_preset 成功", r.startswith("OK"), r)
    check("12. config 预设已更新", cfg.get("character.current_preset") == "温柔")

    # 13. 审计日志（用 set_speaking_frequency：有权限成功 OK + 非法档位拒绝）
    sc, cfg, tmp, _ = make_env(perm_freq=True)
    await sc.set_speaking_frequency("quiet")     # OK
    await sc.set_speaking_frequency("不存在档位")  # 拒绝
    audit_lines = (tmp / "data" / "audit.log").read_text(encoding="utf-8").strip().splitlines()
    entries = [json.loads(ln) for ln in audit_lines]
    check("13. 审计日志有2条", len(entries) == 2)
    check("13. 第1条结果为OK", entries[0]["result"] == "OK")
    check("13. 第2条结果含拒绝", entries[1]["result"].startswith("拒绝"))

    # ── 汇总 ──────────────────────────────────────────────────
    total = len(_results)
    passed = sum(1 for _, ok, _ in _results if ok)
    failed = total - passed
    print(f"\n  {passed}/{total} 通过" + (f"，{failed} 失败 ❌" if failed else " ✅"))
    return failed == 0


if __name__ == "__main__":
    ok = asyncio.run(run_tests())
    sys.exit(0 if ok else 1)
