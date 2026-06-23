"""测试语音库多参考音频：验证随机选择逻辑。"""
import json
from pathlib import Path
from core.config import load_config
from core.action.speak import Speaker


def test_multi_ref_selection():
    """测试：每个情绪配置多个参考音频，随机选择其中一个。"""
    print("=" * 70)
    print("Test: Multi-reference audio selection")
    print("=" * 70)

    # 直接修改 library.json（默认语音库）进行测试
    lib_path = Path("D:/code/self/newTouch/data/voices/library.json")

    # 备份原始文件
    backup_path = lib_path.with_suffix('.json.backup')
    if lib_path.exists():
        backup_path.write_text(lib_path.read_text(encoding='utf-8'), encoding='utf-8')

    # 创建测试语音库
    test_lib = {
        "gpt_weights": "",
        "sovits_weights": "",
        "emotions": {
            "happy": {
                "refs": [
                    {
                        "ref_audio_path": "path/to/happy1.wav",
                        "prompt_text": "开心参考1",
                        "prompt_lang": "zh"
                    },
                    {
                        "ref_audio_path": "path/to/happy2.wav",
                        "prompt_text": "开心参考2",
                        "prompt_lang": "zh"
                    },
                    {
                        "ref_audio_path": "path/to/happy3.wav",
                        "prompt_text": "开心参考3",
                        "prompt_lang": "zh"
                    }
                ]
            },
            "sad": {
                "refs": [
                    {
                        "ref_audio_path": "path/to/sad1.wav",
                        "prompt_text": "悲伤参考",
                        "prompt_lang": "zh"
                    }
                ]
            }
        }
    }

    lib_path.write_text(json.dumps(test_lib, ensure_ascii=False, indent=2), encoding="utf-8")

    # 测试随机选择
    cfg = load_config()
    speaker = Speaker(cfg)
    speaker._cur_emotion = "happy"

    # 多次调用，统计选择分布
    selections = []
    for _ in range(30):
        ref_audio, prompt_text, prompt_lang = speaker._ref_for_emotion()
        selections.append(prompt_text)

    # 统计
    from collections import Counter
    counts = Counter(selections)

    print("\n情绪: happy (3个参考音频)")
    print(f"  总调用次数: 30")
    print(f"  选择分布:")
    for prompt, count in counts.items():
        percentage = (count / 30) * 100
        print(f"    '{prompt}': {count}次 ({percentage:.1f}%)")

    # 验证：所有3个参考都应该被选到
    unique_selected = set(selections)
    expected = {"开心参考1", "开心参考2", "开心参考3"}

    status = "[OK]" if unique_selected == expected else "[FAIL]"
    print(f"\n{status} 随机性验证")
    print(f"  期望选到: {expected}")
    print(f"  实际选到: {unique_selected}")

    # 测试单个参考（不应该报错）
    speaker._cur_emotion = "sad"
    ref_audio, prompt_text, prompt_lang = speaker._ref_for_emotion()

    print(f"\n情绪: sad (1个参考音频)")
    print(f"  选择结果: {prompt_text}")
    status2 = "[OK]" if prompt_text == "悲伤参考" else "[FAIL]"
    print(f"{status2} 单参考验证")

    # 恢复原始文件
    if backup_path.exists():
        lib_path.write_text(backup_path.read_text(encoding='utf-8'), encoding='utf-8')
        backup_path.unlink()

    return unique_selected == expected and prompt_text == "悲伤参考"


def test_backward_compatibility():
    """测试：旧格式（单个参考）自动兼容。"""
    print("\n\n" + "=" * 70)
    print("Test: Backward compatibility with old format")
    print("=" * 70)

    lib_path = Path("D:/code/self/newTouch/data/voices/library.json")
    backup_path = lib_path.with_suffix('.json.backup2')
    if lib_path.exists():
        backup_path.write_text(lib_path.read_text(encoding='utf-8'), encoding='utf-8')

    # 创建旧格式语音库
    test_lib = {
        "gpt_weights": "",
        "sovits_weights": "",
        "emotions": {
            "neutral": {
                "ref_audio_path": "path/to/neutral.wav",
                "prompt_text": "中性参考",
                "prompt_lang": "zh"
            }
        }
    }

    lib_path.write_text(json.dumps(test_lib, ensure_ascii=False, indent=2), encoding="utf-8")

    cfg = load_config()
    speaker = Speaker(cfg)
    speaker._cur_emotion = "neutral"

    ref_audio, prompt_text, prompt_lang = speaker._ref_for_emotion()

    print(f"\n旧格式语音库")
    print(f"  情绪: neutral")
    print(f"  ref_audio_path: {ref_audio}")
    print(f"  prompt_text: {prompt_text}")
    print(f"  prompt_lang: {prompt_lang}")

    status = "[OK]" if prompt_text == "中性参考" and ref_audio == "path/to/neutral.wav" else "[FAIL]"
    print(f"\n{status} 旧格式兼容性")

    # 恢复
    if backup_path.exists():
        lib_path.write_text(backup_path.read_text(encoding='utf-8'), encoding='utf-8')
        backup_path.unlink()

    return prompt_text == "中性参考"


def test_empty_refs_fallback():
    """测试：refs 为空时回退到全局默认。"""
    print("\n\n" + "=" * 70)
    print("Test: Empty refs fallback to global default")
    print("=" * 70)

    lib_path = Path("D:/code/self/newTouch/data/voices/library.json")
    backup_path = lib_path.with_suffix('.json.backup3')
    if lib_path.exists():
        backup_path.write_text(lib_path.read_text(encoding='utf-8'), encoding='utf-8')

    # 创建空 refs 的语音库
    test_lib = {
        "gpt_weights": "",
        "sovits_weights": "",
        "emotions": {
            "unknown": {
                "refs": []  # 空数组
            }
        }
    }

    lib_path.write_text(json.dumps(test_lib, ensure_ascii=False, indent=2), encoding="utf-8")

    cfg = load_config()
    speaker = Speaker(cfg)
    speaker._cur_emotion = "unknown"

    ref_audio, prompt_text, prompt_lang = speaker._ref_for_emotion()

    print(f"\n空 refs 数组")
    print(f"  情绪: unknown")
    print(f"  回退结果: {prompt_text}")
    print(f"  (应该是 config 中的全局默认)")

    # 空 refs 应该回退到全局默认（config 的 modules.tts.prompt_text）
    status = "[OK]" if prompt_text else "[FAIL]"
    print(f"\n{status} 空 refs 回退验证")

    # 恢复
    if backup_path.exists():
        lib_path.write_text(backup_path.read_text(encoding='utf-8'), encoding='utf-8')
        backup_path.unlink()

    return bool(prompt_text)


if __name__ == "__main__":
    print("语音库多参考音频测试\n")

    results = []
    results.append(("Multi-ref selection", test_multi_ref_selection()))
    results.append(("Backward compatibility", test_backward_compatibility()))
    results.append(("Empty refs fallback", test_empty_refs_fallback()))

    print("\n\n" + "=" * 70)
    print("Summary: Voice library multi-reference audio")
    print("=" * 70)

    for name, passed in results:
        status = "[PASS]" if passed else "[FAIL]"
        print(f"{status} {name}")

    all_passed = all(r[1] for r in results)
    if all_passed:
        print("\n所有测试通过！")
        print("\n功能：")
        print("✓ 每个情绪支持多个参考音频（refs 数组）")
        print("✓ TTS 生成时随机选择其中一个")
        print("✓ 向后兼容旧格式（自动转换）")
        print("✓ 空 refs 回退到全局默认")
        print("\n新格式示例：")
        print("""
{
  "emotions": {
    "happy": {
      "refs": [
        {"ref_audio_path": "happy1.wav", "prompt_text": "text1", "prompt_lang": "zh"},
        {"ref_audio_path": "happy2.wav", "prompt_text": "text2", "prompt_lang": "zh"}
      ]
    }
  }
}
        """)
        print("\n管理平台使用：")
        print("1. 打开管理平台 -> 语音库 Tab")
        print("2. 选择情绪档，点击'+ 添加参考音频'")
        print("3. 填写多个参考音频的路径和文字")
        print("4. 保存后，TTS 每次随机选择一个参考")
    else:
        print("\n部分测试失败，请检查输出。")
