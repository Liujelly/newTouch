"""测试主动思考链（CoT）解析器（Token 密集型优化 #2）。

测试 _parse_monologue_cot() 的各种场景。
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

from core.cognition import _parse_monologue_cot, _parse_monologue


def test_cot_speak():
    """测试 CoT 格式：开口说话。"""
    raw = """
嗯...现在都晚上10点了。
他上次说那个项目这周要交，应该快截止了吧。
这个点还没睡，会不会在加班？
我记得他是那种会熬夜赶工的性格...
要不要问一句呢？
算了，还是问一下吧。

[决定：开口]
"项目进展怎么样了？还在忙吗？"
"""
    result = _parse_monologue_cot(raw)

    print("=== 测试 CoT 开口 ===")
    print(f"thought: {result['thought'][:50]}...")
    print(f"action: {result['action']}")
    print(f"text: {result['text']}")

    assert result['action'] == 'speak', f"action 应该是 speak，实际：{result['action']}"
    assert '项目进展' in result['text'], f"text 应该包含要说的话，实际：{result['text']}"
    assert '晚上10点' in result['thought'], f"thought 应该包含思考过程，实际：{result['thought']}"
    print("✅ 通过\n")


def test_cot_silent():
    """测试 CoT 格式：保持沉默。"""
    raw = """
嗯...他现在应该在忙吧。
上次打扰他工作，他好像有点不太方便。
还是别打扰了，等他忙完再说。

[决定：沉默]
"""
    result = _parse_monologue_cot(raw)

    print("=== 测试 CoT 沉默 ===")
    print(f"thought: {result['thought'][:50]}...")
    print(f"action: {result['action']}")
    print(f"text: {result['text']}")

    assert result['action'] == 'silent', f"action 应该是 silent，实际：{result['action']}"
    assert result['text'] == '', f"沉默时 text 应该为空，实际：{result['text']}"
    assert '应该在忙' in result['thought'], f"thought 应该包含思考过程"
    print("✅ 通过\n")


def test_cot_look():
    """测试 CoT 格式：看看他。"""
    raw = """
他好像很久没回消息了。
不知道在干什么...
要不看看他在做什么吧。

[决定：看看他]
"""
    result = _parse_monologue_cot(raw)

    print("=== 测试 CoT 看看他 ===")
    print(f"thought: {result['thought'][:50]}...")
    print(f"action: {result['action']}")

    assert result['action'] == 'look', f"action 应该是 look，实际：{result['action']}"
    assert result['text'] == '', f"look 时 text 应该为空"
    print("✅ 通过\n")


def test_cot_no_decision_mark():
    """测试 CoT 格式：没有决策标记（默认沉默）。"""
    raw = """
嗯...不知道说什么好。
就这样吧。
"""
    result = _parse_monologue_cot(raw)

    print("=== 测试 CoT 无决策标记 ===")
    print(f"action: {result['action']}")

    assert result['action'] == 'silent', f"无决策标记应该默认 silent"
    assert '不知道说什么' in result['thought'], f"全文应该作为 thought"
    print("✅ 通过\n")


def test_cot_no_quotes():
    """测试 CoT 格式：开口但没有引号（兼容）。"""
    raw = """
想问问他吃饭了吗。

[决定：开口]
吃饭了吗？
"""
    result = _parse_monologue_cot(raw)

    print("=== 测试 CoT 无引号 ===")
    print(f"action: {result['action']}")
    print(f"text: {result['text']}")

    assert result['action'] == 'speak'
    assert '吃饭' in result['text'], f"应该提取决策标记后的文本"
    print("✅ 通过\n")


def test_parse_monologue_switch():
    """测试 _parse_monologue 的开关切换。"""
    cot_text = "[决定：开口]\n\"测试消息\""
    json_text = '{"thought": "测试", "action": "speak", "text": "测试消息"}'

    print("=== 测试 parse_monologue 开关 ===")

    # use_cot=True 走 CoT 解析
    result_cot = _parse_monologue(cot_text, use_cot=True)
    print(f"CoT 模式: action={result_cot['action']}, text={result_cot['text']}")
    assert result_cot['action'] == 'speak'
    assert '测试消息' in result_cot['text']

    # use_cot=False 走 JSON 解析
    result_json = _parse_monologue(json_text, use_cot=False)
    print(f"JSON 模式: action={result_json['action']}, text={result_json['text']}")
    assert result_json['action'] == 'speak'
    assert '测试消息' in result_json['text']

    print("✅ 通过\n")


def test_cot_emotion_tag():
    """测试 CoT 提取情绪标签（可选）。"""
    raw = """
好想他啊...

[决定：开口]
"想你了呢 <emo:affection>"
"""
    result = _parse_monologue_cot(raw)

    print("=== 测试 CoT 情绪标签 ===")
    print(f"emotion: {result['emotion']}")
    print(f"text: {result['text']}")

    assert result['emotion'] == 'affection', f"应该提取情绪标签"
    assert '<emo:' not in result['text'], f"text 中应该移除情绪标签"
    assert '想你了呢' in result['text']
    print("✅ 通过\n")


def test_cot_japanese_marker():
    """回归 problem 2：回复语言非中文时模型把决策标记写成日文汉字，应能识别且不泄漏进 thought。"""
    raw = "もう夜だね、彼はまだ仕事してるのかな…\n[決定：沈黙]（决定：沉默）"
    result = _parse_monologue_cot(raw)

    print("=== 测试 CoT 日文决策标记 ===")
    print(f"action: {result['action']}")
    print(f"thought: {result['thought']}")

    assert result['action'] == 'silent', f"日文沈黙标记应解析为 silent，实际：{result['action']}"
    assert '決定' not in result['thought'] and '决定' not in result['thought'], \
        f"决策标记不应残留在 thought 里，实际：{result['thought']}"
    assert 'まだ仕事' in result['thought'], f"标记前的思考过程应保留，实际：{result['thought']}"
    print("✅ 通过\n")


def test_cot_marker_only():
    """回归 problem 1/2：模型只吐一个光秃秃的标记、前面无独白，thought 应为空且不含标记残留。"""
    for raw in ("[决定：沉默]", "[決定：沈黙]", "[決定：沈黙]（决定：沉默）", "［决定：沉默］"):
        result = _parse_monologue_cot(raw)
        print(f"=== 仅标记 {raw!r} -> action={result['action']}, thought={result['thought']!r} ===")
        assert result['action'] == 'silent', f"{raw} 应为 silent"
        assert result['thought'] == '', f"仅标记时 thought 应为空，实际：{result['thought']!r}"
    print("✅ 通过\n")


def test_cot_japanese_speak_marker():
    """日文开口标记应识别为 speak 并提取引号内的话。"""
    raw = 'ちょっと心配だな…\n[決定：開口] 「元気にしてる？」'
    result = _parse_monologue_cot(raw)

    print("=== 测试 CoT 日文开口标记 ===")
    print(f"action: {result['action']}, text: {result['text']}")

    assert result['action'] == 'speak', f"日文開口标记应解析为 speak，实际：{result['action']}"
    assert '元気' in result['text'], f"应提取引号内的话，实际：{result['text']}"
    assert '心配' in result['thought'], f"标记前思考应保留"
    print("✅ 通过\n")


if __name__ == "__main__":
    print("=== 测试主动思考链（CoT）解析器 ===\n")

    test_cot_speak()
    test_cot_silent()
    test_cot_look()
    test_cot_no_decision_mark()
    test_cot_no_quotes()
    test_parse_monologue_switch()
    test_cot_emotion_tag()
    test_cot_japanese_marker()
    test_cot_marker_only()
    test_cot_japanese_speak_marker()

    print("=== 全部测试通过 ===")
