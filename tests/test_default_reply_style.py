"""统一默认回复风格测试。"""

from core.character import CharacterCard, DEFAULT_REPLY_STYLE, build_reactive_prompt


def test_default_style_without_card_instruction():
    system, _ = build_reactive_prompt(CharacterCard(name="测试角色"), "用户", "你好", [])
    assert DEFAULT_REPLY_STYLE in system


def test_default_style_not_duplicated():
    card = CharacterCard(name="测试角色", post_history_instructions=DEFAULT_REPLY_STYLE)
    system, _ = build_reactive_prompt(card, "用户", "你好", [])
    assert system.count(DEFAULT_REPLY_STYLE) == 1


if __name__ == "__main__":
    test_default_style_without_card_instruction()
    test_default_style_not_duplicated()
    print("test_default_reply_style: all passed")
