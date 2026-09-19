from pytest import approx

from sodalite_backend.inference.prompt_weights import has_attention_syntax, parse_prompt_attention


def test_parse_prompt_attention_leaves_plain_text_unchanged() -> None:
    fragments = parse_prompt_attention("a cat")

    assert [(fragment.text, fragment.weight) for fragment in fragments] == [("a cat", 1.0)]
    assert not has_attention_syntax("a cat")


def test_parse_prompt_attention_supports_emphasis_and_deemphasis() -> None:
    fragments = parse_prompt_attention("a (cat) and [dog]")

    assert [(fragment.text, fragment.weight) for fragment in fragments] == [
        ("a ", 1.0),
        ("cat", approx(1.1)),
        (" and ", 1.0),
        ("dog", approx(1 / 1.1)),
    ]


def test_parse_prompt_attention_supports_explicit_and_nested_weights() -> None:
    fragments = parse_prompt_attention("(red (cat:1.3))")

    assert [(fragment.text, fragment.weight) for fragment in fragments] == [
        ("red ", approx(1.1)),
        ("cat", approx(1.1 * 1.3)),
    ]
