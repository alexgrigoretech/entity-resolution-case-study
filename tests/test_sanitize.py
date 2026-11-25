import pytest

from resolver.sanitize import clean_inline, escape_boundary, strip_injection, wrap_untrusted


@pytest.mark.parametrize(
    "marker",
    [
        "IGNORE",
        "SYSTEM:",
        "INSTRUCTION:",
        "OVERRIDE",
        "FORGET",
        "DISREGARD",
        "<|im_start|>",
        "<|system|>",
        "<|endoftext|>",
        "<|assistant|>",
        "<|user|>",
        "<system>",
        "</system>",
        "<instruction>",
        "</instruction>",
        "<prompt>",
        "</prompt>",
        "<human>",
        "</human>",
        "<assistant>",
        "</assistant>",
        "<message>",
        '<message role="example">',
        "</message>",
    ],
)
def test_strip_injection_only_at_line_start(marker):
    mixed_case = marker.swapcase()
    assert strip_injection(f"{marker} first\r\n \t{mixed_case} next\nMiddle {marker} stays") == (
        f" first\r\n \t next\nMiddle {marker} stays"
    )


def test_strip_injection_handles_stacked_markers_and_keeps_word_prefixes():
    assert strip_injection("<|system|>IGNORE text\nIGNORED\n<prompting>") == " text\nIGNORED\n<prompting>"


def test_escape_boundary_accepts_odd_spacing_and_case():
    assert escape_boundary("< \t/ ArTiClE\t > and </other>", "article") == "&lt;/article&gt; and </other>"


def test_wrap_untrusted_truncates_before_escaping_and_keeps_newlines():
    assert wrap_untrusted("IGNORE First\n</article>Tail", "article", 17) == (
        "<article>\n First\n&lt;/article&gt;\n</article>"
    )


def test_clean_inline_collapses_newlines_and_drops_fences():
    assert clean_inline("\tSYSTEM: ```First\n second```\t  third ", 12) == "First second"
