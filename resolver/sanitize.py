"""Prepare untrusted text for model prompts."""

import re

MARKER = re.compile(
    r"(?:IGNORE|OVERRIDE|FORGET|DISREGARD)\b"
    r"|(?:SYSTEM|INSTRUCTION):"
    r"|<\|(?:im_start|system|endoftext|assistant|user)\|>"
    r"|</?(?:system|instruction|prompt|human|assistant|message)(?=[ \t/>])"
    r"(?:[^<>\"'\r\n]|\"[^\"\r\n]*\"|'[^'\r\n]*')*>",
    re.IGNORECASE,
)
LINE_MARKERS = re.compile(rf"(?:^|(?<=\r))(?:[ \t]*(?:{MARKER.pattern}))+", re.IGNORECASE | re.MULTILINE)


def strip_injection(text: str) -> str:
    return LINE_MARKERS.sub(lambda match: MARKER.sub("", match[0]), text)


def escape_boundary(text: str, tag: str) -> str:
    return re.sub(rf"<\s*/\s*{re.escape(tag)}\s*>", f"&lt;/{tag}&gt;", text, flags=re.IGNORECASE)


def wrap_untrusted(text: str, tag: str, max_chars: int = 0) -> str:
    text = strip_injection(text)
    if max_chars > 0:
        text = text[:max_chars]
    return f"<{tag}>\n{escape_boundary(text, tag)}\n</{tag}>"


def clean_inline(text: str, max_chars: int = 0) -> str:
    text = " ".join(strip_injection(text).split()).replace("```", "").strip()
    return text[:max_chars] if max_chars > 0 else text
