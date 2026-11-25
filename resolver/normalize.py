"""Normalize entity names and links for comparison."""

import re
import unicodedata
from urllib.parse import unquote

from .records import DatabaseRecord

LEGAL_FORMS = frozenset(
    [
        "inc",
        "incorporated",
        "ltd",
        "limited",
        "llc",
        "llp",
        "lp",
        "plc",
        "corp",
        "corporation",
        "co",
        "company",
        "gmbh",
        "ag",
        "sa",
        "sas",
        "sarl",
        "bv",
        "nv",
        "ab",
        "as",
        "oy",
        "srl",
        "spa",
        "pte",
        "pty",
    ]
)


def name_key(value: str) -> str:
    decomposed = unicodedata.normalize("NFKD", value)
    unmarked = "".join(char for char in decomposed if not unicodedata.category(char).startswith("M"))
    folded = unmarked.casefold()
    return " ".join("".join(char if char.isalnum() else " " for char in folded).split())


def org_key(value: str) -> str:
    parts = name_key(value).split()
    while len(parts) > 1 and parts[-1] in LEGAL_FORMS:
        parts.pop()
    return " ".join(parts)


def link_key(field: str, value: str) -> str:
    """Normalize a link using the same rules for every field."""
    value = unicodedata.normalize("NFC", unquote(value.strip()))
    value = re.sub(r"^[a-z][a-z0-9+.-]*://", "", value.lower())
    value = re.split(r"[?#]", value, maxsplit=1)[0]
    value = re.sub(r"^www\.", "", value)
    value = re.sub(r"^[a-z]{2}\.linkedin\.com/", "linkedin.com/", value)
    value = re.sub(r"^(mobile\.)?twitter\.com/", "x.com/", value)
    return value.rstrip("/")


def slug_aliases(record: DatabaseRecord) -> set[str]:
    aliases = set()
    domain = link_key("domain", record.links.get("domain", ""))
    if domain:
        aliases.add(name_key(domain.split("/", 1)[0].split(".", 1)[0]))
    prefixes = (
        ("linkedin", "linkedin.com/company/"),
        ("crunchbase", "crunchbase.com/organization/"),
        ("x", "x.com/"),
    )
    for field, prefix in prefixes:
        link = link_key(field, record.links.get(field, ""))
        if not link.startswith(prefix):
            continue
        slug = name_key(link[len(prefix) :].split("/", 1)[0])
        if field == "crunchbase":
            slug = re.sub(r"(?:^| )(?:[0-9]|[0-9a-f]{4})$", "", slug)
        if slug:
            aliases.add(slug)
    return aliases - {""}


def person_name_variants(value: str) -> set[str]:
    key = name_key(value)
    parts = key.split()
    if len(parts) >= 3:
        return {key, f"{parts[0]} {parts[-1]}"}
    return {key}
