import pytest

from resolver.normalize import link_key, name_key, org_key, person_name_variants, slug_aliases
from resolver.records import DatabaseRecord


def test_name_key():
    assert name_key("  Fröß, ＨＡＲＢＯＲ__42! ") == "fross harbor 42"
    assert person_name_variants("Aïla E. Moss") == {"aila e moss", "aila moss"}


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("Frosted Kestrel Company Ltd. LLC", "frosted kestrel"),
        ("GmbH AG", "gmbh"),
        ("Saffron AB Lab", "saffron ab lab"),
    ],
)
def test_org_key(value, expected):
    assert org_key(value) == expected


@pytest.mark.parametrize(
    ("field", "value", "expected"),
    [
        ("domain", " HTTPS://WWW.SAFFRON-OWL.EXAMPLE///?keep=no ", "saffron-owl.example"),
        ("linkedin", "http://de.linkedin.com/company/Moon%2DKite/#about", "linkedin.com/company/moon-kite"),
        ("crunchbase", "ftp://WWW.CRUNCHBASE.COM/organization/North_Swift/", "crunchbase.com/organization/north_swift"),
        ("x", "https://mobile.twitter.com/Velvet_Kite/?s=20", "x.com/velvet_kite"),
        ("x", "twitter.com/Fern%65%CC%81#fragment", "x.com/ferné"),
        ("domain", "  ", ""),
    ],
)
def test_link_key(field, value, expected):
    assert link_key(field, value) == expected


@pytest.mark.parametrize("suffix", ["7", "f3a1"])
def test_slug_aliases(suffix):
    record = DatabaseRecord(
        "db",
        "Quiet Finch",
        "",
        {
            "domain": "https://www.moon-kite.example/path",
            "linkedin": "https://linkedin.com/company/North_Swift/about/",
            "crunchbase": f"https://crunchbase.com/organization/quiet-finch-{suffix}",
            "x": "https://twitter.com/velvet_tern",
        },
    )
    assert slug_aliases(record) == {"moon kite", "north swift", "quiet finch", "velvet tern"}
