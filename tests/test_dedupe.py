from resolver.dedupe import dedupe
from resolver.records import NEW, Decision, InputRecord


def test_organization_groups_follow_input_order_and_skip_matches():
    rows = [
        InputRecord("z", "Moss Marlin Ltd", "Marine sensors track ocean tides and seabed movement."),
        InputRecord("a", "Moss Marlin", "Marine sensors track ocean tides and seabed movement."),
        InputRecord("middle", "Moss Marlin LLC", "Handmade pastries and baked chocolate biscuits."),
        InputRecord("b", "Moss Marlin Inc", "Marine sensors track ocean tides."),
        InputRecord("earliest_id", "Moss Marlin Sensors", "Marine sensors track ocean tides and seabed movement."),
    ]
    decisions = [Decision(row.input_id, NEW, 1.0, "fixture") for row in rows]
    decisions[1].match_id = "existing"
    assert dedupe(rows, decisions, "organizations") == 1
    assert [decision.dup_of for decision in decisions] == ["", "", "", "z", "z"]


def test_people_groups_require_shared_context_near_a_name():
    rows = [
        InputRecord("first", "Pera Alder", "Pera Alder of Verdant Granite Beacon Ventures shared the news."),
        InputRecord("second", "Pera B. Alder", "Verdant Granite Beacon Group partner Pera B. Alder signed the letter."),
        InputRecord("third", "Pera Alder", "Pera Alder of Crimson Nest Books wrote another novel."),
    ]
    decisions = [Decision(row.input_id, NEW, 1.0, "fixture") for row in rows]
    assert dedupe(rows, decisions, "people") == 1
    assert [decision.dup_of for decision in decisions] == ["", "first", ""]
