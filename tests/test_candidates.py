from resolver.candidates import CandidateIndex
from resolver.decide import decide, is_ambiguous
from resolver.records import NEW, DatabaseRecord, Decision, InputRecord


def test_organization_ranking_uses_alias_and_description():
    database = [
        DatabaseRecord("z", "Hazel Gull", "Designs underwater sonar instruments", {"domain": "opal-wren.example"}),
        DatabaseRecord("a", "Opal Wren", "Bakes sourdough pastries"),
        DatabaseRecord("b", "Opal Wren LLC", "Bakes sourdough pastries"),
        DatabaseRecord("c", "Frosty Lamp", "Studies volcanic ash"),
    ]
    record = InputRecord("one", "Opal Wren Ltd.", "Designs underwater sonar instruments")
    candidates = CandidateIndex(database, "organizations").candidates(record)
    assert [candidate.record.id for candidate in candidates] == ["z", "a", "b"]
    assert candidates[0].features["name_score"] == 100
    assert candidates[0].features["exact_name"]
    assert candidates[0].features["text_score"] > candidates[1].features["text_score"]
    assert candidates[1].features["name_rivals"] == 2
    assert decide(record, candidates, "organizations")[0] == "z"


def test_people_require_organization_evidence():
    database = [
        DatabaseRecord("a", "Mira Thistle", "Chair at Granite Capital Group."),
        DatabaseRecord("b", "Mira B. Thistle", "Partner at Linden Ventures; adviser at Lantern Management."),
    ]
    record = InputRecord("one", "Mira Thistle", "Mira Thistle said Linden and Lantern will invest.")
    index = CandidateIndex(database, "people")
    candidates = index.candidates(record)
    assert {candidate.record.id: candidate.features["org_overlap"] for candidate in candidates} == {"a": 0, "b": 2}
    assert all(candidate.features["name_in_text"] for candidate in candidates)
    assert decide(record, candidates, "people")[0] == "b"

    record = InputRecord("two", "Mira Thistle", "Mira Thistle discussed a capital group and its ventures.")
    candidates = index.candidates(record)
    match_id, confidence, reason = decide(record, candidates, "people")
    assert match_id == NEW
    assert is_ambiguous(candidates, Decision(record.input_id, match_id, confidence, reason))
