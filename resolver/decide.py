"""Make deterministic match decisions from candidate evidence."""

from . import config
from .candidates import Candidate, Features
from .records import NEW, Decision, InputRecord, Kind


def _organization_reason(features: Features) -> tuple[str, float] | None:
    name = features["name_score"]
    text = features["text_score"]
    if features["exact_name"] and text >= config.ORG_TEXT_EXACT:
        return "exact_name+text", config.ORG_TEXT_EXACT
    if name >= config.ORG_NAME_FUZZY and text >= config.ORG_TEXT_FUZZY:
        return "fuzzy_name+text", config.ORG_TEXT_FUZZY
    if text >= config.ORG_TEXT_ALONE and name >= config.ORG_NAME_WITH_TEXT:
        return "text+name", config.ORG_TEXT_ALONE
    return None


def decide(input_record: InputRecord, candidates: list[Candidate], kind: Kind) -> tuple[str, float, str]:
    if not candidates:
        return NEW, 1.0, "no_candidate"
    if kind == "organizations":
        passing = []
        for candidate in candidates:
            rule = _organization_reason(candidate.features)
            if rule is not None:
                reason, threshold = rule
                passing.append((candidate, reason, threshold))
        if passing:
            candidate, reason, threshold = max(passing, key=lambda item: item[0].features["text_score"])
            confidence = min(1.0, 0.5 + candidate.features["text_score"] - threshold)
            return candidate.record.id, confidence, reason
    else:
        passing_people = [item for item in candidates if item.features["name_score"] >= config.PERSON_NAME_SCORE]
        if passing_people:
            candidate = max(passing_people, key=lambda item: item.features.get("org_overlap", 0))
            overlap = candidate.features.get("org_overlap", 0)
            if overlap >= config.PERSON_ORG_OVERLAP:
                return candidate.record.id, min(1.0, 0.5 + 0.15 * (overlap - config.PERSON_ORG_OVERLAP)), "name+org"
    return NEW, 1 - candidates[0].features["name_score"] / 100, "insufficient_evidence"


def is_ambiguous(candidates: list[Candidate], decision: Decision) -> bool:
    if decision.match_id == NEW:
        return any(item.features["name_score"] >= config.AMBIGUOUS_NAME_SCORE for item in candidates)
    return decision.confidence < config.AMBIGUOUS_BELOW
