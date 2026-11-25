"""Run offline resolution and optional network stages for each input type."""

import asyncio
import logging
from collections.abc import Awaitable, Callable
from dataclasses import replace
from pathlib import Path

from . import config, links, research
from .adjudicate import adjudicate, fallback
from .candidates import CandidateIndex, passes_lexical
from .decide import decide, is_ambiguous
from .dedupe import dedupe_async
from .embeddings import embed_matrix, text_for
from .httpcache import RunClient
from .links import select
from .llm import OpenRouter
from .records import NEW, Decision, Kind, read_database, read_inputs
from .report import Row
from .search import Serper

LOGGER = logging.getLogger(__name__)


async def process_kind(
    db_path: Path,
    input_path: Path,
    kind: Kind,
    client: RunClient,
    router: OpenRouter,
    serper: Serper,
    research_usage: research.ResearchUsage,
) -> tuple[list[Row], dict[str, str], int]:
    database = read_database(db_path, kind)
    records = read_inputs(input_path, kind)
    index = CandidateIndex(database, kind)
    links_enabled = router.chat_enabled and bool(serper.api_key)
    links_index = links.index_links(database, kind) if links_enabled else {}
    matrix = inputs = None
    if router.embed_enabled and database and records and not client.expired():
        matrix = await embed_matrix(router, [text_for(record) for record in database])
        if matrix is not None and not client.expired():
            inputs = await embed_matrix(router, [text_for(record) for record in records])

    rows = []
    for position, record in enumerate(records):
        try:
            vector = inputs[position] if inputs is not None else None
            candidates = index.candidates(record, vector, matrix)
            lexical = [
                candidate
                for candidate in candidates
                if passes_lexical(candidate.features["name_score"], candidate.features["text_score"])
            ][: config.TOP_K]
            offline = Decision(record.input_id, *decide(record, lexical, kind))
            rows.append(Row(record, candidates, offline, offline, is_ambiguous(lexical, offline)))
        except Exception as error:  # noqa: BLE001
            LOGGER.warning("Row %s failed (%s)", record.input_id, type(error).__name__)
            offline = Decision(record.input_id, NEW, 0.0, "offline_error")
            rows.append(Row(record, [], offline, fallback(offline), False, outcome="row_failed"))

    searches: dict[tuple[str, ...], asyncio.Task[links.LinkOptions]] = {}
    selections: dict[tuple[str, ...], asyncio.Task[dict[str, links.Selection]]] = {}
    responses: dict[tuple[str, ...], asyncio.Task[research.ResearchResult | None]] = {}
    researched_rows: list[tuple[Row, tuple[str, ...]]] = []

    async def search(row: Row) -> None:
        if client.expired():
            return
        queries = tuple(links.search_queries(row.record, kind, row.decision.employer))
        if queries not in searches:
            searches[queries] = asyncio.create_task(links.collect(serper, list(queries), kind))
        hits = await searches[queries]
        row.options = hits if row.options is None else links.merge_options(row.options, hits)

    async def match(row: Row, position: int) -> None:
        if client.expired():
            row.decision = fallback(row.offline, "deadline")
            return
        hits = links.link_evidence(row.options or {}, links_index)
        if hits:
            vector = inputs[position] if inputs is not None else None
            row.candidates = index.candidates(row.record, vector, matrix, hits)
        row.attempted = bool(row.candidates)
        row.decision = await adjudicate(router, row.record, row.candidates, kind, row.offline)
        row.decision.model = router.chat_model(row.decision.model)
        if client.expired() and not row.decision.model:
            row.decision = replace(
                fallback(row.offline, "deadline"),
                fast_confidence=row.decision.fast_confidence,
                strong_confidence=row.decision.strong_confidence,
                escalated=row.decision.escalated,
            )

    async def enrich(row: Row) -> None:
        if client.expired():
            row.outcome = "skipped"
            return
        record, decision = row.record, row.decision
        key = record.name, record.text, decision.employer, decision.role
        if key not in selections:
            selections[key] = asyncio.create_task(select(router, record, decision, kind, row.options or {}))
            if config.RESEARCH_ENABLED:
                responses[key] = asyncio.create_task(
                    research.respond(router, serper, record, decision, kind, research_usage)
                )
        if key in responses:
            researched_rows.append((row, key))
        row.selections = await selections[key]
        if client.expired():
            row.outcome = "skipped"
        for name, selection in row.selections.items():
            if selection.confidence is not None and (url := selection.emitted(name)):
                decision.links[name] = url
                decision.link_confidences[name] = selection.confidence

    async def safe(row: Row, work: Awaitable[None], *, link_stage: bool = False) -> None:
        try:
            await work
        except Exception as error:  # noqa: BLE001
            LOGGER.warning("Row %s failed (%s)", row.record.input_id, type(error).__name__)
            if link_stage:
                row.outcome = "link_failed"
                row.selections = {}
                row.decision.links = {}
                row.decision.link_confidences = {}
            else:
                row.outcome = "row_failed"
                row.decision = fallback(row.offline)

    async def search_pending(eligible: Callable[[Row], bool], *, link_stage: bool = False) -> None:
        if serper.api_key:
            await asyncio.gather(
                *(
                    safe(row, search(row), link_stage=link_stage)
                    for row in rows
                    if row.outcome == "ok" and eligible(row)
                )
            )

    def needs_search(row: Row) -> bool:
        return row.decision.match_id == NEW and (
            row.options is None or links.employer_refines_search(row.record, kind, row.decision.employer)
        )

    await search_pending(lambda row: row.offline.match_id == NEW or row.ambiguous)
    if router.chat_enabled or (router.api_key and client.expired()):
        await asyncio.gather(
            *(safe(row, match(row, position)) for position, row in enumerate(rows) if row.outcome == "ok")
        )
    await search_pending(needs_search, link_stage=True)
    if links_enabled:
        await asyncio.gather(
            *(
                safe(row, enrich(row), link_stage=True)
                for row in rows
                if row.outcome == "ok" and row.decision.match_id == NEW
            )
        )

    if responses:
        results = dict(zip(responses, await asyncio.gather(*responses.values(), return_exceptions=True), strict=True))
        for row, key in researched_rows:
            result = results[key]
            if isinstance(result, BaseException):
                LOGGER.warning("Research for row %s failed (%s)", row.record.input_id, type(result).__name__)
            elif row.outcome == "ok":
                row.research = result
    rejected_domains: set[int] = set()
    for row in rows:
        row.apply_research()
        if row.research and (domain := row.research.selections.get("domain")) and domain.confirmation_only:
            combined = row.combined.get("domain")
            if combined is None or combined.agreement != "both":
                rejected_domains.add(id(row.research))
    research_usage.research_rejected += len(rejected_domains)

    decisions = [row.decision for row in rows]
    groups = await dedupe_async(
        records, decisions, kind, router if router.chat_enabled and not client.expired() else None
    )
    return rows, {record.id: record.name for record in database}, groups
