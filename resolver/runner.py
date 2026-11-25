"""Run the resolver and write submissions, review data and statistics."""

import asyncio
import json
import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import asdict
from pathlib import Path
from typing import Any

import httpx

from . import config
from .combine import AGREEMENTS
from .httpcache import Cache, RunClient
from .llm import ModelLedger, OpenRouter
from .pipeline import process_kind
from .preflight import PreflightResult, network_keys, preflight
from .preflight import active_stages as base_stages
from .records import KINDS, NEW, Kind, write_submission
from .report import Row, build_payload, render
from .research import ResearchUsage
from .review import ReviewEntry, write_review
from .search import Serper

LOGGER = logging.getLogger(__name__)
DEFAULT_CACHE = Path(__file__).resolve().parents[1] / "cache" / "cache.sqlite"
REPORT_TEMPLATE = Path(__file__).with_name("report_template.html")


def active_stages(network: bool, router: OpenRouter | None = None) -> dict[str, bool]:
    stages = base_stages(network, router)
    return {**stages, "research": stages["links"] and config.RESEARCH_ENABLED}


def summarize_agreement(stats: dict[Kind, dict[str, Any]]) -> dict[str, dict[str, int]]:
    totals = {name: dict.fromkeys(AGREEMENTS, 0) for name in config.LINK_MIN_CONF}
    for kind in stats.values():
        for row in kind["decisions"]:
            for name, agreement in row["agreement"].items():
                totals[name][agreement] += 1
    return totals


def summarize_run(ledger: ModelLedger, entries: list[ReviewEntry], attempted: int) -> dict[str, Any]:
    escalations = sum(decision.escalated for _, _, decision, _ in entries)
    network = ledger.to_dict()
    return {
        "network": network,
        "models": ledger.by_model(),
        "serper_queries": network.get("serper", {}).get("calls", 0),
        "escalation_count": escalations,
        "escalation_rate": escalations / attempted if attempted else 0.0,
        "disagreement_count": sum(decision.models_disagree for _, _, decision, _ in entries),
        "links_emitted": {
            field: sum(bool(decision.links.get(field)) for _, _, decision, _ in entries)
            for field in config.LINK_MIN_CONF
        },
    }


def run(
    db_dir: str | Path,
    input_dir: str | Path,
    out_dir: str | Path,
    network: bool = True,
    cache_path: str | Path = DEFAULT_CACHE,
    *,
    transport: httpx.AsyncBaseTransport | None = None,
    include_report: bool = True,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    on_preflight: Callable[[dict[str, PreflightResult]], None] | None = None,
) -> dict[Kind, dict[str, Any]]:
    started = clock()
    cache = Cache(cache_path)
    try:
        client = RunClient(cache, ModelLedger(), started + config.DEADLINE_SECONDS, transport, clock=clock, sleep=sleep)
        return asyncio.run(
            _run(Path(db_dir), Path(input_dir), Path(out_dir), network, client, started, include_report, on_preflight)
        )
    finally:
        cache.close()


async def _run(
    db_dir: Path,
    input_dir: Path,
    out_dir: Path,
    network: bool,
    client: RunClient,
    started: float,
    include_report: bool,
    on_preflight: Callable[[dict[str, PreflightResult]], None] | None,
) -> dict[Kind, dict[str, Any]]:
    stats: dict[Kind, dict[str, Any]] = {}
    rows_by_type: dict[Kind, tuple[list[Row], dict[str, str]]] = {}
    entries: list[ReviewEntry] = []
    router_key, serper_key = network_keys(network)
    router, serper = OpenRouter(client, router_key), Serper(client, serper_key)
    research_usage = ResearchUsage()
    link_flags = {}
    attempted = 0
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        jobs = {}
        for kind in KINDS:
            input_path = input_dir / f"input_{kind}.csv"
            if not input_path.is_file():
                continue
            db_path = db_dir / f"existing_{kind}.csv"
            if not db_path.is_file():
                raise FileNotFoundError(f"Missing database file: {db_path}")
            jobs[kind] = db_path, input_path
        checks = await preflight(router)
        if checks and on_preflight is not None:
            on_preflight(checks)
        results = await asyncio.gather(
            *(
                process_kind(db, inputs, kind, client, router, serper, research_usage)
                for kind, (db, inputs) in jobs.items()
            )
        )
        for kind, (rows, names, groups) in zip(jobs, results, strict=True):
            decisions = [row.decision for row in rows]
            write_submission(out_dir / f"submission_{kind}.csv", kind, decisions)
            matched = sum(decision.match_id != NEW for decision in decisions)
            stats[kind] = {
                "rows": len(rows),
                "matched": matched,
                "new": len(rows) - matched,
                "dup_groups": groups,
                "ambiguous": sum(row.ambiguous for row in rows),
                "decisions": [row.report() for row in rows],
            }
            attempted += sum(row.attempted for row in rows)
            entries.extend((kind, row.record, row.decision, names.get(row.decision.match_id, "")) for row in rows)
            link_flags.update({(kind, row.record.input_id): row.link_flags() for row in rows if row.combined})
            if include_report:
                rows_by_type[kind] = rows, names
    finally:
        await client.aclose()

    report = {
        "types": stats,
        "active_stages": active_stages(network, router),
        "preflight": checks,
        **summarize_run(client.ledger, entries, attempted),
        **asdict(research_usage),
        "link_agreement": summarize_agreement(stats),
        "review_queue_size": write_review(out_dir / "review_queue.csv", entries, link_flags),
        "deadline_hit": bool(router_key or serper_key) and client.expired(),
        "wall_seconds": client.clock() - started,
    }
    (out_dir / "run_stats.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if include_report:
        if REPORT_TEMPLATE.is_file():
            render(build_payload(report, rows_by_type), REPORT_TEMPLATE, out_dir / "report.html")
        else:
            LOGGER.warning("Report template is missing. Skipping the HTML review report.")
    return stats
