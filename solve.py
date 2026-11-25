"""Write entity resolution submissions and review files."""

import argparse
from pathlib import Path

from resolver.preflight import PreflightResult
from resolver.runner import DEFAULT_CACHE, active_stages, run


def _print_preflight(checks: dict[str, PreflightResult]) -> None:
    for model, result in checks.items():
        if not result["ok"]:
            status = f"HTTP {result['status']}" if result.get("status") is not None else "request failed"
            print(f"Model preflight failed for {model} ({status}) {result['error']}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, required=True, help="Directory containing database CSVs")
    parser.add_argument("--input", type=Path, required=True, help="Directory containing incoming CSVs")
    parser.add_argument("--out", type=Path, required=True, help="Directory for submissions and statistics")
    parser.add_argument("--no-network", action="store_true", help="Disable network access")
    parser.add_argument("--no-report", action="store_true", help="Skip the HTML review report")
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE, help="SQLite response cache path")
    args = parser.parse_args()
    network = not args.no_network
    stages = ", ".join(f"{name} {'on' if enabled else 'off'}" for name, enabled in active_stages(network).items())
    print(f"Network stages: {stages}")
    try:
        stats = run(
            args.db,
            args.input,
            args.out,
            network=network,
            cache_path=args.cache,
            include_report=not args.no_report,
            on_preflight=_print_preflight,
        )
    except FileNotFoundError as error:
        parser.error(str(error))
    for kind, counts in stats.items():
        print(
            f"{kind}: {counts['rows']} rows, {counts['matched']} matched, {counts['new']} new, "
            f"{counts['dup_groups']} duplicate groups, {counts['ambiguous']} ambiguous"
        )


if __name__ == "__main__":
    main()
