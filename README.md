# Entity resolver

Match incoming organization and people records against an existing database, group duplicates among new records, and find links for new entities.

## Run and grade

```bash
uv run solve.py --db data --input data --out out
uv run python data/grade.py out
uv run python tools/evaluate.py --out out --key data
uv run pytest -q
```

The submissions for the case-study input are in `submission/`, next to the review report of that run. The write-up is `WRITEUP.md`.

Set `OPENROUTER_API_KEY` to enable embeddings, model adjudication and model-assisted duplicate checks. Set `SERPER_API_KEY` to enable search. Both keys are needed to select links. With both keys absent or the `--no-network` flag set, the pipeline uses local rules and leaves link fields blank. The CLI prints which network stages are configured.

Set `RESOLVER_MODEL_FAST`, `RESOLVER_MODEL_STRONG` or `RESOLVER_MODEL_EMBED` to override the OpenRouter model IDs.

With an OpenRouter key, the run checks each configured model before processing rows. A working chat model covers both tiers if the other fails. Failed checks are printed and included in `run_stats.json`. Chat requests allow up to 800 output tokens, or 1,600 for research, and default to low reasoning effort. Set `RESOLVER_REASONING_EFFORT` to override the effort or set it to an empty string to omit the field. Preflight retries transient failures and unusable HTTP 200 responses up to three total attempts. After HTTP 400 or 422, or when the probe exhausts its attempts on an unusable HTTP 200, preflight tries again without reasoning. If that succeeds, later requests omit the field. A truncated response ends a normal chat request and is logged once.

Model prompts clean short evidence fields and wrap descriptions and articles in data tags.

Homepage link selection prioritizes organization profiles from JSON-LD `sameAs` and `twitter:site` metadata.

Set `RESOLVER_QUOTE_QUERIES` to `1`, `true` or `yes` to quote entity names and known employers in search queries.

With both API keys set, the research responder runs a separate set of quoted searches for new entities and asks one model to choose URLs from those results. Agreement raises link confidence. When both answers clear the confidence threshold and disagree, the field is left blank for that row before duplicate groups share links, and the row is flagged for review. Set `RESOLVER_RESEARCH` to `0`, `false` or `no` to disable the responder, or set `RESOLVER_MODEL_RESEARCH` to override its model, which defaults to the strong model.

Usable network responses are cached in `cache/cache.sqlite` next to `solve.py`. Pass `--cache` with a path to use a different file. The network deadline is 540 seconds. Organizations and people run concurrently within that deadline. The pipeline writes the submissions, review queue and run statistics after the network stages finish or reach the deadline.

Open `out/report.html` to review decisions, candidate evidence and links. Pass `--no-report` to skip it. The offline evaluator writes `evaluation.md` with confidence calibration and cross-validated link thresholds. The `--folds` and `--seed` options default to 5 and 7.

## Directory map

- `solve.py` runs the pipeline and prints a summary.
- `resolver/` contains CSV I/O, normalization, candidate ranking, decisions, deduplication, link enrichment, review output, and HTTP caching.
- `resolver/runner.py` runs the per-type pipeline and writes submissions, statistics, and review outputs.
- `resolver/report.py` embeds run data in the HTML review template.
- `tools/evaluate.py` calibrates completed runs against the key offline.
- `tools/link_agreement.py` replays link policies and reports agreement.
- `tests/` exercises matching, pipeline output, and HTTP behavior with synthetic data.
- `data/` contains the case-study data and grader.
- `cache/cache.sqlite` is the shipped cache of every search, embedding and model response of the submitted run.
- `submission/` holds the two submission files and `report.html` from that run.
- `out/` receives the submissions, `review_queue.csv`, `run_stats.json`, `report.html`, `evaluation.md`, and grade reports.
