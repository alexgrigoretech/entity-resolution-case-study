# Entity resolution case study: write-up

Alex Grigore. Every number below is measured on the visible set (160 rows, 237 points).

## 1. Directory map and process

| Path | What it holds |
|---|---|
| `solve.py` | The one entry command: `uv run solve.py --db <dir> --input <dir> --out <dir>`. |
| `resolver/normalize.py`, `candidates.py`, `embeddings.py` | Name and link normalization, then a shortlist of database rows per input from name similarity, TF-IDF text similarity, embeddings and old names recovered from link slugs. |
| `resolver/decide.py`, `adjudicate.py` | The offline rule and the LLM cascade that makes the match or NEW decision. |
| `resolver/search.py`, `links.py`, `research.py`, `combine.py` | Serper queries, one fetch of the entity's own homepage, link selection, and a second independent link answer that is merged with the first. |
| `resolver/dedupe.py` | Groups repeated new entities and sets `dup_of`. |
| `resolver/httpcache.py`, `llm.py`, `sanitize.py` | SQLite response cache, retries with backoff, spend ledger, OpenRouter client, prompt sanitizing. |
| `resolver/review.py`, `report.py`, `report_template.html` | Review queue CSV and a self-contained HTML review page. |
| `tools/evaluate.py` | Self-grading only: calibration tables and cross-validated thresholds. Never imported by the solve path. |
| `tests/` | 217 tests, all on synthetic records and mock transports. |
| `cache/cache.sqlite` | Shipped cache of every search, embedding and LLM response. |

**Process.** Each input row gets a shortlist of at most eight database rows. Names alone are not enough here. On the visible set 12 organizations match a row with a different name, and 7 inputs share an exact name with a row that is a different entity. So the shortlist also uses the database's own link slugs as aliases (a row named "axio" still carries a Crunchbase slug with its former name) and embeddings of the descriptions.

Rows that the offline rule cannot settle are searched first. If a LinkedIn or Crunchbase URL found for the input already belongs to a database row, that row joins the shortlist with a link-hit flag. This turns enrichment into match evidence and is what catches renames.

A fast model then reads the input text against the shortlist and returns an offered id or NEW with a confidence. Below `0.8` the same prompt goes to a stronger model. The model can only return an id it was offered, and when both calls fail the offline rule stands, so one failed call never stops a run.

Links are chosen by index from classified search hits and the homepage's JSON-LD `sameAs`, so a URL cannot be invented. A wrong link costs as much as a right one earns, so a field is emitted only above a per-field confidence threshold and left blank otherwise. For new entities a second responder answers the same question from its own quoted queries. When the two agree the link is emitted, and when they disagree the field stays blank and the row is flagged for review. Where both answered, every link was right (108 of 108).

**How I kept it honest.** No rule mentions a specific name, and the answer key is read only by the grader and `tools/evaluate.py`. Thresholds are cross-validated: held-out folds score 193 link points against 197 in-sample, and they did not move any threshold, so the shipped values are the ones chosen before the first live run. Results on the visible set: offline rules only 54.0%, full pipeline 91.3% (216.33 of 237; organizations 91.9%, people 90.1%). A cold-cache run takes 252 s against the 10 minute limit and costs 0.75 USD (0.005 per row) plus 774 search queries, with 1.25% of rows escalated. Remaining errors by cause: 40 missed links, 5 wrong links, 2 links where none exists, 1 wrong merge and 1 missed duplicate.

**What I would not trust yet.** Link recall. 29 of the 40 missed links never appeared in any search result, so no threshold recovers them; they need different queries. The weakest fields are X handles for people (63.6% recall) and organization domains (77.1%). The domain query is the name plus the description, and for 6 of the 10 misses no search result contained the homepage. I also would not read the 100% decision accuracy above 0.9 confidence as a property of the system, because it is 153 rows from one dataset.

## 2. Production system design

**Shape.** Inbound records land on a queue with an idempotency key (hash of type, name, text and source), so replays and retries are harmless. Workers run the same stages as this repository, but each stage is its own queue so a slow search provider cannot hold up matching.

**Matching and dedupe.** The in-memory shortlist becomes three indexed lookups in Postgres: exact match on normalized link keys (unique index), trigram similarity on names and aliases (`pg_trgm`), and nearest neighbours on description embeddings (`pgvector`). In a stream, dedupe stops being a separate step. A record judged new is written at once as a provisional entity, so its twin arriving a second later simply matches it. Two twins processed concurrently are serialized by an advisory lock on the blocking key, and the upsert is idempotent.

**Decision tiers.** Cheapest first: deterministic rules settle the clear cases, a fast model takes the ambiguous ones, a strong model takes what the fast one is unsure about, and what is left goes to a person. On this dataset I sent every row to the fast model, which settled 158 of 160; 2 went to the strong model and 3 landed in the review queue. At volume the rule tier has to take the clear cases first. Today the rules alone score 54.0%, so here they are the fallback when a model call fails, and earning that tier is the first piece of production work. Tier thresholds come from the calibration tables and are re-derived when the data shifts.

**Enrichment.** Link finding is asynchronous and decoupled from the match decision, because a match needs no links and a new entity is usable before its links arrive. Per-provider concurrency limits and retry with backoff are already in the client. In production I would add a time-to-live on cached search results, a dead-letter queue after the last retry, and a re-check job, since links rot and companies rename.

**Cost and latency.** Measured here: 0.005 USD and 4.8 search queries per row (a new entity costs four more queries than a match, for the second link answer), and 252 s of wall-clock for 160 rows run concurrently. Levers, in order: skip the model when rules are confident, cache by normalized entity instead of by raw query, send non-urgent traffic through batch pricing, and keep embeddings for database rows precomputed so only inputs are embedded.

**Human review.** Every decision carries a confidence, a reason and flags, and the review queue is sorted so the least certain rows come first. The HTML page in this repository is the prototype: decision, evidence, candidates, links, and a corrections export. In production the queue is ranked by expected cost of being wrong, and a wrong merge ranks above a missed match because it corrupts an existing record.

**Corrections feed back three ways.** At once, an accepted merge adds the input name as an alias of the entity, and a rejected pair becomes a do-not-merge constraint the matcher checks before any model call. On a schedule, reviewed rows join a labelled set that recalibrates the thresholds and runs as a regression suite before any prompt or model change ships. And as monitoring, the review overturn rate per tier and per field is the health metric, so a rising rate means a threshold or a prompt has drifted.

**Reversibility.** Source records are never overwritten. Merges are rows in a log, so a wrong merge is undone by splitting, and every entity can be rebuilt from its sources.

I run the same pattern (search, homepage fetch with per-host limits, model enrichment with cost tracking) in production in Kustiq, a product of mine, which is where the sanitizing and structured-data steps here come from.
