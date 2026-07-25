# Museum AI Search

Natural-language search over a museum collection. A visitor types
`"big meat-eating dinosaur"` or `"T rex"` and gets the right specimen back.

Prototype stage: the collection is 12 hand-written records in
[`app/specimens.py`](app/specimens.py). Everything else is the real pipeline.

## How it works

Three stages, and the middle one is the point.

**1. Enrich (offline, once per specimen).** Claude reads a catalog record and
writes back the words a visitor would actually type — `"T. rex"`,
`"tyrant lizard"`, `"large carnivorous dinosaur"`, `"predator"` — plus a
visitor-facing `category` (`fossil`, not `Animalia`). Costs one API call per
specimen, ever. Cached in `app/data/enriched.json`.

**2. Embed (at startup).** `sentence-transformers` encodes the *enriched* text,
not the raw catalog row. Vectors are cached and re-encoded only when the text
changes.

Measured on this collection with `all-MiniLM-L6-v2`:

| query | raw catalog text | enriched |
|---|---|---|
| `"T rex"` | *T. rex* at rank 1 (0.60) | rank 1 (0.69) |
| `"large predator"` | a **deer** at rank 1; no wolf in top 3 | *Canis lupus* at rank 1 |

`"T rex"` works either way — the tokenizer splits `Tyrannosaurus rex` and the
shared `rex` subword does the job. Enrichment earns its keep on `"large
predator"`, where nothing in the catalog row connects a wolf to the concept.
That is the case keyword search can never reach.

**Keep `search_text` short.** MiniLM mean-pools token vectors, so every
off-concept word dilutes the result. A 34-word prose paragraph about T. rex
scores 0.395 on `"big meat-eating dinosaur"` — losing to an unenriched
*Apatosaurus* (0.464), a herbivore. A 10-word alias phrase with the same facts
scores 0.583. Locality and collector are deliberately excluded from the embedded
text: they're already matched exactly by the structured filters, so embedding
them only costs ranking.

**3. Search (per request).** Claude turns the query into structured filters
against a schema, so `kingdom` is always one of the three values the catalog
actually uses. Then:

- `kingdom` and `category` **exclude** non-matches. Asking for minerals and
  getting a wolf at rank 4 is worse than getting fewer results.
- `family` / `genus` / `species` / `locality` / `collector` **boost**.
- Cosine similarity over the enriched text supplies the rest.

### Two models, on purpose

`app/ai.py` sets `ENRICH_MODEL = "claude-opus-4-8"` and
`QUERY_MODEL = "claude-haiku-4-5"`. The jobs have opposite cost curves and
opposite blast radii.

| | enrichment | query parsing |
|---|---|---|
| how often | once per specimen, ever | every search, forever |
| dominates lifetime cost? | no | **yes** |
| a mistake… | is frozen into `enriched.json` and silently skews every future search | affects one search; the visitor retypes |
| latency matters? | no, it's offline | **yes, it's in front of a search box** |
| task | open-ended generation grounded in taxonomy | extraction into a schema with three legal `kingdom` values |

For 50k specimens and 10k searches/month, downgrading *enrichment* to Haiku saves
~$125 once and degrades every search permanently. Downgrading *query parsing*
saves ~$384/year and the errors evaporate. So: spend where mistakes compound,
economize where they don't.

**Haiku's parse quality here is untested.** The reasoning is from task shape, not
evidence. Before trusting it, run ~30 realistic visitor queries through both
models and diff the extracted filters — `"specimens Emily Davis collected in
Georgia"` and `"minerals that look like gold"` are the cases where a weaker model
might drop `collector` or invent a `kingdom`. If it fails, `claude-sonnet-5` is
the middle rung.

### Why enrichment runs offline

Enrichment cost scales with the size of the collection (once), not with traffic.
A 50,000-specimen catalog is a single batch job; every search afterwards is a
vector lookup. Enriching per request would put an LLM call on the hot path for
zero additional recall.

At real collection size, swap the loop in `app/enrich.py` for the
[Batches API](https://platform.claude.com/docs/en/build-with-claude/batch-processing)
— same prompt, half the cost.

### Provenance

Everything under `enrichment` is model-generated. It exists so records can be
*found*, and it is never returned to the client. `/api/ai-search` responds with
the museum's own catalog fields only.

This is a deliberate boundary, not an oversight. A hallucinated common name that
merely mis-ranks a search result is a bug; the same string rendered in a museum's
UI as though the institution asserted it is a different kind of problem. The
enrichment prompt forbids inventing dates, collectors, localities, and
provenance, and `app/search.py` never puts enrichment on the wire.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt

cp .env.example .env            # then add your key
```

`sentence-transformers` pulls in torch (~2.5 GB). Without it the app still runs
and falls back to lexical matching over the enriched text — enrichment alone is
enough to make `"T rex"` work. You lose conceptual matches like
`"large predator"` → *Canis lupus*.

## Run

```bash
python -m app.enrich            # once; writes app/data/enriched.json
uvicorn app.main:app --reload
```

`app.enrich` skips specimens it has already enriched, so re-running after adding
records only pays for the new ones.

```bash
curl -X POST http://127.0.0.1:8000/api/ai-search \
  -H "Content-Type: application/json" \
  -d '{"query": "T rex"}'
```

`GET /` reports whether enrichment and semantic search are actually active.

### Degraded mode

Structured filter extraction is the only part of search that calls Claude. The
embedding index and the scoring are local. So when the API is unreachable —
outage, rate limit, expired key, unfunded account — search **degrades instead of
failing**: the raw query goes straight to the semantic half, and the response
comes back `200` with

```json
{ "degraded": true, "degraded_reason": "rate_limited", "results": [ ... ] }
```

You lose `"collected by Emily Davis in Georgia"` (that needs filter extraction).
You keep `"shiny gold rock"` → *Pyrite*. Surface `degraded` in the UI — a visitor
whose filters were silently dropped should be told.

This means you can run and evaluate the whole ranking pipeline **without an API
key**, which is useful while you're still waiting on credits.

**The cost of this design:** an invalid key degrades silently and indefinitely.
Nothing breaks, so nobody notices. Every fallback writes an `ERROR` or `WARNING`
log line naming the cause — alert on those, or you will ship a museum site whose
"AI search" has quietly been plain vector search for a month.

## Connecting Specify7

The prototype runs on the hand-written list in `app/specimens.py`. Real data
comes from a Specify7 server via `app/specify_sync.py`:

```
Specify7 API  ──sync──▶  data/specimens.json  ──▶  enrich / index / search
```

**Specify7 is contacted only during sync** — never at enrichment, startup, or
search. A visitor typing "T rex" does not hit Specify7; the collection is pulled
to a local snapshot and served from memory. So Specify7's latency and uptime
never touch the search box; a stale sync is the worst a Specify7 outage can do.

```bash
python -m app.specify_sync --dry-run      # fetch + map, print, write nothing
python -m app.specify_sync --limit 20     # first 20 records, to test the mapping
python -m app.specify_sync                 # full sync -> data/specimens.json
python -m app.enrich                       # re-enriches only new/changed records
```

`load_specimens()` in `app/specimens.py` returns the synced file if it exists,
else the demo list — the single seam between prototype and production. Once
`data/specimens.json` exists, everything downstream uses it with no code change.

Two functions in `specify_sync.py` are **stubs you must fill in** — the rest
(pagination, retry, atomic write, change reporting) works as written:

- `_authenticate()` — Specify7 auth varies by deployment (session login vs. API
  key). The docstring walks both.
- `_map_record()` — Specify7 records are relational; `scientificName`, `family`,
  `kingdom`, `locality`, `collector` are all traversals, not flat fields. The
  docstring maps the standard Specify7 path to each. Read one real record first
  (`{BASE_URL}/api/specify/collectionobject/?limit=1`) — exact key names depend
  on your collection's schema config.

**Re-enrichment is now change-aware.** `enrich.py` hashes the fields it sends to
Claude and stores the hash. A record a curator edits in Specify7 keeps its id but
changes content, so the hash differs and it gets re-enriched; untouched records
are skipped for free. Run `app.enrich` after every sync.

**Watch the kingdom vocabulary.** `KINGDOMS` (in `specimens.py`) and the `Kingdom`
`Literal` (in `ai.py`) are a fixed list. Real data may hold kingdoms the demo
doesn't — sync prints the ones it observed and warns on any not in the list. A
kingdom missing from both places can never be produced as a filter, silently
making those specimens unreachable by kingdom. It's the most likely quiet
breakage when you switch off demo data.

## Evaluation

```bash
python -m evals.run                          # opus vs haiku, 33 cases each
python -m evals.run --models claude-haiku-4-5
python -m evals.run --limit 5                # smoke test, 5 cases
python -m evals.run --json evals/out.json
```

Costs ~$0.20 for a full two-model run. Nothing is cached.

Three metrics, in descending order of how much they should worry you:

| metric | meaning |
|---|---|
| **hallucinated** | The parser invented a filter. `"T rex"` → `locality="South Dakota"` silently excludes every T. rex found anywhere else. **A wrong filter is worse than no filter** — no filter degrades to semantic search; a wrong one excludes. |
| **missed** | The parser dropped a filter the query plainly stated. Costs precision; the semantic half usually still finds the specimen. |
| **top-1** | Did the right specimen rank first, end to end. Exercises enrichment and embeddings too — a clean parse with a bad top-1 means the problem is downstream. |

`evals/cases.py` holds 33 cases: 22 `expect` assertions (conservative — only where
a competent parser has no real choice) and 124 `forbid` assertions (aggressive —
hallucination is the failure that matters). Seven cases are pure semantics
(`"fool's gold"`, `"large predator"`) where *every* structured field must stay
null.

The runner exits non-zero on any hallucination, so it can gate CI. It also aborts
after the first `400`, since a 400 is a property of the request or the account,
not of one query.

**Run this before trusting the Opus→Haiku downgrade on the query parser.** That
choice is currently justified by reasoning about task shape, not by evidence.

## Layout

| File | |
|---|---|
| `app/specimens.py` | Demo records, controlled vocabulary, `load_specimens()` |
| `app/specify_sync.py` | Pull real records from Specify7 → `data/specimens.json` |
| `app/ai.py` | Claude client, Pydantic schemas, query parsing, enrichment |
| `app/enrich.py` | Offline enrichment pass (`python -m app.enrich`) |
| `app/embeddings.py` | Encode + cache vectors, cosine similarity |
| `app/search.py` | Hybrid scoring: hard filters, field boosts, similarity |
| `app/main.py` | FastAPI endpoints |
| `evals/cases.py` | 33 visitor queries with expected / forbidden filters |
| `evals/run.py` | Model comparison harness (`python -m evals.run`) |

## Known limits

- **12 specimens is too few to judge ranking quality.** Cosine similarity always
  returns *an* ordering, so the results will always look plausible. Don't tune
  the weights in `app/search.py` against this dataset — you'd be fitting noise.
- The lexical fallback folds a trailing `s` and nothing else. It is not a
  stemmer. Embeddings are the answer to morphology.
- `KINGDOMS` and `CATEGORIES` in `app/specimens.py` are hard-coded. Derive them
  from the live catalog when you swap in real data, or the query parser will
  emit filters that match nothing.
- `sentence-transformers` contacts huggingface.co on startup to validate its
  model cache, even when the weights are already on disk. Fine locally; it will
  fail on an air-gapped host. Pin with `local_files_only=True` before deploying
  anywhere without egress.
- Haiku's parse quality on real visitor queries is **unmeasured**. See below.
