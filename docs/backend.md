# Backend Technical Documentation

Museum AI Search — a natural-language search engine over a natural history museum
collection. A visitor types `"big meat-eating dinosaur"` or `"beetles from
Georgia"` and gets the right specimens back, ranked.

This document walks the backend end to end: the data pipeline that produces the
snapshot the app serves, the offline enrichment that makes records findable, and
the per-request search path that answers a query. It is written to be read
top-to-bottom by someone who has never seen the code.

- **Stack:** Python 3, FastAPI, Pydantic, the Anthropic SDK (Claude),
  `sentence-transformers` (MiniLM embeddings), NumPy, PyMySQL.
- **Scale:** ~31,840 specimens in the current snapshot — a UGA insect collection,
  a herbarium slice, and a small archaeology collection (projectile points) that
  carries its own taxonomy (see §2).
- **Entry point:** `app/main.py` (the HTTP API). Everything else is a module it
  composes.

---

## 1. The big picture

There are **three stages**, and they run at three different times:

```
                (build time)          (offline, once per specimen)     (startup)
  Specify 7 DB ──▶ specimens.json ──▶ enriched.json ──────────────▶ vector index
                   [build_snapshot]    [enrich / enrich_batch]        [embeddings]
                                                                          │
                                                                          ▼
  visitor query ───────────────────────────────────────────▶  /api/ai-search
                                                               [main → ai/search/
                                                                orders/suggest]
```

The single most important design idea: **the expensive, AI-heavy work happens
offline and is cached; the per-request path is cheap and mostly local.**
Enrichment cost scales with the *size of the collection* (paid once), not with
*traffic*. Every search afterward is one small Claude call plus a local vector
lookup.

The second important idea: **degradation, not failure.** If Claude is
unreachable, search still works (semantic + lexical over local data). If
embeddings aren't installed, search still works (lexical fallback). No single
dependency can take the whole thing down.

---

## 2. Stage 0 — Getting the data (the snapshot)

The app **never talks to the live Specify 7 database at request time.** It reads
a static JSON snapshot, `app/data/specimens.json`. This is what keeps a slow or
down Specify server from ever touching search — a stale server only makes a
stale snapshot.

There are two ways to produce that snapshot, and they are two different seams for
two different situations:

### `app/build_snapshot.py` — read the Specify DB directly

Used in the current setup. It connects straight to the MySQL/MariaDB that backs
Specify 7 and flattens each `CollectionObject` into a lean record.

**Step by step:**

1. `load_taxa()` reads the **entire taxon tree** into memory once (it's a small
   table, and every specimen needs to walk it).
2. `fetch_rows()` runs one SQL query joining `collectionobject` →
   `determination` (only `IsCurrent = 1`) → `collectingevent` → `locality`. It
   returns identifiers, the current determination's `TaxonID`, and a locality
   name.
3. `ranks_for()` walks each specimen's taxon **up the tree by `ParentID`**,
   collecting `{rank name: name}`. Taxonomy is read **by rank NAME, not RankID** —
   because this database holds **two taxon trees that reuse the same RankIDs**: a
   biology tree and an archaeology tree where rank 10 means "Points" (not
   "Kingdom") and rank 30 means "Specimen" (not "Phylum"). Reading by name (joined
   from `taxontreedefitem`) keeps them straight; reading by id would slot a stone
   point's typology into the `kingdom` field. The RankID-0 root ("Life"/"Root") is
   still dropped, and a `seen` set guards against a malformed parent cycle.
4. `flatten()` emits a unified record (with `null` where absent), tagged with the
   `discipline` its taxonomy came from so downstream can label it correctly:
   - **always:** `{id, catalogNumber, altCatalogNumber, discipline, locality}`
   - **biology** (`discipline: "biology"`): `kingdom, phylum, class, order, family,
     genus, species`
   - **archaeology** (`discipline: "archaeology"`): `culturalPeriod, points,
     specimen` — e.g. *Late Paleo Indian Period → Incurvate Base Points → Dalton*

   Every record carries all fields; only its discipline's set is filled. `phylum`
   and `class` are display-only — excluded from the search embedding, so adding
   them forced no re-encode (see §4).
5. `_write_atomic()` writes to a temp file then renames, so a crash mid-write
   can't corrupt the snapshot the running server reads.

Records with **no current determination** get null taxonomy — we deliberately do
*not* fall back to a superseded identification. `_report()` prints how many that
was, a breakdown by **discipline** (biology / archaeology), and any observed
kingdom the query parser isn't configured to emit (see §6, the `KINGDOMS` sync
requirement).

```bash
python -m app.build_snapshot --dry-run   # build + print first record, write nothing
python -m app.build_snapshot --limit 5   # first 5, to eyeball the mapping
python -m app.build_snapshot             # full build
```

> **Connection note (non-obvious):** the live Specify runs natively in WSL, and a
> Windows-native MySQL squats on `0.0.0.0:3306`, so the build must be **run from
> inside WSL** to reach the real MariaDB. DB name is `SpecifyDB`. Config comes
> from this app's own `.env` (`SPECIFY_DB_*`).

### `app/specify_sync.py` — pull from the Specify 7 REST API

An alternative source for a hosted Specify instance where you don't have direct
DB access. Most of it works as written (config, pagination, retry/backoff,
atomic write, change reporting). **Two functions are seams you fill in against
your own instance:** `_authenticate()` and `_map_record()` — because Specify's
REST records are deeply normalized and the exact JSON keys depend on your
collection's schema config. The traversal is mapped out in comments.

Either way, the output is the same `specimens.json` shape, so **nothing
downstream knows or cares which source produced it.**

### `app/specimens.py` — the one seam between prototype and production

`load_specimens()` returns the synced file if it exists, else a 12-record
hand-written `DEMO_SPECIMENS` list. That one function is the entire boundary:
enrichment, embeddings, and search all call it, so swapping in real data is one
build/sync run and **zero code changes.**

This module also owns two controlled vocabularies:

- `KINGDOMS` — the exact kingdom strings the catalog uses (currently
  `["Plantae", "Animalia"]`). The query parser is constrained to these so it
  can't invent `"Mineralia"` for a record stored as `"Mineral"`.
- `CATEGORIES` — visitor-facing groupings that **aren't** kingdoms:
  `["plant", "animal", "insect", "mineral", "fossil"]`. Visitors search for
  "fossils" and "bugs"; those aren't taxonomic ranks, so they're assigned during
  enrichment.

---

## 3. Stage 1 — Enrichment (offline AI, once per specimen)

**The problem:** a catalog row says `Canis lupus`. A visitor types `"large
predator"` or `"wolf skeleton"`. Nothing in the raw row connects those. This is
the gap keyword search can never close.

**The fix:** before serving, Claude reads each record and writes back the words a
visitor would actually type — common names, abbreviations, broader search
aliases (body plan, diet, habitat) — plus a visitor-facing `category`. This is
cached in `app/data/enriched.json`, keyed by specimen id.

### The critical provenance rule

**Everything enrichment produces is model-generated and is retrieval-only.** It
exists to make records *findable* and is **never shown to a visitor as though the
museum asserted it.** The search API only ever returns the museum's own original
catalog fields. (See §5 — the enriched text is used to *rank*, then discarded
from the response.)

### What Claude returns — the `Enrichment` schema (`app/ai.py`)

```
common_names     everyday names           ("gray wolf")
abbreviations    shorthand a visitor types ("T rex")
search_aliases   broader words: body plan, diet, habitat, grouping ("predator")
category         one of the 5 CATEGORIES
search_text      a short, concept-dense phrase of what the specimen IS
```

`search_text` is the field that gets embedded, and its shape matters enormously
(see §4). It must be short — ~10-20 words — naming what the specimen *is*
(scientific name, common names, key descriptors), with **no locality or
collector** and no full sentences.

### The prompt is deliberately tiny — and that's an engineering decision

`ENRICH_SYSTEM` is terse on purpose. It's fixed input re-sent on **every one of
~31,800 calls**, and it sits below Opus's 4,096-token minimum cacheable prefix,
so it *can't* be prompt-cached — shrinking it is the only lever. A 50-record A/B
test against a much longer, example-heavy version showed **no quality loss**
(the extra prose was steering a human reader, not the model) while cutting total
input tokens **~42%** (~$56 off the full batched run). Even the Pydantic field
descriptions are kept short for the same reason, and the class docstring is a
plain comment — because Pydantic sends a class docstring to the model as the
schema description, which would bill on every call.

### Change detection — never re-pay for unchanged records

`content_hash()` fingerprints only the fields that *affect* the output
(`kingdom, order, family, genus, species, locality`). Enrichment is keyed by id,
but a curator can edit a record while keeping its id. Keying re-enrichment on
this hash means an **edited** record gets re-enriched while an **untouched** one
is skipped for free. Identifiers like `catalogNumber` are deliberately excluded
from the hash — changing them shouldn't trigger a re-enrich.

### Safety refusals and the fallback model

Opus's safety classifier is stricter on bio-adjacent names — crop pests like
*Spodoptera frugiperda* (fall armyworm) — than Sonnet, which enriches them fine.
So on a **refusal** (`stop_reason == "refusal"`), the pipeline retries once with
`FALLBACK_MODEL = "claude-sonnet-5"`. Only if **both** models refuse is a
specimen skipped — and it stays findable by its Latin name, just un-enriched.
This distinction (a refusal vs. a transient API error) is handled explicitly:
transient errors (429/5xx/network) skip and get retried next run; a 400 stops
the whole run because it applies to every record identically.

### Two ways to run it

**Synchronous (`app/enrich.py`)** — good for samples and quality checks. Writes
after every specimen, so an interrupted run resumes without re-paying.

```bash
python -m app.enrich --limit 200     # random sample (NOT first-N; ids are grouped by upload)
python -m app.enrich                 # enrich everything new/changed
```

**Batches API (`app/enrich_batch.py`)** — for the one-time full ~31k run. **Half
the price**, and it sidesteps burst-refusal fragility because Anthropic paces the
batch server-side. It's a small resumable state machine persisted to disk:

```bash
python -m app.enrich_batch submit    # send the job, save its id, exit
python -m app.enrich_batch collect   # re-run until it says "Done"
```

`collect` advances through phases: collect primary → submit a *second* batch for
the refused specimens on the fallback model → collect that. The batch id lives in
a state file, so a closed terminal or dropped connection loses nothing. Results
come back keyed by `custom_id` (the specimen id), so out-of-order results map
straight back.

> **Current status:** the full batched run is complete — `enriched.json` holds
> **31,794 of 31,798** records. The remaining 4 (notorious crop pests like the
> fall armyworm) were declined by both Opus and Sonnet and stay findable by their
> Latin names, just un-enriched. Refusal rate: 0.29%.

---

## 4. Stage 2 — Embeddings (semantic index, at startup)

`app/embeddings.py` turns the enriched `search_text` of every specimen into a
vector, so a query can be matched by **meaning** rather than shared words.

- **Model:** `all-MiniLM-L6-v2` (local, via `sentence-transformers`).
- **Vectors are unit-normalized**, so cosine similarity is a plain dot product
  (`index @ query_vector`).
- **Cached to disk** (`app/data/vectors.npz`) with a SHA-256 **fingerprint** of
  all the texts + model name. Change any enriched text, or reorder them, and the
  fingerprint changes and forces a re-encode; otherwise startup reuses the cache.

### Why `search_text` must stay short (the measured reason)

MiniLM **mean-pools** its token vectors, so every off-concept word drags the
vector away from what the specimen *is*. Measured on this collection:

| query | outcome |
|---|---|
| 34-word prose paragraph about T. rex | scores **0.395** on "big meat-eating dinosaur" — *loses* to an unenriched Apatosaurus (a herbivore) |
| 10-word alias phrase, same facts | scores **0.583** |

That's why locality and collector are kept out of the embedded text — they're
already matched exactly by structured filters, so embedding them only costs
ranking quality. And it's why `_search_text()` in `search.py` also excludes
`order`, `phylum`, `class`, and the internal `discipline` flag
(`_NON_EMBEDDED_FIELDS`): they're Latin rank words no visitor types into the
semantic box, and keeping them out leaves the embedded text byte-identical to
before those fields existed, so adding them forces no re-encode. (The archaeology
ranks — `culturalPeriod`/`points`/`specimen` — are *not* excluded: "Dalton" and
"Woodland Period" are exactly what a visitor would search an artifact by.)

### Graceful degradation

`available()` checks whether `sentence-transformers` is even installed (it pulls
in ~2.5 GB of torch). If it's absent, `self.semantic` is `False` and search falls
back to **lexical matching over the enriched text** instead of crashing.

---

## 5. Stage 3 — Search (per request)

This is the hot path. `POST /api/ai-search` with `{query, audience}`. Here's what
happens, in order.

### In plain English — one search, start to finish

Imagine a visitor types **"beetles from Georgia."** Here's the whole journey,
without the jargon:

1. **Two readers look at the query at the same time.**
   - A plain **dictionary** (no AI) spots the word "beetles" and knows that means
     the *beetle order*, Coleoptera. Cheap, instant, and works even if the AI is
     down.
   - A small, fast **AI model** reads the sentence and fills in a little form:
     *this is an insect, collected in Georgia, and in plain words they want "beetles
     from Georgia."* It only fills a box when the query truly supports it — a wrong
     guess would hide good results.

2. **The collection is narrowed to the sure things.** Anything that definitely
   doesn't fit is thrown out before scoring even starts — ask for beetles and every
   non-beetle is gone. (These strict "hard" filters are kingdom, category, and
   order.)

3. **What's left gets scored.** Each surviving beetle earns points for (a) matching
   a specific thing you named — its family, genus, species, or locality — and (b)
   how close its *description* is in meaning to what you asked. That second part is
   what can match "beetles" to a record that never uses that exact word. Being
   specific (a named family *and* place) is tuned to beat a vague meaning-match.

4. **The top 10 come back,** each tagged with *why* it ranked — e.g. "matched on
   order, locality~Georgia, similarity 0.48" — so the ranking is never a black box.
   Only the museum's own catalog facts go out; the AI-written search words are used
   to rank, then dropped.

5. **Alongside the results** the visitor also gets a one-tap plain-English
   **summary**, a few **related-search chips**, and — importantly — an **honest note
   if part of their query matched nothing.**

**When you ask for something that isn't there.** Say you search **"plants from
Georgia,"** but the herbarium is entirely Californian — there are *no* Georgia
plants. Rather than quietly hand back California plants as if they were Georgian,
the search does two things:

- **It tells you:** *"No results are actually from Georgia — showing related results
  from elsewhere."*
- **It cleans up the ranking:** it drops the dead word "Georgia" from the
  meaning-match (that word only adds noise, because a specimen's location isn't part
  of its "what it is" description), so you get a representative spread of the plants
  that *do* exist instead of a lopsided clump of one genus.

The detailed, mechanical version of each step follows.

### 5.0 The index is built once, at import

`main.py` constructs a single `SearchIndex()` at module load. Its constructor
loads specimens, loads enrichments, builds the search texts, embeds them (or
loads the cached vectors), and precomputes collection-wide **facet counts**
(family/genus/species/locality/category → counts) that the suggestion engine
reads. Doing any of this per-request would re-encode the whole collection on
every search.

### 5.1 Parse the query into structured filters (`app/ai.py`)

Claude (Haiku, see §6) reads the natural-language query and fills the
`SearchFilters` schema:

```
kingdom      only if named/unambiguously implied
category     visitor-facing grouping (fossil wins over animal for extinct things)
family/genus/species    taxonomic fields, when the query supports them
locality     place collected
semantic_query   the intent in plain words ("big scary dinosaur" → "large predatory dinosaur")
explanation      one sentence on how the query was read
```

Because every field is **schema-enforced**, `kingdom` can never come back as a
list or an invented value — which is what lets the search code call `.lower()`
with no defensive coercion. The parser is told (in `QUERY_SYSTEM`) to **leave a
field null unless the query genuinely supports it**, because a wrong filter
*silently hides* correct results, which is worse than no filter. If Claude
returns `stop_reason == "refusal"`, that's raised as a `RuntimeError`. If
`semantic_query` comes back empty, it's backfilled with the raw query (an empty
one would silently disable the entire semantic half).

### 5.2 Degrade instead of erroring (`_parse_or_degrade` in `main.py`)

**No Claude failure ever returns an error to the visitor.** Structured filters
are an enhancement; the index, embeddings, and scoring are all local and keep
working without them. `_parse_or_degrade()` catches each failure mode
specifically — ordered most-specific first, because `AuthenticationError`,
`BadRequestError`, and `RateLimitError` all subclass `APIStatusError`:

| exception | `degraded_reason` | note |
|---|---|---|
| `AuthenticationError` | `auth_error` | bad/missing key — logged at **error** |
| `BadRequestError` | `bad_request` | bad schema or unbillable account |
| `RateLimitError` | `rate_limited` | transient (warning) |
| `APIConnectionError` | `unreachable` | transient (warning) |
| `APIStatusError` | `api_error` | other 4xx/5xx |
| `RuntimeError` | `refused` | Claude declined the query |

On any degrade, `fallback_filters()` returns filters with everything null except
`semantic_query = query`, and the response carries `degraded: true` so the
frontend can tell the visitor *why* their filters didn't apply. Operators find
out through the error log and the flag — **so alert on that log line**, since an
invalid key otherwise degrades silently forever.

### 5.3 Detect taxonomic orders — deterministically, no AI (`app/orders.py`)

Insect records carry a Latin order (`Coleoptera`, `Diptera`), but visitors type
English ("beetles", "flies"). Orders are a **closed, tiny vocabulary** (~23 in
this collection, ever), so this is a **hand-maintained dictionary, not a model
call.** `detect_orders()` does whole-word matching (so "fly" doesn't match
"butterfly") of common names against the *raw query*.

Because it's a plain lookup, order-level common-name search **works with no API
credits and even when Claude is down** — it runs independently of the parser.
`"beetles and flies"` returns both orders so the caller can OR them. (Species
common names, by contrast, are tens of thousands and open-ended — those need
enrichment; orders are small enough to hand-table.)

### 5.4 Score and rank (`SearchIndex.search` in `app/search.py`)

This is a **hybrid** of hard filters, structured-field boosts, and semantic
similarity.

**a. Hard filters exclude (`_passes_hard_filters`).** `kingdom`, `category`, and
detected `orders` *remove* non-matches rather than down-ranking them — asking for
minerals and getting a wolf at rank 4 is worse than getting fewer results.
Un-enriched specimens have no enrichment `category`, so one is **derived from their
taxonomy** (`_fallback_category`: an `Insecta` record → `insect`, `Plantae` →
`plant`). Without that, a pest species both models refused to enrich had `category
= None`, slipped past a `category=plant` filter, and — being from Georgia — falsely
satisfied the locality filter, polluting a plants search. Free-text fields stay
soft.

**b. Field scores boost (`_field_score`).** Weighted matches on the remaining
candidates:

```
family / genus / species   +3.0 each   (exact, case-insensitive equality)
locality                   +2.0        (substring — a visitor's phrasing rarely equals the catalog's)
semantic similarity        × 5.0       (cosine, capped at the 2-field-match level)
```

The weights are tuned so that **naming a family + a locality outranks a fuzzy
conceptual hit** — someone specific should beat someone vague.

**c. Semantic or lexical.** If embeddings are available, cosine similarity of
`semantic_query` against every candidate vector is added (`× 5.0`). If not, a
`_lexical_score()` fallback does whole-word matching (tolerant of a trailing
plural "s") at a deliberately low weight (`0.75`/word) — lexical is the fallback,
not the plan.

**d. Honest about filters that matched nothing.** A soft filter only *boosts*, never
excludes — so if the query named a `family`/`genus`/`species`/`locality` that **no
candidate matches**, the search would otherwise return unrelated records as if they
fit. `search()` detects this and returns an **`unmatched_filters`** list, so the UI
can say *"nothing is from Georgia — showing related results"* instead of implying a
match. And because **locality is absent from the embeddings** (§4), an unmatched
locality is *stripped from `semantic_query` before ranking* (`_strip_locality`):
leaving it in only drags the query vector toward noise — `"plants collected in
Georgia"` once clustered every result on a single genus, while `"plants"` ranks a
representative spread. Taxonomic terms are *kept* even when unmatched, since a genus
name like "Papilio" still helps find relatives semantically.

**e. Filter, sort, cap.** Anything scoring ≤ 0 is dropped; the rest sort by score
descending; the top 10 are returned.

**f. Only the museum's own fields go out.** Each result is
`{**specimen, score, matched_on}` — the original catalog record plus a score and
a `matched_on` trace (e.g. `["genus=Papilio", "similarity=0.42"]`). **The
model-generated enrichment is never in the response** — returning a generated
common name here would let it surface in the UI as if the catalog asserted it.
This is the provenance rule from §3, enforced at the boundary.

### 5.5 Build "You might also be interested in" suggestions (`app/suggest.py`)

After the results are known, `build_suggestions()` produces up to 4 suggestion
chips — **audience-aware**, and computed entirely from **local facets, with no
extra Claude call:**

- **student → broaden.** Sibling groups elsewhere in the collection, plus one
  "zoom in" facet within the current results. Exploratory.
- **researcher → pivot.** Hold the filters the query used and vary one axis
  (locality, adjacent taxa). Precise.

Every candidate carries a **predicate**, and `_validate_and_pack()` runs it
against the collection to guarantee the chip returns **at least one specimen not
already on screen** before it's shown — so **no chip is ever a dead end.**
Crucially, validation is a **local predicate scan, not another embedding pass** —
candidates arrive as structured facets, so this adds zero encodes. It also works
in degraded mode, since it reads results and facets, never Claude. (Full design:
`docs/audience-suggestions.md`.)

### 5.6 The response

```jsonc
{
  "original_query": "...",
  "ai_parsed_query": { /* the SearchFilters */ },
  "detected_orders": ["Coleoptera"],
  "results": [ /* museum fields + score + matched_on */ ],
  "suggestions": [ /* up to 4 grounded chips */ ],
  "unmatched_filters": [ /* soft filters the query named that matched nothing, */
                        /* e.g. {"field":"locality","value":"Georgia"} */ ],
  "degraded": false,
  "degraded_reason": null,
  "cost": { "model": "...", "input_tokens": N, "output_tokens": N, "cost_usd": 0.00... }
}
```

`cost` is the per-search list-price cost of the one query-parse call (see §6),
logged one line per billed search. It's `null` on a degrade — the parse failed
before returning usage, which is **not** the same as $0.

---

## 6. Two models, on purpose

`app/ai.py` sets `ENRICH_MODEL = "claude-opus-4-8"` and
`QUERY_MODEL = "claude-haiku-4-5"`. This split is deliberate — the two jobs have
**opposite cost curves and opposite blast radii:**

| | **Enrichment (Opus)** | **Query parsing (Haiku)** |
|---|---|---|
| how often | once per specimen, ever | every search, forever |
| dominates | one-time build cost | lifetime cost |
| a mistake… | is frozen into `enriched.json`, silent, unreviewable, compounds at scale | affects one search; the visitor retypes and it's gone |
| task size | write good aliases (judgment) | read 10 words, fill 8 constrained fields (small) |
| verdict | **don't economize** | cheap *and* fast — and fast in front of a search box is a feature |

`usage_cost()` computes a per-call list-price record (with cache-read/write
accounting that stays correct if caching is added later). `MODEL_PRICES` is a
list-price estimate, not a bill — the console is the record of truth.

---

## 7. Evaluation harness (`evals/`)

`python -m evals.run` compares query-parser models on the same real queries
(`evals/cases.py`, grounded in actual families/genera/localities in the
snapshot). It measures three things, in descending order of how much they should
worry you:

1. **hallucinated** — a filter invented from nothing (`"T rex"` →
   `locality="South Dakota"`). The sharp instrument: a wrong filter *excludes*,
   while a missing one degrades to semantic search. **A hallucination fails CI**
   (non-zero exit).
2. **missed** — a filter the query plainly stated and the parser dropped. Costs
   precision; the semantic half usually still finds it.
3. **top1** — did the right specimen rank first, end to end (exercises
   enrichment + embeddings too, not just the parser).

Latest run (`evals/last-run.json`): both Opus and Haiku scored a clean sweep —
20/20 extracted, 0 hallucinations, 3/3 top-1 — with **Haiku ~6× cheaper**
($0.030 vs $0.187) and ~4.5× faster. That's the evidence behind choosing Haiku
for the per-request path.

---

## 8. Where to change things (quick map)

| You want to… | Edit |
|---|---|
| Change how a query is parsed | `QUERY_SYSTEM` / `SearchFilters` in `app/ai.py` |
| Change what aliases enrichment writes | `ENRICH_SYSTEM` / `Enrichment` in `app/ai.py` |
| Add a common name for an insect order | `ORDER_COMMON_NAMES` in `app/orders.py` |
| Retune ranking | `FIELD_WEIGHTS` / `SEMANTIC_WEIGHT` in `app/search.py` |
| Change suggestion behavior | `app/suggest.py` (see `docs/audience-suggestions.md`) |
| Add a new kingdom value | `KINGDOMS` in `specimens.py` **and** `Kingdom = Literal[...]` in `ai.py` — both, or the parser can't emit it |
| Point at real Specify data | run `build_snapshot` (direct DB) or fill the seams in `specify_sync.py` (REST) |
| Swap the embedding model | `MODEL_NAME` in `app/embeddings.py` (invalidates the vector cache) |

## 9. Operational runbook

Full rebuild from scratch:

```bash
python -m app.build_snapshot          # 1. DB → specimens.json  (run from WSL)
python -m app.enrich_batch submit     # 2. enrich (half price); then:
python -m app.enrich_batch collect    #    re-run until "Done"
uvicorn app.main:app --reload         # 3. serve (embeds at startup, cached after)
```

After editing records, just re-run `build_snapshot` then `enrich` — change
detection re-pays only for what actually changed, and the vector cache
re-encodes only the texts that moved.
