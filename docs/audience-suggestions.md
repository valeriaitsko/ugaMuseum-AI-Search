# Audience-aware "You might also be interested in"

Design spec for adding audience-aware suggestions to the AI search, and the
contract the frontend integration depends on. Strategy: **grounded + validated**
— every suggested query is derived from real collection facets and test-run
against the local index before it's shown, so no chip is ever a dead end.

Nothing here contradicts the existing pipeline; suggestions are computed *after*
a normal search, from data we already have in hand.

---

## 1. Why this lives in the Python backend

After a search, the backend is holding the two things suggestion generation
needs, and the frontend is holding neither:

- **`ai_parsed_query`** — the `SearchFilters` (`app/ai.py:52`): `kingdom`,
  `category`, `family`, `genus`, `species`, `locality`, `semantic_query`.
- **the collection index** — every specimen and its facets (`app/search.py`).

So `POST /api/ai-search` gains an `audience` input and a `suggestions` output.
The frontend just renders chips and re-runs search when one is clicked.

---

## 2. Request / response contract

### Request

```jsonc
POST /api/ai-search
{
  "query": "flowering plants in Georgia",
  "audience": "student"   // "student" | "researcher", defaults to "student"
}
```

`audience` defaults to `"student"` to match the frontend's own default
(`useAudienceStore` — `fMuseum/src/store/useAudienceStore.ts:14`).

### Response (additions only)

```jsonc
{
  "original_query": "...",
  "ai_parsed_query": { ... },
  "results": [ ... ],
  "degraded": false,
  "degraded_reason": null,

  "suggestions": [
    {
      "label": "Oaks",              // chip text, short (≤ ~24 chars)
      "query": "Quercus",           // what re-runs the search when clicked
      "kind": "narrow",             // "broaden" | "narrow" | "pivot"
      "reason": "more specific"     // optional, for a tooltip
    }
  ]
}
```

`suggestions` is **always present** (empty array if nothing survives
validation). Capped at **4**. Order: most relevant first.

The contract is identical for both audiences — only the *contents* differ. The
frontend renders the same chip row either way and does not branch on audience.

---

## 3. What "student" vs "researcher" means

Both audiences see the row. The difference is entirely in how candidates are
generated from the parsed filters.

### Student → broaden (exploratory)

Ignore `locality` / collector. Goal: open new doors.

1. **Sibling groups** — categories/high-level groups present *elsewhere* in the
   collection that aren't in the current results. (1–2 suggestions)
2. **One zoom-in** — the dominant family/genus in the current results, offered
   as a narrower query. (1 suggestion)

> ⚠️ **Data reality:** the live production snapshot is a **herbarium —
> `Plantae` only** (`app/specimens.py:139`, `app/ai.py:48`). There are no
> mammals or dinosaurs in the real collection, so cross-kingdom "broaden"
> suggestions only appear on the multi-kingdom *demo* list. On real data,
> "broaden" for a student must mean breadth *within* plants — other families,
> growth habits (`"trees"`, `"wildflowers"`, `"grasses"`), regions. The
> validation step (§4) enforces this automatically: any suggestion with no
> backing specimens is dropped, so the feature degrades gracefully to
> whatever the collection actually holds.

### Researcher → pivot (precision)

Hold the structured filters the query used and vary **one axis** — this is the
"take dates and localities into account" behavior. Fire a pivot only on axes the
query actually constrained; otherwise fall back to the strongest facet in the
results.

- `locality` set → **"same taxon, other localities"** (hold family/genus, drop
  locality) and **"same locality, other taxa"** (hold locality, drop taxon).
- `family`/`genus`/`species` set → **adjacent taxa** present in the collection
  (sibling genera in the same family; sibling species in the same genus).
- date / collector set → pivot there (**requires a schema addition — see §6**).

---

## 4. Algorithm

### Precompute at startup (in `SearchIndex.__init__`)

Facet counts over the whole collection — cheap, done once:

```python
self.facets = {
    "category": Counter(...),   # via enrichment category
    "family":   Counter(...),
    "genus":    Counter(...),
    "locality": Counter(...),
}
```

### Per request, after `index.search(filters)` returns `results`

```
1. Determine the "current facet" — the dominant family/genus/category/locality
   of the top results, and whatever `filters` explicitly set.

2. Build candidate suggestions per the audience rules in §3. Each candidate is
   a *constructed* SearchFilters object (we already know e.g. family="Quercus"),
   NOT a raw string that needs re-parsing.

3. VALIDATE (the grounded step): run index.search(candidate_filters) locally.
   Drop the candidate if it returns 0 results, or if its result set is
   effectively identical to what's already on screen. No Claude call — the
   filters are already structured, so this is pure local search. Zero API cost.

4. Label each survivor (§5), dedupe by label and by result-set, cap at 4.
```

The reason this is cheap: facet-derived candidates arrive **already
structured**, so validation is a local `search()` call, not another
Claude round-trip. Suggestion generation adds **no API cost** to a search.

### ⚠️ Validate structurally — do NOT re-encode

The only expensive operation in `search()` is `embeddings.similarities()`
(`app/embeddings.py:80`), which runs a MiniLM forward pass (~10–40ms on CPU) to
turn a fuzzy query into a vector. Everything else — hard filters, field scoring,
the cosine dot product against the pre-built matrix — is microsecond-scale.

Suggestion candidates are **not fuzzy** — they're exact facets you constructed
(`family="Quercus"`), so they don't need the embedding half at all. Validate a
candidate with `_passes_hard_filters` + a field-match check (or a `search()`
variant that skips `similarities()`), **not** a full semantic search.

- Naive (full `search()` per candidate): 6–8 candidates × one encode each ≈
  **+100–300ms**. Avoid this.
- Structural (no encode): tens of thousands of dict comparisons ≈
  **sub-millisecond**.

The parent search already pays for exactly one encode (the user's real query).
Done right, validation adds **zero** additional encodes — the feature is
effectively free on top of the search already running.

---

## 5. Labels (no Claude needed)

Derive human-friendly chip text from data already loaded:

- family/genus → reuse `enriched.json` `common_names` when present
  (`Quercus` → "Oaks", `Fagaceae` → "Beeches & oaks"); fall back to the
  scientific token otherwise.
- category → the plain word ("Minerals", "Fossils").
- locality pivot → the place name ("Also from: Montana").

Enrichment is already retrieval-only in this codebase; using it for labels keeps
that boundary — labels are navigation, never presented as catalog fact.

---

## 6. Prerequisite for full researcher support (schema gap)

The user's researcher scenario names **dates and localities**. Locality is
already parsed. **Dates and collector are NOT** in `SearchFilters` today
(`app/ai.py:52`). To pivot on them:

- Add `collector: Optional[str]` and a date field (`year: Optional[int]` or a
  `date_from`/`date_to` range) to `SearchFilters`, and mention them in
  `QUERY_SYSTEM` so the parser extracts them.
- `collector` already exists on specimen records; a date field will arrive with
  the real Specify7 sync (the demo records have none).

Until then, researcher pivots cover locality + taxonomy, which is most of the
value. Date/collector pivots are a clean follow-up.

---

## 7. Frontend integration notes (for the collaborator)

- The AI search is a **separate FastAPI service**, not the frontend's Node
  server. Two options:
  - **Vite proxy (recommended):** proxy e.g. `/ai` → the FastAPI origin in
    `vite.config.ts`, so the browser calls it same-origin like `/api`.
  - **Direct call:** CORS is already open for `:5173`/`:3000`
    (`app/main.py:15`), so the browser can hit it directly if preferred.
- Add one client method (`fMuseum/src/api/client.ts`) that POSTs
  `{ query, audience: useAudienceStore.getState().audience }`. The audience
  plumbing already exists there for `/specimens`, `/chat`, `/quiz`.
- Render `suggestions` as a chip row under the results. Clicking a chip sets the
  search box to `suggestion.query` and re-runs search.
- **Audience toggle already re-fires search** (`useSpecimenSearch` re-runs on
  audience change — `fMuseum/src/hooks/useSpecimenSearch.ts:82`), so flipping
  Student↔Researcher refreshes the suggestions for free.

---

## 8. Build order

1. Add `audience` to `SearchRequest`; thread it into a new
   `suggest(filters, results, audience)` on `SearchIndex`.
2. Precompute facet counters at startup.
3. Implement student (broaden + zoom) and researcher (pivot) candidate builders.
4. Add the local validation pass; cap + dedupe.
5. Add labels from enrichment.
6. (Follow-up) extend `SearchFilters` with `collector` + date for researcher
   date/locality pivots.
