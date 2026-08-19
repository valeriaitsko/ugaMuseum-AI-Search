import logging
from contextlib import asynccontextmanager
from typing import Literal

import anthropic
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.ai import (
    QUERY_MODEL,
    SearchFilters,
    fallback_filters,
    parse_search_query_with_usage,
    usage_cost,
)
from app.orders import detect_orders
from app.search import SearchIndex
from app.suggest import build_suggestions
from app.summarize import summarize_results

log = logging.getLogger(__name__)

# Built once at import: loads enrichments, encodes vectors (cached on disk).
# Doing this per request would re-encode the whole collection every search.
index = SearchIndex()


def _warm_query_encoder() -> None:
    """Load the query-encoder model at boot instead of on the first search.

    SearchIndex() above loads the *vector cache*, but on a cache hit it never
    calls the encoder -- so the MiniLM model that embeds the visitor's query text
    stays unloaded until the first real search, whose one-time ~minute cold load
    overruns the frontend proxy's request timeout and surfaces as "backend timed
    out". Encoding one throwaway query here pays that cost at startup, where
    nothing is on a clock, so the first visitor search returns in seconds.

    Best-effort: a failure here just restores the old lazy behaviour (a slow
    first search), so it must never stop the server from booting.
    """
    if not (index.semantic and index.vectors is not None):
        return
    try:
        from app import embeddings

        embeddings.similarities("warm up the query encoder", index.vectors)
        log.info("[startup] query encoder warmed")
    except Exception:
        log.warning("[startup] query-encoder warm-up failed; first search will be slow",
                    exc_info=True)


@asynccontextmanager
async def lifespan(_: FastAPI):
    _warm_query_encoder()
    yield


app = FastAPI(title="Museum AI Search", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class SearchRequest(BaseModel):
    query: str
    # Drives the "You might also be interested in" suggestions: students get
    # broader sibling groups, researchers get precise pivots on the filters they
    # used. Defaults to "student" to match the frontend's own default.
    audience: Literal["student", "researcher"] = "student"


class SummaryRequest(BaseModel):
    # The specimen ids currently shown -- the summary is grounded in exactly these,
    # looked up in the in-memory index (no re-search, no re-parse). Sending ids
    # rather than whole records keeps the payload tiny and the stats authoritative.
    ids: list[int]
    audience: Literal["student", "researcher"] = "student"
    # The original query, for phrasing context only ("your search for ..."). Never
    # a source of facts -- those come from the specimens the ids resolve to.
    query: str = ""


@app.get("/")
def root():
    return {
        "message": "Museum AI backend is running",
        "specimens": len(index.specimens),
        "enriched": index.enriched,
        "semantic_search": index.semantic,
    }


@app.post("/api/ai-search")
def ai_search(req: SearchRequest):
    query = req.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    filters, degraded_reason, cost = _parse_or_degrade(query)

    # One line per billed search -- the running-cost record now that the Claude
    # call lives here, not in the Node server. `cost` is None on a degraded search
    # (the parse call failed before we got usage), which is not the same as $0.
    if cost:
        log.info(
            "[ai-search] %s %d in / %d out = $%.6f -- %r",
            cost["model"], cost["input_tokens"], cost["output_tokens"], cost["cost_usd"], query,
        )

    # Detected from the raw query by a local dictionary, not Claude -- so "beetles"
    # narrows to Coleoptera even in degraded mode / with no credits. Runs on the
    # original query, not filters.semantic_query, which Claude may have reworded.
    orders = detect_orders(query)

    results = index.search(filters, orders=orders)

    # Grounded, validated, and computed from local facets -- no extra Claude call,
    # and it still works when the parse degraded (it reads results, not filters).
    suggestions = build_suggestions(index, filters, results, req.audience, query)

    return {
        "original_query": query,
        "ai_parsed_query": filters.model_dump(),
        "detected_orders": orders,
        "results": results,
        "suggestions": suggestions,
        # True when Claude was unreachable and the filters are empty. The frontend
        # should say so -- a visitor who searched "collected by Emily Davis" and
        # silently got unfiltered results deserves to know why.
        "degraded": degraded_reason is not None,
        "degraded_reason": degraded_reason,
        # Per-search list-price cost of the query-parse call, or null when no call
        # was billed (degraded). The Node proxy passes this through to the UI.
        "cost": cost,
    }


@app.post("/api/ai-summary")
def ai_summary(req: SummaryRequest):
    """On-demand, audience-aware summary of the results the visitor is looking at.

    Grounded in the specimens the ids resolve to (looked up locally); Claude only
    phrases the locally-computed stats. Unknown ids are dropped rather than erroring
    -- a stale id shouldn't sink the whole summary.
    """
    specimens = [
        index.specimens[index.position[i]] for i in req.ids if i in index.position
    ]
    if not specimens:
        raise HTTPException(status_code=400, detail="No known specimens to summarize.")

    result = summarize_results(index, specimens, req.audience, req.query.strip())

    cost = result.get("cost")
    if cost:
        log.info(
            "[ai-summary] %s %d in / %d out = $%.6f -- %d specimens, %s",
            cost["model"], cost["input_tokens"], cost["output_tokens"],
            cost["cost_usd"], len(specimens), req.audience,
        )
    return result


def _parse_or_degrade(query: str) -> tuple[SearchFilters, str | None, dict | None]:
    """Parse the query with Claude, or fall back to semantic-only search.

    Returns (filters, degraded_reason, cost). `cost` is the per-search cost record
    on success and None on any degrade -- the parse call failed before returning
    usage, so there's no billed figure to report (which is not the same as $0).

    No Claude failure returns an error to the visitor. Structured filters are an
    enhancement; the index and scoring are local and keep working without them.
    Serving Pyrite for "shiny gold rock" beats serving a 500.

    Operators find out through the error-level log and the `degraded` flag, not
    through an outage. Alert on that log line -- an invalid key degrades silently
    and forever otherwise, which is the one real cost of this design.

    Clauses are ordered most-specific first: AuthenticationError, BadRequestError,
    and RateLimitError all subclass APIStatusError, so a bare APIStatusError above
    them would swallow all three.
    """
    try:
        filters, usage = parse_search_query_with_usage(query)
        return filters, None, usage_cost(usage, QUERY_MODEL)

    except anthropic.AuthenticationError:
        log.error("ANTHROPIC_API_KEY is missing or invalid -- serving degraded results")
        return fallback_filters(query), "auth_error", None

    except anthropic.BadRequestError as exc:
        # Bad schema, or an account that can't be billed. Not transient.
        log.error("Claude rejected the request -- serving degraded results: %s", exc.message)
        return fallback_filters(query), "bad_request", None

    except anthropic.RateLimitError:
        log.warning("Rate limited by the Claude API -- serving degraded results")
        return fallback_filters(query), "rate_limited", None

    except anthropic.APIConnectionError:
        log.warning("Could not reach the Claude API -- serving degraded results")
        return fallback_filters(query), "unreachable", None

    except anthropic.APIStatusError as exc:
        log.error("Claude API error %s -- serving degraded results: %s", exc.status_code, exc.message)
        return fallback_filters(query), "api_error", None

    except RuntimeError:
        # parse_search_query raises this on stop_reason == "refusal".
        log.warning("Claude declined to parse the query -- serving degraded results")
        return fallback_filters(query), "refused", None
