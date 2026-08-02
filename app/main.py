import logging
from typing import Literal

import anthropic
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.ai import SearchFilters, fallback_filters, parse_search_query
from app.search import SearchIndex
from app.suggest import build_suggestions

log = logging.getLogger(__name__)

app = FastAPI(title="Museum AI Search")

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

# Built once at import: loads enrichments, encodes vectors (cached on disk).
# Doing this per request would re-encode the whole collection every search.
index = SearchIndex()


class SearchRequest(BaseModel):
    query: str
    # Drives the "You might also be interested in" suggestions: students get
    # broader sibling groups, researchers get precise pivots on the filters they
    # used. Defaults to "student" to match the frontend's own default.
    audience: Literal["student", "researcher"] = "student"


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

    filters, degraded_reason = _parse_or_degrade(query)
    results = index.search(filters)

    # Grounded, validated, and computed from local facets -- no extra Claude call,
    # and it still works when the parse degraded (it reads results, not filters).
    suggestions = build_suggestions(index, filters, results, req.audience, query)

    return {
        "original_query": query,
        "ai_parsed_query": filters.model_dump(),
        "results": results,
        "suggestions": suggestions,
        # True when Claude was unreachable and the filters are empty. The frontend
        # should say so -- a visitor who searched "collected by Emily Davis" and
        # silently got unfiltered results deserves to know why.
        "degraded": degraded_reason is not None,
        "degraded_reason": degraded_reason,
    }


def _parse_or_degrade(query: str) -> tuple[SearchFilters, str | None]:
    """Parse the query with Claude, or fall back to semantic-only search.

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
        return parse_search_query(query), None

    except anthropic.AuthenticationError:
        log.error("ANTHROPIC_API_KEY is missing or invalid -- serving degraded results")
        return fallback_filters(query), "auth_error"

    except anthropic.BadRequestError as exc:
        # Bad schema, or an account that can't be billed. Not transient.
        log.error("Claude rejected the request -- serving degraded results: %s", exc.message)
        return fallback_filters(query), "bad_request"

    except anthropic.RateLimitError:
        log.warning("Rate limited by the Claude API -- serving degraded results")
        return fallback_filters(query), "rate_limited"

    except anthropic.APIConnectionError:
        log.warning("Could not reach the Claude API -- serving degraded results")
        return fallback_filters(query), "unreachable"

    except anthropic.APIStatusError as exc:
        log.error("Claude API error %s -- serving degraded results: %s", exc.status_code, exc.message)
        return fallback_filters(query), "api_error"

    except RuntimeError:
        # parse_search_query raises this on stop_reason == "refusal".
        log.warning("Claude declined to parse the query -- serving degraded results")
        return fallback_filters(query), "refused"
