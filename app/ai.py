"""Claude API layer: query parsing (per request) and specimen enrichment (offline)."""

from typing import Literal, Optional

import anthropic
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from app.specimens import CATEGORIES, KINGDOMS

load_dotenv()

# Two jobs, opposite cost curves and opposite blast radii.
#
# ENRICH_MODEL runs once per specimen, ever, and its output is frozen into
# enriched.json and read by every search from then on. A bad alias -- calling
# Apatosaurus a predator -- is permanent, silent, and unreviewable at collection
# scale. Errors compound. Don't economize.
#
# QUERY_MODEL runs on every search, forever, so it dominates lifetime cost. The
# task is small: read ten words, fill eight schema-constrained fields. A mistake
# affects one search, the visitor retypes, it's gone. Errors evaporate. And it
# sits in front of a search box, so Haiku being fast is a feature, not a discount.
ENRICH_MODEL = "claude-opus-4-8"
QUERY_MODEL = "claude-haiku-4-5"

# List price (USD) per million tokens, as (input, output). Goes stale silently if
# Anthropic changes rates -- it's a list-price estimate for these tokens, not a
# bill; the console is the record. An unknown model prices to 0 rather than guess.
MODEL_PRICES: dict[str, tuple[float, float]] = {
    "claude-opus-4-8": (5.0, 25.0),
    "claude-sonnet-5": (3.0, 15.0),
    "claude-haiku-4-5": (1.0, 5.0),
}


def usage_cost(usage, model: str) -> dict:
    """A per-call cost record from an Anthropic usage object, at list price.

    `input_tokens` is the uncached input; cache writes bill ~1.25x and reads ~0.1x
    the input rate. The query parser doesn't cache (short prompt), so those are
    normally 0 -- the formula just stays correct if caching is added later.
    """
    price_in, price_out = MODEL_PRICES.get(model, (0.0, 0.0))
    cache_write = getattr(usage, "cache_creation_input_tokens", 0) or 0
    cache_read = getattr(usage, "cache_read_input_tokens", 0) or 0
    cost_usd = (
        usage.input_tokens * price_in
        + cache_write * price_in * 1.25
        + cache_read * price_in * 0.10
        + usage.output_tokens * price_out
    ) / 1_000_000
    return {
        "model": model,
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "cache_write_tokens": cache_write,
        "cache_read_tokens": cache_read,
        "cost_usd": round(cost_usd, 6),
    }


_client: anthropic.Anthropic | None = None


def get_client() -> anthropic.Anthropic:
    """Built on first use, not at import.

    Anthropic() raises when no credentials are present, and search.py imports
    SearchFilters from this module -- a module-level client would make the whole
    app unimportable (and untestable) without a key.

    No api_key argument on purpose: the SDK reads ANTHROPIC_API_KEY, and also
    picks up an `ant auth login` profile when the env var is unset.
    """
    global _client
    if _client is None:
        _client = anthropic.Anthropic()
    return _client

# Must stay in sync with specimens.KINGDOMS (which reflects what the snapshot
# actually holds). Now a mixed snapshot -- Plantae (herbarium) + Animalia (insects).
# If build_snapshot reports a new kingdom, add it in both places or the parser can
# never emit it.
Kingdom = Literal["Plantae", "Animalia"]
Category = Literal["plant", "animal", "insect", "mineral", "fossil"]


class SearchFilters(BaseModel):
    """Structured filters extracted from a visitor's natural-language query.

    Every field is schema-enforced, so `kingdom` can never come back as
    "Mineralia" or as a list. That guarantee is what makes the search code
    below able to call .lower() without defensive coercion.
    """

    kingdom: Optional[Kingdom] = Field(
        None, description="Only if the query names a kingdom explicitly or unambiguously implies one."
    )
    category: Optional[Category] = Field(
        None, description="Broad visitor-facing grouping. 'fossil' wins over 'animal' for extinct specimens."
    )
    family: Optional[str] = Field(None, description="Taxonomic family, e.g. Canidae.")
    genus: Optional[str] = Field(None, description="Genus only, e.g. Canis.")
    species: Optional[str] = Field(None, description="Species epithet only, e.g. lupus.")
    locality: Optional[str] = Field(None, description="Place the specimen was collected.")
    semantic_query: str = Field(
        description="What the visitor is conceptually looking for, in plain words. "
        "Never empty -- fall back to the original query."
    )
    explanation: str = Field(description="One sentence on how you read the query.")


# Retrieval-only aliases for one specimen. IMPORTANT: none of this is catalog data
# -- it's model-generated to make the record findable, never shown to visitors as
# fact (see enrich.py). Kept as a comment, not a docstring: Pydantic sends a class
# docstring to the model as the schema's description, so it would be billed on every
# one of ~31,800 enrichment calls. The field descriptions below are deliberately
# terse for the same reason -- a longer schema is fixed input re-sent every call.
# A 50-record A/B confirmed these short descriptions match the verbose originals on
# quality while cutting input tokens ~42% (~$56 off the batched full run).
class Enrichment(BaseModel):
    common_names: list[str] = Field(description="Everyday names.")
    abbreviations: list[str] = Field(description="Shorthand a visitor might type.")
    search_aliases: list[str] = Field(
        description="Broader search words: body plan, diet, habitat, grouping."
    )
    category: Category
    search_text: str = Field(
        description="Short concept-dense phrase naming what the specimen IS; see the "
        "enrichment system prompt for the exact shape."
    )


QUERY_SYSTEM = f"""You extract structured search filters for a natural history museum catalog.

Only populate a field when the query genuinely supports it. Leave it null otherwise --
a wrong filter silently hides correct results, which is worse than no filter.

The catalog stores kingdom as exactly one of: {", ".join(KINGDOMS)}.
The catalog groups specimens into: {", ".join(CATEGORIES)}.

Category reflects how a visitor groups things, which is not always the kingdom:
- Insects and other arthropods (beetles, flies, butterflies, bees, spiders) are
  category 'insect', not 'animal', even though their kingdom is Animalia.
- Extinct organisms known from fossils are category 'fossil', not 'animal'.
- 'animal' is for vertebrates and other non-insect, non-fossil animals.

semantic_query should capture intent, not keywords. For "big scary dinosaur" write
"large predatory dinosaur", not "big scary dinosaur"."""


# This prompt is deliberately terse: it's fixed input re-sent on every one of
# ~31,800 enrichment calls, and it's below Opus's 4,096-token minimum cacheable
# prefix, so it can't be cached -- shrinking it is the only lever. A 50-record A/B
# vs a much longer, example-and-justification-heavy version showed no quality loss
# (the extra prose was steering a human reader, not the model) while cutting total
# input ~42%. Keep new rules as short directives; don't reintroduce the essays.
ENRICH_SYSTEM = """You add search aliases to a museum specimen so visitors can find it. You get the catalog's fields for one specimen.

Rules:
- Never invent facts (date, collector, locality, institution, measurement). Output is for text matching, not display.
- Only add names/descriptors grounded in the given taxonomy or material ("Canis lupus" -> "gray wolf", "predator"; not "collected 1923").
- Prefer words a non-specialist types ("T rex", "wolf skeleton"), not jargon.
- If unsure an alias is correct, leave it out.
- search_text: a short concept-dense phrase (~10-20 words) of what the specimen IS -- scientific name, common names, key descriptors. No locality/collector, no full sentences.
- category by how visitors group it: insects/arthropods -> 'insect'; fossils -> 'fossil'; other animals -> 'animal'; plants -> 'plant'; minerals -> 'mineral'.

Example search_text: "Brachypalpus oarus, a hoverfly / flower fly. Bee-mimic pollinating fly, family Syrphidae." """


def parse_search_query_with_usage(query: str, model: str | None = None):
    """parse_search_query, plus the token usage and the model that ran.

    The eval harness needs both -- to price a run and to compare models on the
    same queries. `model` overrides QUERY_MODEL for exactly that purpose.
    """
    response = get_client().messages.parse(
        model=model or QUERY_MODEL,
        max_tokens=4096,
        system=QUERY_SYSTEM,
        messages=[{"role": "user", "content": query}],
        output_format=SearchFilters,
    )

    if response.stop_reason == "refusal":
        raise RuntimeError("Claude declined to parse this query.")

    parsed = response.parsed_output
    # The schema requires semantic_query, but an empty string would still validate
    # and would silently disable the whole semantic half of the search.
    if not parsed.semantic_query.strip():
        parsed.semantic_query = query
    return parsed, response.usage


def parse_search_query(query: str) -> SearchFilters:
    """Turn a visitor's query into structured filters. Raises on API failure."""
    filters, _ = parse_search_query_with_usage(query)
    return filters


def fallback_filters(query: str) -> SearchFilters:
    """Filters for when Claude can't be reached.

    Structured extraction is the only part of search that needs the API. The
    embedding index, the scoring, and the enriched text all live locally, so a
    Claude outage should cost the visitor their filters -- not their search.

    Every optional field stays None, which means no hard filters and no field
    boosts. The query goes through as-is to the semantic half.
    """
    return SearchFilters(
        semantic_query=query,
        explanation="Claude was unavailable; searched without structured filters.",
    )


def enrich_specimen(specimen: dict, model: str | None = None) -> Enrichment:
    """Generate retrieval-only aliases for one specimen. Raises on API failure.

    `model` overrides ENRICH_MODEL -- used to sample-compare models on a subset
    before committing the full (expensive) pass to one of them.
    """
    catalog_fields = "\n".join(
        f"{key}: {value}" for key, value in specimen.items() if key != "id"
    )

    response = get_client().messages.parse(
        model=model or ENRICH_MODEL,
        max_tokens=4096,
        system=ENRICH_SYSTEM,
        messages=[{"role": "user", "content": catalog_fields}],
        output_format=Enrichment,
    )

    if response.stop_reason == "refusal":
        raise RuntimeError(f"Claude declined to enrich specimen {specimen['id']}.")

    return response.parsed_output
