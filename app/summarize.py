"""Audience-aware natural-language summary of the results a visitor is looking at.

On-demand (a "Summarize these results" button), not per search -- so it costs a
Haiku call only when someone actually asks for it.

Grounded like the suggestions: the numbers (counts, localities, dominant taxa) are
computed locally from the shown specimens, and Claude is asked only to PHRASE those
facts, never to supply them -- with an explicit instruction not to invent taxa,
places, or figures. So the tone adapts by audience while the content stays true to
the catalog.

  student    -> short, warm, plain words; ends by inviting a tap on a card.
  researcher -> concise and precise; distribution across localities and taxa.

Degrades: if Claude is unreachable the same local stats fill a deterministic
template, so the button still returns a (plainer) summary offline.
"""

from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING

import anthropic

from app.ai import QUERY_MODEL, get_client, usage_cost

if TYPE_CHECKING:
    from app.search import SearchIndex

# Haiku, same as the query parser: the task is small and sits in front of a button.
SUMMARY_MODEL = QUERY_MODEL

# How many of each facet to name in the facts we hand Claude -- enough to be
# specific, bounded so the prompt stays small.
_TOP_N = 5


def _f(specimen: dict, field: str) -> str:
    """A specimen field, normalized. '', 'N/a', 'na' -> ''."""
    value = specimen.get(field)
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text.lower() in ("", "n/a", "na") else text


def _dedup_binomial(genus: str, species: str) -> str:
    """The catalog stores `species` as either the epithet or the full binomial.
    Collapse a doubled genus so "Papilio" + "Papilio glaucus" -> "Papilio glaucus".
    """
    genus, species = genus.strip(), species.strip()
    if not species:
        return genus
    if species.lower().startswith(genus.lower() + " "):
        return species
    return f"{genus} {species}".strip()


def compute_stats(index: "SearchIndex", specimens: list[dict]) -> dict:
    """Everything the summary is allowed to state, computed from the shown records."""
    categories = Counter(c for s in specimens if (c := index.category_of(s)))
    families = Counter(v for s in specimens if (v := _f(s, "family")))
    genera = Counter(v for s in specimens if (v := _f(s, "genus")))
    localities = Counter(v for s in specimens if (v := _f(s, "locality")))
    species = Counter(
        b for s in specimens
        if (b := _dedup_binomial(_f(s, "genus"), _f(s, "species"))) and _f(s, "species")
    )
    return {
        "total": len(specimens),
        "categories": categories,
        "families": families,
        "genera": genera,
        "localities": localities,
        "species": species,
        "num_localities": len(localities),
    }


def _fmt_counter(counter: Counter, n: int = _TOP_N) -> str:
    return ", ".join(f"{name} ({count})" for name, count in counter.most_common(n)) or "none recorded"


def _format_facts(stats: dict, query: str) -> str:
    """The grounded facts block handed to Claude. Claude phrases only these."""
    lines = []
    if query:
        lines.append(f'Search query: "{query}"')
    lines.append(f"Specimens shown: {stats['total']}")
    if stats["categories"]:
        lines.append(f"Groups: {_fmt_counter(stats['categories'])}")
    if stats["families"]:
        lines.append(f"Families: {_fmt_counter(stats['families'])}")
    if stats["genera"]:
        lines.append(f"Genera: {_fmt_counter(stats['genera'])}")
    if stats["species"]:
        lines.append(f"Species: {_fmt_counter(stats['species'])}")
    if stats["localities"]:
        lines.append(f"Localities ({stats['num_localities']}): {_fmt_counter(stats['localities'])}")
    return "\n".join(lines)


SUMMARY_SYSTEM = {
    "student": (
        "You write a short, friendly summary of museum search results for a curious kid or "
        "young student. Two or three short sentences, warm and simple, everyday words. Use ONLY "
        "the facts you are given -- never invent a species, place, or number that is not listed. "
        "End by inviting them to tap or click a card to see its full name and where it was found. "
        "At most one emoji. Just a little paragraph -- no headings, no lists."
    ),
    "researcher": (
        "You write a concise, precise summary of museum specimen search results for a researcher. "
        "Two to four sentences, neutral and informative. Bring out the distribution: how many "
        "specimens, across how many localities (name the notable ones with counts), and the "
        "dominant taxa with counts. Use ONLY the facts you are given -- never invent a taxon, "
        "place, or number that is not listed. No second person, no emoji, no lists -- plain prose."
    ),
}


def _template_summary(stats: dict, audience: str) -> str:
    """Deterministic fallback when Claude is unreachable -- plainer, still grounded."""
    total = stats["total"]
    nloc = stats["num_localities"]
    top_group = stats["categories"].most_common(1)
    top_taxon = (stats["genera"] or stats["families"]).most_common(1)

    if audience == "researcher":
        parts = [f"{total} specimen{'s' if total != 1 else ''} across "
                 f"{nloc} localit{'ies' if nloc != 1 else 'y'}."]
        if top_taxon:
            name, count = top_taxon[0]
            parts.append(f"Dominant taxon {name} ({count} of {total}).")
        return " ".join(parts)

    group_word = top_group[0][0] + "s" if top_group else "specimens"
    place_phrase = (f" from {nloc} different place{'s' if nloc != 1 else ''}" if nloc else "")
    return (f"We found {total} {group_word}{place_phrase}! "
            f"Click a card to learn more about each one.")


def summarize_results(
    index: "SearchIndex", specimens: list[dict], audience: str, query: str = ""
) -> dict:
    """Return {summary, degraded, cost} for the given shown specimens.

    Numbers come from compute_stats (local, exact); Claude only phrases them. On any
    Claude failure the same stats fill a template and `degraded` is True -- the
    button never returns nothing.
    """
    stats = compute_stats(index, specimens)
    audience = "researcher" if audience == "researcher" else "student"

    try:
        response = get_client().messages.create(
            model=SUMMARY_MODEL,
            max_tokens=300,
            system=SUMMARY_SYSTEM[audience],
            messages=[{"role": "user", "content": _format_facts(stats, query)}],
        )
        if response.stop_reason == "refusal":
            raise RuntimeError("Claude declined to summarize these results.")
        text = "".join(b.text for b in response.content if b.type == "text").strip()
        if not text:
            raise RuntimeError("Empty summary from Claude.")
        return {"summary": text, "degraded": False, "cost": usage_cost(response.usage, SUMMARY_MODEL)}

    except (anthropic.APIError, RuntimeError):
        # Same philosophy as the query parser: a summary is an enhancement, not a
        # feature the visitor should lose to a Claude outage. Fall back to the
        # deterministic phrasing of the same grounded stats.
        return {"summary": _template_summary(stats, audience), "degraded": True, "cost": None}
