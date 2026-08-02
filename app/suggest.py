"""Audience-aware "You might also be interested in" suggestions.

Grounded + validated: every suggestion is derived from real collection facets and
verified to return at least one *new* specimen before it is shown, so no chip is
ever a dead end. Both audiences get the same response shape; only the candidate
generation differs.

  student    -> broaden. Sibling groups elsewhere in the collection, plus one
                "zoom in" facet within the current results. Exploratory.
  researcher -> pivot. Hold the structured filters the query used and vary one
                axis (locality, adjacent taxa). Precise.

Cost note: candidates arrive as structured facets, not fuzzy phrases, so
validation is a local predicate scan -- NOT another embedding pass. The parent
search already pays for the one MiniLM encode; this adds zero encodes. See
docs/audience-suggestions.md.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Callable, Optional, TYPE_CHECKING

from app.ai import SearchFilters

if TYPE_CHECKING:
    from app.search import SearchIndex

MAX_SUGGESTIONS = 4

# Plain-word labels/queries for the visitor-facing categories. Used on the
# multi-kingdom demo data; the live herbarium is Plantae-only so cross-category
# broadening simply produces no candidates there (validation drops them).
_CATEGORY_LABEL = {"plant": "Plants", "animal": "Animals", "mineral": "Minerals", "fossil": "Fossils"}

_REASON = {
    "broaden": "a broader area of the collection",
    "narrow": "a more specific group",
    "pivot": "closely related records",
}


@dataclass
class _Candidate:
    """A suggestion plus the predicate that proves it isn't empty.

    `query` is what the frontend re-runs (it goes back through the normal parser).
    `predicate` is how we verify grounding locally -- decoupled on purpose, so
    validation never needs to re-parse or re-embed.
    """

    label: str
    query: str
    kind: str  # "broaden" | "narrow" | "pivot"
    predicate: Callable[[dict], bool]


def _f(specimen: dict, field: str) -> str:
    """A specimen field, normalized for comparison. '', 'N/a', 'na' -> ''."""
    value = specimen.get(field)
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text.lower() in ("", "n/a", "na") else text.lower()


def build_suggestions(
    index: "SearchIndex",
    filters: SearchFilters,
    results: list[dict],
    audience: str,
    original_query: str = "",
) -> list[dict]:
    """Return up to MAX_SUGGESTIONS grounded, validated suggestion chips.

    Works even in degraded mode: it reads result anchors and collection facets,
    never Claude, so a search that fell back to semantic-only still gets
    suggestions.
    """
    anchors = _anchors(index, results)

    if audience == "researcher":
        candidates = _researcher_candidates(index, filters, anchors)
    else:
        candidates = _student_candidates(index, anchors)

    shown_ids = {r["id"] for r in results}
    return _validate_and_pack(index, candidates, shown_ids, filters, original_query)


def _anchors(index: "SearchIndex", results: list[dict]) -> dict:
    """Facets to build suggestions around.

    The `result_*` sets are what's *already on screen* -- they drive the "don't
    suggest what they're looking at" exclusion, so they come strictly from the
    results and are empty on a no-match search (turning it into a browse-the-
    collection set). The `dominant_*` anchors fall back to the whole collection
    when nothing matched, so zoom-in still has something to point at.
    """
    families = Counter(v for s in results if (v := _f(s, "family")))
    genera = Counter(v for s in results if (v := _f(s, "genus")))
    categories = Counter(c for s in results if (c := index.category_of(s)))

    coll_families = families or index.facets["family"]
    coll_genera = genera or index.facets["genus"]

    return {
        "dominant_family": _top_key(coll_families),
        "dominant_genus": _top_key(coll_genera),
        "result_families": {k.lower() for k in families},
        "result_genera": {k.lower() for k in genera},
        "result_categories": set(categories),
    }


def _top_key(counter: Counter) -> Optional[str]:
    return counter.most_common(1)[0][0].lower() if counter else None


# ── Student: broaden + one zoom-in ────────────────────────────────────────────


def _student_candidates(index: "SearchIndex", anchors: dict) -> list[_Candidate]:
    candidates: list[_Candidate] = []

    # 1. Broaden. Prefer sibling *categories* when the collection actually spans
    #    more than one (the demo data): "plants" -> "Minerals", "Fossils". On the
    #    Plantae-only herbarium there is one category, so fall back to sibling
    #    families, which is the meaningful breadth there.
    if len(index.facets["category"]) >= 2:
        for category, _ in index.facets["category"].most_common():
            if category in anchors["result_categories"]:
                continue
            label = _CATEGORY_LABEL.get(category, category.title())
            candidates.append(
                _Candidate(label, label, "broaden", lambda s, c=category: index.category_of(s) == c)
            )
    else:
        for family, _ in index.facets["family"].most_common():
            fam = family.lower()
            if fam in anchors["result_families"]:
                continue
            candidates.append(
                _Candidate(
                    _taxon_label(index, "family", fam, family),
                    family,
                    "broaden",
                    lambda s, fam=fam: _f(s, "family") == fam,
                )
            )

    # 2. Zoom in on the current results: a narrower taxon inside them.
    candidates.extend(_zoom_in(index, anchors))
    return candidates


def _zoom_in(index: "SearchIndex", anchors: dict) -> list[_Candidate]:
    # Results centered on a single genus -> offer a specific species in it.
    if len(anchors["result_genera"]) == 1 and anchors["dominant_genus"]:
        genus = anchors["dominant_genus"]
        species = Counter(
            _f(s, "species") for s in _members(index, "genus", genus) if _f(s, "species")
        )
        out = []
        for sp, _ in species.most_common():
            binomial = f"{genus.title()} {sp}"
            out.append(
                _Candidate(
                    binomial.title(),
                    binomial,
                    "narrow",
                    lambda s, g=genus, sp=sp: _f(s, "genus") == g and _f(s, "species") == sp,
                )
            )
        return out

    # Results span a family -> offer a genus within it.
    if anchors["dominant_family"]:
        family = anchors["dominant_family"]
        genera = Counter(
            _f(s, "genus") for s in _members(index, "family", family) if _f(s, "genus")
        )
        out = []
        for genus, _ in genera.most_common():
            if genus in anchors["result_genera"]:
                continue
            out.append(
                _Candidate(
                    _taxon_label(index, "genus", genus, genus.title()),
                    genus.title(),
                    "narrow",
                    lambda s, g=genus: _f(s, "genus") == g,
                )
            )
        return out

    return []


# ── Researcher: hold filters, pivot one axis ──────────────────────────────────


def _researcher_candidates(index: "SearchIndex", filters: SearchFilters, anchors: dict) -> list[_Candidate]:
    candidates: list[_Candidate] = []

    locality = (filters.locality or "").strip().lower()
    locality = locality if locality not in ("", "n/a", "na") else ""

    # Which taxon level did the query actually pin down? Most specific wins.
    taxon_field, taxon_value = _explicit_taxon(filters)

    # 1. Locality pivots -- the researcher's "take localities into account" case.
    if locality:
        if taxon_field:
            candidates.append(
                _Candidate(
                    f"{taxon_value.title()} elsewhere",
                    taxon_value.title(),
                    "pivot",
                    lambda s, fld=taxon_field, v=taxon_value.lower(), loc=locality: (
                        _f(s, fld) == v and _f(s, "locality") not in ("", loc)
                    ),
                )
            )
        candidates.append(
            _Candidate(
                f"More from {filters.locality}",
                filters.locality,
                "pivot",
                lambda s, loc=locality, fld=taxon_field, v=(taxon_value or "").lower(): (
                    _f(s, "locality") == loc and (not fld or _f(s, fld) != v)
                ),
            )
        )

    # 2. Adjacent taxa. Anchor on the level the query pinned, else on the results.
    anchor_genus = (filters.genus or "").lower() or anchors["dominant_genus"]
    anchor_family = (filters.family or "").lower() or _family_of_genus(index, anchor_genus) or anchors["dominant_family"]

    if anchor_genus:
        # Sibling genera in the same family (lateral, precise).
        family = _family_of_genus(index, anchor_genus) or anchor_family
        if family:
            for genus, _ in _genera_in_family(index, family):
                if genus == anchor_genus:
                    continue
                candidates.append(
                    _Candidate(
                        _taxon_label(index, "genus", genus, genus.title()),
                        genus.title(),
                        "pivot",
                        lambda s, g=genus: _f(s, "genus") == g,
                    )
                )
        # One drill into a species of the anchor genus.
        species = Counter(
            _f(s, "species") for s in _members(index, "genus", anchor_genus) if _f(s, "species")
        )
        for sp, _ in species.most_common(1):
            binomial = f"{anchor_genus.title()} {sp}"
            candidates.append(
                _Candidate(
                    binomial.title(),
                    binomial,
                    "narrow",
                    lambda s, g=anchor_genus, sp=sp: _f(s, "genus") == g and _f(s, "species") == sp,
                )
            )
    elif anchor_family:
        for genus, _ in _genera_in_family(index, anchor_family):
            candidates.append(
                _Candidate(
                    _taxon_label(index, "genus", genus, genus.title()),
                    genus.title(),
                    "pivot",
                    lambda s, g=genus: _f(s, "genus") == g,
                )
            )

    return candidates


# ── Validation + packing ──────────────────────────────────────────────────────


def _validate_and_pack(
    index: "SearchIndex",
    candidates: list[_Candidate],
    shown_ids: set,
    filters: SearchFilters,
    original_query: str,
) -> list[dict]:
    """Keep only candidates that match >=1 specimen not already on screen, dedupe
    by query, and cap. The grounding guarantee lives here: a chip that survives
    this loop always returns something new when clicked."""
    seen = {original_query.strip().lower(), (filters.semantic_query or "").strip().lower()}
    seen.discard("")

    packed: list[dict] = []
    for candidate in candidates:
        query = candidate.query.strip()
        key = query.lower()
        if not query or key in seen:
            continue
        # Structural validation -- a predicate scan, no embedding.
        if not _is_grounded(index, candidate, shown_ids):
            continue

        seen.add(key)
        packed.append(
            {
                "label": candidate.label,
                "query": query,
                "kind": candidate.kind,
                "reason": _REASON.get(candidate.kind, ""),
            }
        )
        if len(packed) >= MAX_SUGGESTIONS:
            break

    return packed


def _is_grounded(index: "SearchIndex", candidate: _Candidate, shown_ids: set) -> bool:
    """Does this candidate return real specimens worth showing?

    The bar differs by intent. A broaden/pivot is a lateral move: it must reveal
    at least one specimen not already on screen, or it adds nothing. A narrow is a
    zoom *into* the current results, so its hits are expected to already be shown;
    it only has to be non-empty and an actual subset (fewer than what's up now),
    otherwise it isn't narrowing anything.
    """
    matches = [s for s in index.specimens if candidate.predicate(s)]
    if not matches:
        return False

    if candidate.kind == "narrow":
        shown_count = len(shown_ids) or len(index.specimens)
        return len(matches) < shown_count

    return any(s["id"] not in shown_ids for s in matches)


# ── Small collection helpers ──────────────────────────────────────────────────


def _members(index: "SearchIndex", field: str, value: str) -> list[dict]:
    return [s for s in index.specimens if _f(s, field) == value]


def _genera_in_family(index: "SearchIndex", family: str) -> list[tuple[str, int]]:
    genera = Counter(_f(s, "genus") for s in _members(index, "family", family) if _f(s, "genus"))
    return genera.most_common()


def _family_of_genus(index: "SearchIndex", genus: Optional[str]) -> Optional[str]:
    if not genus:
        return None
    for specimen in index.specimens:
        if _f(specimen, "genus") == genus and _f(specimen, "family"):
            return _f(specimen, "family")
    return None


def _explicit_taxon(filters: SearchFilters) -> tuple[Optional[str], Optional[str]]:
    """The most specific taxon the query pinned down, as (field, value)."""
    if filters.species:
        return "species", filters.species
    if filters.genus:
        return "genus", filters.genus
    if filters.family:
        return "family", filters.family
    return None, None


def _taxon_label(index: "SearchIndex", field: str, value: str, fallback: str) -> str:
    """A friendly chip label for a taxon, borrowing an enrichment common name when
    one exists (genus-level names are reliable; family labels stay scientific to
    avoid labeling a whole family by one member's common name)."""
    if field == "genus":
        for specimen in index.specimens:
            if _f(specimen, field) == value:
                enrichment = index.enrichments.get(str(specimen["id"]))
                if enrichment and enrichment.get("common_names"):
                    return enrichment["common_names"][0].title()
                break
    return fallback
