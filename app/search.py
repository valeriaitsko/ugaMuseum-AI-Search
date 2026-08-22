"""Hybrid retrieval: structured filters + semantic similarity over enriched text."""

import json
import re
from collections import Counter
from pathlib import Path

from app import embeddings
from app.ai import SearchFilters
from app.specimens import load_specimens

ENRICHED_PATH = Path(__file__).parent / "data" / "enriched.json"

# How much a matched structured field is worth. Exact taxonomic hits are strong
# evidence; a locality mentioned in passing is weaker.
FIELD_WEIGHTS = {
    "family": 3.0,
    "genus": 3.0,
    "species": 3.0,
    "locality": 2.0,
}

# Cosine sits in [0, 1] for anything plausibly related, so this is roughly the
# ceiling a purely semantic match can contribute. Tuned to sit just under a
# two-field exact match: someone who names a family and a locality should
# outrank a fuzzy conceptual hit.
SEMANTIC_WEIGHT = 5.0

# Weight per matched word when embeddings are unavailable. Deliberately low --
# lexical matching is the fallback, not the plan.
LEXICAL_WEIGHT = 0.75

# Substring matching made "a" and "in" score points against every record.
# Word-boundary matching plus a stopword list is the fix.
STOPWORDS = frozenset(
    "a an the of in on at from with and or for to is are was were this that "
    "show me all find any some please specimen specimens".split()
)


def _tokens(text: str) -> list[str]:
    return [
        word
        for word in re.findall(r"[a-z0-9]+", text.lower())
        if len(word) > 2 and word not in STOPWORDS
    ]


def _contains_word(haystack: str, needle: str) -> bool:
    """Whole-word containment. `_contains_word("forest", "ore")` is False."""
    return re.search(rf"\b{re.escape(needle.lower())}\b", haystack.lower()) is not None


class SearchIndex:
    """Loads enrichments once at startup and, if possible, embeds them."""

    def __init__(self) -> None:
        self.specimens = load_specimens()
        self.enrichments = self._load_enrichments()
        self.search_texts = [self._search_text(s) for s in self.specimens]
        self.position = {s["id"]: i for i, s in enumerate(self.specimens)}

        self.semantic = embeddings.available()
        self.vectors = embeddings.build_index(self.search_texts) if self.semantic else None

        # Collection-wide facet counts, computed once. Suggestion generation reads
        # these to answer "what else is in the collection near this result?" without
        # rescanning every request. See app/suggest.py.
        self.facets = self._build_facets()

    @property
    def enriched(self) -> bool:
        return bool(self.enrichments)

    def category_of(self, specimen: dict) -> str | None:
        """Public accessor for a specimen's enrichment category (may be None).

        Suggestion generation lives in a sibling module and needs this without
        reaching into the private helper.
        """
        return self._category(specimen)

    def _build_facets(self) -> dict[str, Counter]:
        facets: dict[str, Counter] = {
            "category": Counter(),
            "family": Counter(),
            "genus": Counter(),
            "species": Counter(),
            "locality": Counter(),
        }
        for specimen in self.specimens:
            category = self._category(specimen)
            if category:
                facets["category"][category] += 1
            for field in ("family", "genus", "species", "locality"):
                value = specimen.get(field)
                # "N/a" is the herbarium snapshot's null sentinel for locality and
                # sometimes species; it is not a facet anyone would want suggested.
                if value and str(value).strip().lower() not in ("", "n/a", "na"):
                    facets[field][str(value).strip()] += 1
        return facets

    def _load_enrichments(self) -> dict[str, dict]:
        if not ENRICHED_PATH.exists():
            return {}
        return json.loads(ENRICHED_PATH.read_text(encoding="utf-8"))

    # Fields kept OUT of the embedded fallback text. `order`/`phylum`/`class` are
    # Latin rank words no visitor types, and excluding them keeps the cached vectors
    # byte-identical (no re-encode). `discipline` is an internal biology/archaeology
    # flag, not search content. The archaeology ranks (culturalPeriod/points/
    # specimen) are deliberately NOT here -- "Dalton", "Archaic Period" etc. are
    # exactly what a visitor would search an artifact by.
    _NON_EMBEDDED_FIELDS = ("id", "discipline", "order", "phylum", "class")

    def _search_text(self, specimen: dict) -> str:
        """The enriched paragraph if we have one; otherwise the bare catalog row."""
        enrichment = self.enrichments.get(str(specimen["id"]))
        if enrichment:
            return enrichment["search_text"]
        return " ".join(
            str(v) for k, v in specimen.items() if k not in self._NON_EMBEDDED_FIELDS and v
        )

    def _category(self, specimen: dict) -> str | None:
        enrichment = self.enrichments.get(str(specimen["id"]))
        return enrichment["category"] if enrichment else None

    def _passes_hard_filters(
        self, specimen: dict, filters: SearchFilters, orders: list[str] | None = None
    ) -> bool:
        """Categorical filters exclude rather than down-rank.

        Asking for minerals and getting a wolf at rank 4 is worse than getting
        four results. Free-text fields stay soft because the visitor's spelling
        of a locality rarely matches the catalog's exactly.
        """
        if filters.kingdom and specimen.get("kingdom") != filters.kingdom:
            return False

        if filters.category:
            category = self._category(specimen)
            # An un-enriched specimen has no category. Don't exclude it on a
            # field we never computed.
            if category is not None and category != filters.category:
                return False

        # Order is a hard filter like kingdom, but sourced from the deterministic
        # common-name detector (app/orders.py), not from Claude. `orders` is a
        # list because "beetles and flies" yields two; a specimen passes if its
        # order is any of them. A record with no order (not identified that far)
        # is excluded when an order is requested -- "beetles" should return
        # confirmed beetles, and coverage is ~99% anyway.
        if orders and specimen.get("order") not in orders:
            return False

        return True

    def _field_score(self, specimen: dict, filters: SearchFilters) -> tuple[float, list[str]]:
        score = 0.0
        matched: list[str] = []

        # Exact taxonomic fields. Each is a dedicated key on the flattened record
        # (family, genus, species), so this is a direct case-insensitive equality
        # test -- not a scan through a scientific-name string. Guard for null, since
        # a specimen with no current determination has these as None.
        for field in ("family", "genus", "species"):
            value = getattr(filters, field)
            record_value = specimen.get(field)
            if value and record_value and record_value.lower() == value.lower():
                score += FIELD_WEIGHTS[field]
                matched.append(f"{field}={record_value}")

        # Locality stays soft: a visitor's phrasing rarely equals the catalog's
        # exactly, so match it as a substring.
        if filters.locality and specimen.get("locality"):
            if filters.locality.lower() in specimen["locality"].lower():
                score += FIELD_WEIGHTS["locality"]
                matched.append(f"locality~{filters.locality}")

        return score, matched

    def _lexical_score(self, text: str, query: str) -> float:
        """Whole-word matching, tolerant of a trailing plural 's'.

        Only the fallback path folds plurals. Exact-field matching must not:
        `species="lupus"` matching a specimen named "Lupu" would be a real error,
        whereas "crystal" failing to find "crystals" is just a bad search.

        Naive suffix stripping, not a stemmer. Embeddings handle morphology
        properly; this only has to be less wrong than substring matching.
        """
        hits = 0
        for word in _tokens(query):
            variants = {word, word.rstrip("s"), word + "s"}
            if any(_contains_word(text, v) for v in variants if len(v) > 2):
                hits += 1
        return LEXICAL_WEIGHT * hits

    def search(
        self, filters: SearchFilters, orders: list[str] | None = None, limit: int = 10
    ) -> list[dict]:
        candidates = [s for s in self.specimens if self._passes_hard_filters(s, filters, orders)]
        if not candidates:
            return []

        if self.semantic and self.vectors is not None:
            all_scores = embeddings.similarities(filters.semantic_query, self.vectors)
        else:
            all_scores = None

        results = []
        for specimen in candidates:
            score, matched = self._field_score(specimen, filters)
            position = self.position[specimen["id"]]

            if all_scores is not None:
                similarity = max(float(all_scores[position]), 0.0)
                score += SEMANTIC_WEIGHT * similarity
                matched.append(f"similarity={similarity:.2f}")
            else:
                score += self._lexical_score(self.search_texts[position], filters.semantic_query)

            if score <= 0:
                continue

            # Only the museum's own fields go out. Enrichment is retrieval-only:
            # returning a model-generated common name here would let it surface
            # in the UI as though the catalog asserted it.
            results.append({**specimen, "score": round(score, 2), "matched_on": matched})

        results.sort(key=lambda r: r["score"], reverse=True)
        return results[:limit]
