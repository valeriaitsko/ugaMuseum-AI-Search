"""Common-name bridge for taxonomic orders.

Insect records carry a Latin order (Coleoptera, Diptera, ...) but visitors type
English ("beetles", "flies"). Orders are a *closed, tiny* vocabulary -- ~23 in
this collection, ever -- so the bridge is a hand table, not AI. (Species common
names, by contrast, are tens of thousands and open-ended; those need enrichment.)

Because this is a plain dictionary lookup with no API call, order-level common-name
search works with no credits and even when Claude is unreachable -- it runs as a
deterministic step in the search path, independent of the query parser.

COLLECTION-SPECIFIC. These are the orders actually present in the current
snapshot, most-common first. A curator should sanity-check the rarer ones (the
singleton orders at the bottom). If a future upload adds an order not listed here,
its records simply won't be reachable by English name until it's added -- harmless,
not a crash. Rebuild coverage with the order-coverage check to see what's present.

Counts (at build time): Lepidoptera 23.7k, Diptera 4.1k, Coleoptera 2.2k, then a
long tail. The top three are ~95% of the collection.
"""

import re

# Latin order -> the English words a visitor might type for it. Include singular
# AND plural forms explicitly; the matcher is whole-word, so "moth" won't match
# "moths" on its own.
ORDER_COMMON_NAMES: dict[str, list[str]] = {
    "Lepidoptera": ["butterfly", "butterflies", "moth", "moths"],
    "Diptera": ["fly", "flies", "gnat", "gnats", "midge", "midges", "mosquito", "mosquitoes"],
    "Coleoptera": ["beetle", "beetles", "weevil", "weevils"],
    "Trichoptera": ["caddisfly", "caddisflies", "caddis fly", "caddis flies", "caddis"],
    "Ephemeroptera": ["mayfly", "mayflies"],
    "Plecoptera": ["stonefly", "stoneflies"],
    "Odonata": ["dragonfly", "dragonflies", "damselfly", "damselflies"],
    "Hymenoptera": ["bee", "bees", "wasp", "wasps", "ant", "ants", "hornet", "hornets",
                    "sawfly", "sawflies"],
    "Hemiptera": ["true bug", "true bugs", "bug", "bugs", "aphid", "aphids",
                  "cicada", "cicadas", "leafhopper", "leafhoppers"],
    "Neuroptera": ["lacewing", "lacewings", "antlion", "antlions", "ant lion", "ant lions"],
    "Blattodea": ["cockroach", "cockroaches", "roach", "roaches", "termite", "termites"],
    "Thysanoptera": ["thrips", "thrip"],
    "Megaloptera": ["dobsonfly", "dobsonflies", "alderfly", "alderflies", "fishfly", "fishflies"],
    # --- crustaceans / non-insect arthropods ---
    "Isopoda": ["isopod", "isopods", "pillbug", "pillbugs", "woodlouse", "woodlice",
                "sowbug", "sowbugs", "roly poly"],
    "Amphipoda": ["amphipod", "amphipods", "scud", "scuds"],
    "Decapoda": ["crab", "crabs", "shrimp", "shrimps", "crayfish", "crawfish",
                 "lobster", "lobsters", "prawn", "prawns"],
    "Diplostraca": ["water flea", "water fleas", "clam shrimp"],
    "Trombidiformes": ["mite", "mites"],
    # --- molluscs, a fish, an amphibian, a hydrozoan (singletons; review these) ---
    "Venerida": ["clam", "clams", "venus clam", "venus clams"],
    "Littorinimorpha": ["sea snail", "sea snails", "periwinkle", "periwinkles"],
    "Perciformes": ["perch", "perches"],   # NOT "fish" -- Perciformes is one fish order, not all fish
    "Caudata": ["salamander", "salamanders", "newt", "newts"],
    "Anthoathecata": ["hydroid", "hydroids"],
}

# Reverse index: search term -> Latin order. Built once at import. Each order's
# own Latin name maps to itself, so a researcher typing "Coleoptera" gets the
# same order filter a visitor typing "beetles" gets -- otherwise the Latin name
# falls through to noisy semantic search (the embed text is Latin binomials, so
# "Coleoptera" doesn't reliably land on beetles).
_TERM_TO_ORDER: dict[str, str] = {}
for _order, _terms in ORDER_COMMON_NAMES.items():
    _TERM_TO_ORDER[_order.lower()] = _order
    for _term in _terms:
        _TERM_TO_ORDER[_term.lower()] = _order

# One precompiled whole-word pattern per term. `re.escape` handles multi-word
# terms like "true bug"; \b on each side keeps "fly" from matching "butterfly".
_TERM_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(rf"\b{re.escape(term)}\b"), order)
    for term, order in _TERM_TO_ORDER.items()
]


def detect_orders(query: str) -> list[str]:
    """Latin orders implied by common-name words in the query. Deterministic, no API.

    "beetles from Georgia" -> ["Coleoptera"].  "asdf" -> [].  Multiple distinct
    orders ("beetles and flies") return both, so the caller can OR them.
    """
    q = query.lower()
    found: list[str] = []
    for pattern, order in _TERM_PATTERNS:
        if order not in found and pattern.search(q):
            found.append(order)
    return found
