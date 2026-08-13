"""Eval cases for the query parser, grounded in the live insect collection.

Every value here is real: families (Syrphidae, Nymphalidae, Erebidae), genera
(Papilio, Catocala, Mixogaster), and localities that appear in the current
snapshot. The three single-record genera (Mixogaster, Sphecomyia, Xanthandrus)
give unambiguous top-1 answers via exact genus match -- which works without
enrichment, so those top-1 checks are valid right now.

What gets measured per case:

  expect   Fields the parser SHOULD extract -- kept conservative, asserted only
           where a competent parser has no real choice. For a common-name query
           that maps to a real taxon (hoverfly -> Syrphidae, monarch -> Danaus)
           the taxonomic field is NEITHER expected NOR forbidden, so a model that
           correctly knows it isn't penalised and one that doesn't isn't failed.

  forbid   Fields that MUST come back None. This is the sharp instrument -- a
           parser that invents locality="Georgia" from "butterflies" silently
           hides most of the collection. Hallucinated filters are worse than
           missing ones (a miss degrades to semantic search; a wrong filter
           excludes).

  top1     The specimen that should rank first. Set only for the single-record
           genera, where exact-match ranking is unambiguous pre-enrichment.
           None elsewhere -- common-name top-1 (monarch -> Danaus plexippus)
           needs the enrichment pass to work, so it can't be validated yet.

NOTE: the parser does NOT produce an `order` field -- taxonomic order is handled
by the deterministic detect_orders() step in main.py, not by Claude -- so no case
expects one. There is no `collector` field either: the SearchFilters schema
dropped it, since the insect snapshot carries no collector data.
"""

CATEGORICAL_FIELDS = ("kingdom", "category", "family", "genus", "species")
FREETEXT_FIELDS = ("locality",)
ALL_FIELDS = CATEGORICAL_FIELDS + FREETEXT_FIELDS
NO_FILTERS = ALL_FIELDS


CASES = [
    # --- explicit Latin taxonomy; single-record genera give a real top-1 --------
    dict(query="Mixogaster",
         expect=dict(genus="Mixogaster"), forbid=("locality",), top1=460829),
    dict(query="Sphecomyia",
         expect=dict(genus="Sphecomyia"), forbid=("locality",), top1=460942),
    dict(query="Xanthandrus mexicanus",
         expect=dict(genus="Xanthandrus"), forbid=("locality",), top1=460983),
    dict(query="family Syrphidae",
         expect=dict(family="Syrphidae"),
         forbid=("genus", "species", "locality"), top1=None),
    dict(query="Nymphalidae",
         expect=dict(family="Nymphalidae"), forbid=("locality",), top1=None),
    dict(query="Papilio",
         expect=dict(genus="Papilio"), forbid=("locality",), top1=None),
    dict(query="Catocala",
         expect=dict(genus="Catocala"), forbid=("locality",), top1=None),
    dict(query="Erebidae specimens",
         expect=dict(family="Erebidae"),
         forbid=("genus", "species", "locality"), top1=None),

    # --- common-name groups: must not invent a specific taxon -------------------
    dict(query="butterflies", expect=dict(category="insect"),
         forbid=("family", "genus", "species", "locality"), top1=None),
    dict(query="moths", expect=dict(category="insect"),
         forbid=("family", "genus", "species", "locality"), top1=None),
    dict(query="beetles", expect=dict(category="insect"),
         forbid=("family", "genus", "species", "locality"), top1=None),
    dict(query="colorful winged insects", expect=dict(category="insect"),
         forbid=("family", "genus", "species", "locality"), top1=None),

    # --- common names that DO map to a taxon: don't forbid it, don't require it --
    dict(query="hoverflies", expect=dict(category="insect"),
         forbid=("locality",), top1=None),
    dict(query="monarch butterfly", expect=dict(category="insect"),
         forbid=("locality",), top1=None),

    # --- locality (free-text: "Georgia" is a substring of the catalog value) -----
    dict(query="insects from Georgia",
         expect=dict(locality="Georgia"), forbid=("genus", "species"), top1=None),
    dict(query="Syrphidae collected in Georgia",
         expect=dict(family="Syrphidae", locality="Georgia"),
         forbid=("genus", "species"), top1=None),
    dict(query="specimens from Athens, Georgia",
         expect=dict(locality="Georgia"), forbid=("genus", "species"), top1=None),

    # --- broad category / kingdom -----------------------------------------------
    dict(query="insects", expect=dict(category="insect"),
         forbid=("family", "genus", "species", "locality"), top1=None),
    dict(query="animals", expect=dict(kingdom="Animalia"),
         forbid=("genus", "species", "locality"), top1=None),

    # --- nonsense guard: extract nothing rather than pattern-match into a filter -
    dict(query="asdfghjkl", expect={}, forbid=NO_FILTERS, top1=None),
]
