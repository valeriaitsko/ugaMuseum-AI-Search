"""Eval cases for the query parser, grounded in DEMO_SPECIMENS.

Three things get measured per case:

  expect   Fields the parser SHOULD extract. Kept deliberately conservative --
           only asserted where a competent parser has no real choice. Asserting
           a debatable field would punish correct behavior and make the eval
           lie to you.

  forbid   Fields that MUST come back None. This is the sharp instrument. A
           parser that invents `locality="South Dakota"` from the query "T rex"
           has silently hidden every T. rex specimen collected anywhere else.
           Hallucinated filters are worse than missing ones, because a missing
           filter degrades to semantic search while a wrong one excludes.

  top1     The specimen that should rank first end-to-end. None when the query
           is legitimately ambiguous (four specimens are fossils). Depends on
           enrichment AND embeddings, so it measures the whole pipeline, not
           just the parser.

Specimen ids, for reference:
  1 Rosa carolina        5 Odocoileus virginianus   9 Ammonite sp.
  2 Quercus alba         6 Chelonia mydas          10 Quartz
  3 Pinus taeda          7 Triceratops horridus    11 Pyrite
  4 Canis lupus          8 Tyrannosaurus rex       12 Apatosaurus louisae
"""

# Compared case-insensitively, exact.
CATEGORICAL_FIELDS = ("kingdom", "category", "family", "genus", "species")

# Compared case-insensitively, expected-is-substring-of-parsed. The catalog says
# "Madison County, Georgia"; a parser that extracts "Georgia" is right.
FREETEXT_FIELDS = ("locality", "collector")

ALL_FIELDS = CATEGORICAL_FIELDS + FREETEXT_FIELDS

# Shorthand for "this query names nothing structured; it is pure semantics."
NO_FILTERS = ALL_FIELDS


CASES = [
    # --- explicit taxonomy -------------------------------------------------
    dict(query="Canis lupus",
         expect=dict(genus="Canis", species="lupus"),
         forbid=("locality", "collector"), top1=4),
    dict(query="Tyrannosaurus rex",
         expect=dict(genus="Tyrannosaurus", species="rex"),
         forbid=("locality", "collector"), top1=8),
    dict(query="Quercus",
         expect=dict(genus="Quercus"),
         forbid=("species", "locality", "collector"), top1=2),
    dict(query="specimens in the family Canidae",
         expect=dict(family="Canidae"),
         forbid=("genus", "species", "locality", "collector"), top1=4),
    dict(query="Rosaceae",
         expect=dict(family="Rosaceae"),
         forbid=("locality", "collector"), top1=1),

    # --- abbreviation: the case that motivated enrichment ------------------
    # Deliberately does NOT assert genus. "T rex" -> genus="Tyrannosaurus" is a
    # reasonable expansion and so is leaving it null; both should pass.
    dict(query="T rex", expect={},
         forbid=("locality", "collector", "family"), top1=8),

    # --- collector and locality --------------------------------------------
    dict(query="specimens Emily Davis collected in Georgia",
         expect=dict(collector="Emily Davis", locality="Georgia"),
         forbid=("genus", "species"), top1=2),
    dict(query="anything from Montana",
         expect=dict(locality="Montana"),
         forbid=("kingdom", "family", "genus", "species", "collector"), top1=7),
    dict(query="specimens from the UGA Wildlife Survey",
         expect=dict(collector="UGA Wildlife Survey"),
         forbid=("genus", "species"), top1=5),
    dict(query="collected by the Mineralogy Lab",
         expect=dict(collector="Mineralogy Lab"),
         forbid=("genus", "species"), top1=10),
    dict(query="what was found in Yellowstone",
         expect=dict(locality="Yellowstone"),
         forbid=("family", "genus", "species"), top1=4),

    # --- category: exists only because of enrichment ------------------------
    dict(query="show me fossils",
         expect=dict(category="fossil"),
         forbid=("family", "genus", "species", "locality", "collector"), top1=None),
    dict(query="dinosaur fossils",
         expect=dict(category="fossil"),
         forbid=("locality", "collector"), top1=None),
    dict(query="minerals",
         expect=dict(category="mineral"),
         forbid=("genus", "species", "locality", "collector"), top1=None),
    dict(query="minerals that look like gold",
         expect=dict(category="mineral"),
         forbid=("family", "genus", "species", "locality", "collector"), top1=11),

    # --- kingdom ------------------------------------------------------------
    dict(query="plants collected in Georgia",
         expect=dict(kingdom="Plantae", locality="Georgia"),
         forbid=("genus", "species", "collector"), top1=None),
    dict(query="trees",
         expect=dict(kingdom="Plantae"),
         forbid=("genus", "species", "locality", "collector"), top1=None),
    dict(query="animals from Georgia",
         expect=dict(kingdom="Animalia", locality="Georgia"),
         forbid=("genus", "species", "collector"), top1=None),

    # --- pure semantics: every structured field must stay null --------------
    # These are the queries embeddings exist for. A parser that invents ANY
    # filter here is actively harmful.
    dict(query="shiny gold rock", expect={}, forbid=NO_FILTERS, top1=11),
    dict(query="fool's gold", expect={}, forbid=NO_FILTERS, top1=11),
    dict(query="large predator", expect={}, forbid=NO_FILTERS, top1=4),
    dict(query="big meat-eating dinosaur", expect={}, forbid=NO_FILTERS, top1=8),
    dict(query="sea creature", expect={}, forbid=NO_FILTERS, top1=6),
    dict(query="sauropod", expect={}, forbid=("locality", "collector"), top1=12),
    dict(query="something with a shell", expect={}, forbid=NO_FILTERS, top1=None),

    # --- common names: enrichment is what makes these land ------------------
    dict(query="wild rose", expect={}, forbid=("locality", "collector"), top1=1),
    dict(query="white oak", expect={}, forbid=("locality", "collector"), top1=2),
    dict(query="pine tree", expect={}, forbid=("locality", "collector"), top1=3),
    dict(query="deer skull", expect={}, forbid=("locality", "collector"), top1=5),
    dict(query="green sea turtle", expect={}, forbid=("locality", "collector"), top1=6),
    dict(query="wolves", expect={}, forbid=("locality", "collector"), top1=4),
    dict(query="crystals", expect={},
         forbid=("family", "genus", "species", "locality", "collector"), top1=None),

    # --- nonsense guard ------------------------------------------------------
    # Should extract nothing rather than pattern-match its way into a filter.
    dict(query="asdfghjkl", expect={}, forbid=NO_FILTERS, top1=None),
]
