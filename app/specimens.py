"""Source collection records.

In production these come from Specify7, pulled by app/specify_sync.py into
data/specimens.json. For the prototype it's the hand-written DEMO_SPECIMENS list
below, so the full pipeline (enrich -> embed -> search) runs with no real data
and no Specify7 server.

load_specimens() returns the synced file if it exists, else the demo list. That
one function is the entire seam between prototype and production -- enrich.py and
search.py both call it, so neither knows or cares where the records came from.

Nothing here is generated or inferred. These are the authoritative fields.
"""

import json
from pathlib import Path

SPECIMENS_JSON = Path(__file__).parent / "data" / "specimens.json"

DEMO_SPECIMENS = [
    {
        "id": 1,
        "scientificName": "Rosa carolina",
        "family": "Rosaceae",
        "kingdom": "Plantae",
        "locality": "Athens, Georgia",
        "collector": "John Smith",
        "description": "Wild rose collected in a deciduous forest.",
    },
    {
        "id": 2,
        "scientificName": "Quercus alba",
        "family": "Fagaceae",
        "kingdom": "Plantae",
        "locality": "Madison County, Georgia",
        "collector": "Emily Davis",
        "description": "White oak herbarium specimen.",
    },
    {
        "id": 3,
        "scientificName": "Pinus taeda",
        "family": "Pinaceae",
        "kingdom": "Plantae",
        "locality": "Oconee National Forest",
        "collector": "Michael Lee",
        "description": "Loblolly pine branch specimen.",
    },
    {
        "id": 4,
        "scientificName": "Canis lupus",
        "family": "Canidae",
        "kingdom": "Animalia",
        "locality": "Yellowstone National Park",
        "collector": "Museum Expedition 2018",
        "description": "Gray wolf skeletal specimen.",
    },
    {
        "id": 5,
        "scientificName": "Odocoileus virginianus",
        "family": "Cervidae",
        "kingdom": "Animalia",
        "locality": "North Georgia",
        "collector": "UGA Wildlife Survey",
        "description": "White-tailed deer skull.",
    },
    {
        "id": 6,
        "scientificName": "Chelonia mydas",
        "family": "Cheloniidae",
        "kingdom": "Animalia",
        "locality": "Georgia Coast",
        "collector": "Marine Research Group",
        "description": "Green sea turtle shell specimen.",
    },
    {
        "id": 7,
        "scientificName": "Triceratops horridus",
        "family": "Ceratopsidae",
        "kingdom": "Animalia",
        "locality": "Montana",
        "collector": "Paleontology Expedition",
        "description": "Partial fossil skull.",
    },
    {
        "id": 8,
        "scientificName": "Tyrannosaurus rex",
        "family": "Tyrannosauridae",
        "kingdom": "Animalia",
        "locality": "South Dakota",
        "collector": "Museum Fossil Team",
        "description": "Large theropod fossil.",
    },
    {
        "id": 9,
        "scientificName": "Ammonite sp.",
        "family": "Ammonitidae",
        "kingdom": "Animalia",
        "locality": "Texas",
        "collector": "Field Survey",
        "description": "Marine fossil preserved in limestone.",
    },
    {
        "id": 10,
        "scientificName": "Quartz",
        "family": "Silicate",
        "kingdom": "Mineral",
        "locality": "North Carolina",
        "collector": "Mineralogy Lab",
        "description": "Clear quartz crystal.",
    },
    {
        "id": 11,
        "scientificName": "Pyrite",
        "family": "Sulfide",
        "kingdom": "Mineral",
        "locality": "Colorado",
        "collector": "Rock Collection",
        "description": "Cubic pyrite crystals.",
    },
    {
        "id": 12,
        "scientificName": "Apatosaurus louisae",
        "family": "Diplodocidae",
        "kingdom": "Animalia",
        "locality": "Utah",
        "collector": "Dinosaur Expedition",
        "description": "Sauropod vertebra fossil.",
    },
]

# The exact vocabulary the catalog uses. The query parser is constrained to
# these values so it can't invent "Mineralia" for a record stored as "Mineral".
#
# WITH LIVE DATA: these must reflect what Specify7 actually holds. specify_sync.py
# prints the distinct kingdoms it observed on each run -- update this list AND the
# Kingdom = Literal[...] in app/ai.py to match, or the parser will emit kingdom
# values that filter out every record. The Literal is a compile-time constant;
# there's no way around editing it by hand when the vocabulary changes.
KINGDOMS = ["Plantae", "Animalia"]

# Not a catalog field. Assigned during enrichment because visitors search for
# "fossils", "dinosaurs", and "insects" -- groupings that aren't kingdoms.
# 'insect' is a subset of Animalia, split out because this collection is almost
# entirely insects and visitors think "bugs", not "animals".
CATEGORIES = ["plant", "animal", "insect", "mineral", "fossil"]


def load_specimens() -> list[dict]:
    """Synced Specify7 records if present, else the built-in demo list.

    The single seam between prototype and production. Everything downstream --
    enrichment, the embedding index, search -- goes through here, so swapping in
    real data is one sync run plus zero code changes.
    """
    if SPECIMENS_JSON.exists():
        return json.loads(SPECIMENS_JSON.read_text(encoding="utf-8"))
    return DEMO_SPECIMENS
