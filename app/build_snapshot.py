"""Build data/specimens.json from a local Specify 7 database.

    python -m app.build_snapshot                # full build
    python -m app.build_snapshot --limit 5      # first 5 records, to eyeball the mapping
    python -m app.build_snapshot --dry-run      # build + print the first record, write nothing

Reads the MySQL/MariaDB that backs Specify 7 DIRECTLY (not the live REST API),
flattens each CollectionObject into the lean record the search app expects, and
writes a static snapshot. The app only ever reads that snapshot -- see
app/specimens.py -- so Specify being slow or down never touches search; it only
makes a snapshot stale. Commit specimens.json to the repo; don't regenerate at
deploy time.

    Specify DB  ->  flatten  ->  data/specimens.json  ->  enrich / index / search

Record shape (exactly these keys, values null when absent):
    {id, catalogNumber, altCatalogNumber, kingdom, order, family, genus, species, locality}

WHERE EACH FIELD COMES FROM
  id / catalogNumber / altCatalogNumber   directly on CollectionObject.
  kingdom / family / genus / species      the CURRENT determination only
        (Determination.IsCurrent = 1) -> its Taxon, then walk UP the Taxon tree by
        ParentID, reading each ancestor's RankID. No current determination -> all
        four are null (we do NOT fall back to a superseded identification).
  locality                                CollectionObject -> CollectingEvent ->
        Locality.LocalityName.

Fields are selected by RankID, never by a node's name or its position in the tree.
That is what keeps the tree root (RankID 0, named "Uploaded" in this collection --
a WorkBench import artifact, not a rank) out of the taxonomy fields.

Only standard, visible columns are read (identifiers, taxonomy, locality); none are
schema-hidden, so schema_localization's "ishidden" flags don't come into play here.

Connection config comes from this app's own .env (SPECIFY_DB_*), so the build stays
self-contained and doesn't reach into Specify's settings.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import pymysql
from dotenv import load_dotenv

load_dotenv()

DB_HOST = os.getenv("SPECIFY_DB_HOST", "127.0.0.1")
DB_PORT = int(os.getenv("SPECIFY_DB_PORT", "3306"))
DB_NAME = os.getenv("SPECIFY_DB_NAME", "SpecifyDB")
DB_USER = os.getenv("SPECIFY_DB_USER", "")
DB_PASSWORD = os.getenv("SPECIFY_DB_PASSWORD", "")

OUT_PATH = Path(__file__).parent / "data" / "specimens.json"

# Taxonomy is read by rank NAME, not RankID, because this database holds multiple
# taxon trees that REUSE the same RankIDs for different ranks: rank 10 is "Kingdom"
# in the biology tree but "Points" in the archaeology tree; rank 30 is "Phylum" vs
# "Specimen". Keying by the (unambiguous) rank name lets one flatten pass handle
# both disciplines. Only the RankID-0 root (named "Life"/"Root") is dropped by id.
ROOT_RANK = 0

# rank name in the tree -> field in the flattened record. Biology is the Linnaean
# hierarchy; archaeology is Cultural Period -> Points -> Specimen (projectile-point
# typology: a named point type like "Dalton", grouped by base/notch form, within a
# time period). A record fills exactly one of these two sets.
BIOLOGY_RANKS = {
    "Kingdom": "kingdom", "Phylum": "phylum", "Class": "class", "Order": "order",
    "Family": "family", "Genus": "genus", "Species": "species",
}
ARCHAEOLOGY_RANKS = {
    "Cultural Period": "culturalPeriod", "Points": "points", "Specimen": "specimen",
}


def connect() -> "pymysql.connections.Connection":
    if not DB_USER:
        sys.exit("SPECIFY_DB_USER is not set. Copy .env.example and fill in the SPECIFY_DB_* fields.")
    return pymysql.connect(
        host=DB_HOST,
        port=DB_PORT,
        user=DB_USER,
        password=DB_PASSWORD,
        database=DB_NAME,
        charset="utf8mb4",
        cursorclass=pymysql.cursors.DictCursor,
    )


def load_taxa(cur) -> dict[int, dict]:
    """The whole taxon tree in memory, keyed by TaxonID, each node carrying its rank
    NAME (joined from taxontreedefitem) so we can tell "Kingdom" from "Points" even
    though both are RankID 10 in different trees. One bulk read; every specimen
    walks it, so this beats a recursive query per record."""
    cur.execute(
        """
        SELECT t.TaxonID, t.Name, t.RankID, t.ParentID, tdi.Name AS rankName
        FROM taxon t
        LEFT JOIN taxontreedefitem tdi ON tdi.TaxonTreeDefItemID = t.TaxonTreeDefItemID
        """
    )
    return {row["TaxonID"]: row for row in cur.fetchall()}


def ranks_for(taxon_id: int | None, taxa: dict[int, dict]) -> dict[str, str]:
    """Walk a taxon up its parents, returning {rank name: taxon name}, nearest node
    per rank, excluding the RankID-0 root. {} when taxon_id is missing.

    A `seen` set guards against a malformed parent cycle looping forever.
    """
    ranks: dict[str, str] = {}
    seen: set[int] = set()
    tid = taxon_id
    while tid is not None and tid in taxa and tid not in seen:
        seen.add(tid)
        node = taxa[tid]
        rank = (node["rankName"] or "").strip()
        if node["RankID"] != ROOT_RANK and rank:
            name = (node["Name"] or "").strip()
            if rank not in ranks and name:
                ranks[rank] = name
        tid = node["ParentID"]
    return ranks


def fetch_rows(cur) -> list[dict]:
    """One flat row per CollectionObject: identifiers + current-determination
    TaxonID + locality name. Taxonomy is resolved from TaxonID in Python."""
    cur.execute(
        """
        SELECT co.CollectionObjectID AS id,
               co.CatalogNumber      AS catalogNumber,
               co.AltCatalogNumber   AS altCatalogNumber,
               d.TaxonID             AS taxonId,
               loc.LocalityName      AS locality
        FROM collectionobject co
        LEFT JOIN determination d
               ON d.CollectionObjectID = co.CollectionObjectID AND d.IsCurrent = 1
        LEFT JOIN collectingevent ce ON ce.CollectingEventID = co.CollectingEventID
        LEFT JOIN locality        loc ON loc.LocalityID = ce.LocalityID
        ORDER BY co.CollectionObjectID
        """
    )
    return cur.fetchall()


def _clean(value) -> str | None:
    """Trim to a non-empty string, or None. Keeps empty cells from becoming "".", so
    downstream `field or ""` guards behave and the JSON stays honest."""
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def flatten(rows: list[dict], taxa: dict[int, dict], limit: int | None):
    specimens: list[dict] = []
    seen_ids: set[int] = set()
    no_current_det = 0

    for row in rows:
        cid = row["id"]
        if cid in seen_ids:
            # A CollectionObject should have at most one current determination, but
            # if the data has two, the join yields duplicate rows -- keep the first.
            continue
        seen_ids.add(cid)

        if row["taxonId"] is None:
            no_current_det += 1
        ranks = ranks_for(row["taxonId"], taxa)

        bio = {field: ranks.get(rank) for rank, field in BIOLOGY_RANKS.items()}
        arch = {field: ranks.get(rank) for rank, field in ARCHAEOLOGY_RANKS.items()}
        # Which tree the record's taxonomy came from -- the one clean signal the
        # frontend uses to decide how to label it (a point type, not an insect).
        if any(arch.values()):
            discipline = "archaeology"
        elif any(bio.values()):
            discipline = "biology"
        else:
            discipline = None

        specimens.append(
            {
                "id": cid,
                "catalogNumber": _clean(row["catalogNumber"]),
                "altCatalogNumber": _clean(row["altCatalogNumber"]),
                "discipline": discipline,
                **bio,
                **arch,
                "locality": _clean(row["locality"]),
            }
        )
        if limit and len(specimens) >= limit:
            break

    return specimens, no_current_det


def build(limit: int | None, dry_run: bool) -> None:
    conn = connect()
    try:
        with conn.cursor() as cur:
            taxa = load_taxa(cur)
            rows = fetch_rows(cur)
    finally:
        conn.close()

    specimens, no_current_det = flatten(rows, taxa, limit)
    _report(specimens, no_current_det)

    if dry_run:
        print("\n--dry-run: nothing written. First record:")
        print(json.dumps(specimens[0], indent=2) if specimens else "  (none)")
        return

    _write_atomic(specimens)
    print(f"\nwrote {len(specimens)} records to {OUT_PATH}")
    print("next: `python -m app.enrich` (re-enriches new/changed records), then restart the server.")


def _report(specimens: list[dict], no_current_det: int) -> None:
    """Surface the two things most likely to be silently wrong after a build:
    records with no taxonomy, and kingdoms the query parser can't emit."""
    from collections import Counter

    from app.specimens import KINGDOMS

    disciplines = Counter(s.get("discipline") or "(unidentified)" for s in specimens)
    kingdoms = sorted({s["kingdom"] for s in specimens if s["kingdom"]})
    print(f"built {len(specimens)} specimen(s); {no_current_det} had no current determination (taxonomy null)")
    print(f"disciplines: {dict(disciplines)}")
    print(f"kingdoms observed: {kingdoms}")
    unknown = [k for k in kingdoms if k not in KINGDOMS]
    if unknown:
        print(f"  WARNING: {unknown} not in specimens.KINGDOMS. Add them there AND to")
        print(f"           the Kingdom = Literal[...] in app/ai.py, or they won't filter.")


def _write_atomic(specimens: list[dict]) -> None:
    """Temp file + replace, so a crash mid-write can't corrupt the snapshot the
    running server reads."""
    OUT_PATH.parent.mkdir(exist_ok=True)
    tmp = OUT_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(specimens, indent=2), encoding="utf-8")
    tmp.replace(OUT_PATH)


def main() -> None:
    ap = argparse.ArgumentParser(description="Build specimens.json from the local Specify 7 database.")
    ap.add_argument("--limit", type=int, help="build only the first N records (test the mapping)")
    ap.add_argument("--dry-run", action="store_true", help="build + print the first record, write nothing")
    args = ap.parse_args()
    build(limit=args.limit, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
