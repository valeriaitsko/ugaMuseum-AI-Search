"""Pull specimen records from a Specify7 server into data/specimens.json.

    python -m app.specify_sync                 # full sync
    python -m app.specify_sync --limit 20      # first 20, for testing the mapping
    python -m app.specify_sync --dry-run       # fetch + map, print, write nothing

This is the ONLY place Specify7 is contacted. Enrichment, the embedding index,
and every visitor search read the local data/specimens.json snapshot this writes
-- never the live server. So Specify7 being slow or down never touches search;
it only makes a sync stale. Run this on a schedule (nightly cron) or from a
Specify7 change webhook.

    Specify7  ->  fetch + map  ->  data/specimens.json  ->  enrich / index / search

WHAT'S REAL vs WHAT'S A SEAM
  Real (works as written): config, pagination, ret/backoff, atomic write, change
    reporting, the observed-kingdom check.
  Seam (YOU fill in against your instance): _authenticate() and _map_record().
    Specify7 records are deeply normalized -- a CollectionObject has no flat
    "locality" or "family"; you traverse determinations -> taxon -> parents, and
    collectingevent -> locality / collectors -> agent. The exact JSON keys depend
    on your collection's schema config, so guessing them would give you code that
    looks right and returns None for every record. The traversal is mapped out in
    _map_record() as comments -- follow them against your actual API responses.

Verify your instance's shapes by hitting the API in a browser first, e.g.
    {BASE_URL}/api/specify/collectionobject/?limit=1
and reading one full record before touching _map_record().
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

from app.enrich import content_hash
from app.specimens import KINGDOMS

load_dotenv()

BASE_URL = os.getenv("SPECIFY_BASE_URL", "").rstrip("/")
API_KEY = os.getenv("SPECIFY_API_KEY")
USERNAME = os.getenv("SPECIFY_USERNAME")
PASSWORD = os.getenv("SPECIFY_PASSWORD")
COLLECTION_ID = os.getenv("SPECIFY_COLLECTION_ID")

OUT_PATH = Path(__file__).parent / "data" / "specimens.json"

PAGE_SIZE = 100          # Specify7 (Tastypie) caps page size; 100 is safe.
TIMEOUT = 30             # seconds per HTTP request
MAX_RETRIES = 3

# Specify's default taxon-tree rankids. Ranks are identified by RANKID -- never by
# a node's name, and never by "topmost in the chain". That distinction is the
# whole point here: the tree root in this collection is a real node named
# "Uploaded" (rankid 0). Selecting family/genus/kingdom by rankid means that root
# is structurally excluded and can never leak into a specimen field.
ROOT_RANK = 0            # the tree root (here "Uploaded"); never a real rank
KINGDOM_RANK = 10
FAMILY_RANK = 140
GENUS_RANK = 180         # (kept for reference; genus is searched inside scientificName)
SPECIES_RANK = 220

# Taxon nodes are shared across specimens (every plant shares the same Plantae
# node, every record shares the "Uploaded" root). Cache fetched nodes for the
# life of one sync so a 12k-record collection doesn't re-fetch the root 12k times.
_taxon_cache: dict[str, dict] = {}


# --------------------------------------------------------------------------- #
# HTTP + auth
# --------------------------------------------------------------------------- #

def make_session() -> requests.Session:
    """A requests.Session with auth wired up. SEAM -- see _authenticate()."""
    if not BASE_URL:
        sys.exit("SPECIFY_BASE_URL is not set. Copy .env.example and fill in the Specify7 fields.")
    session = requests.Session()
    session.headers.update({"Accept": "application/json"})
    _authenticate(session)
    return session


def _authenticate(session: requests.Session) -> None:
    """SEAM: authenticate `session` against your Specify7 instance.

    Specify7 auth varies by deployment. The two common shapes -- confirm which
    yours uses by checking with your Specify admin or the API docs for your host:

    (A) Session login (most common). Specify7 is a Django app:
          1. GET  {BASE_URL}/context/login/                -> read the CSRF token
             (from the `csrftoken` cookie or the JSON body).
          2. POST {BASE_URL}/context/login/ with
                 {"username": USERNAME, "password": PASSWORD, "collection": COLLECTION_ID}
             and header  X-CSRFToken: <token>.
          3. The response sets a `sessionid` cookie; `session` carries it onward.
             Also send X-CSRFToken on later POSTs (GETs usually don't need it).
        You must select a collection (step 2's "collection" field) or record
        endpoints return nothing.

    (B) API key, if your instance issues them:
          session.headers["Authorization"] = f"Bearer {API_KEY}"   # or the header
          your host documents -- confirm the exact header name.

    Until you implement this, sync will fail on the first request with a 401/403.
    That failure is expected and tells you the endpoint is reachable.
    """
    if API_KEY:
        # Shape (B). Confirm the header name your instance expects.
        session.headers["Authorization"] = f"Bearer {API_KEY}"
        return

    # Shape (A) goes here. Left unimplemented on purpose -- see the docstring.
    raise NotImplementedError(
        "_authenticate() is a seam. Implement session or API-key auth for your "
        "Specify7 instance (see the docstring), or set SPECIFY_API_KEY."
    )


def _get(session: requests.Session, url: str, params: dict) -> dict:
    """GET with retry/backoff on transient failures. Raises on 4xx (your bug)."""
    for attempt in range(MAX_RETRIES):
        try:
            resp = session.get(url, params=params, timeout=TIMEOUT)
        except requests.RequestException as exc:
            if attempt == MAX_RETRIES - 1:
                raise
            _backoff(attempt, f"network error: {exc}")
            continue

        if resp.status_code < 400:
            return resp.json()
        if resp.status_code in (429, 500, 502, 503, 504) and attempt < MAX_RETRIES - 1:
            _backoff(attempt, f"HTTP {resp.status_code}")
            continue
        # 401/403/404 and the like are configuration/mapping bugs -- fail loudly.
        resp.raise_for_status()
    raise RuntimeError("unreachable")


def _backoff(attempt: int, reason: str) -> None:
    delay = 2 ** attempt
    print(f"  {reason}; retrying in {delay}s...", file=sys.stderr)
    time.sleep(delay)


# --------------------------------------------------------------------------- #
# Fetch
# --------------------------------------------------------------------------- #

def fetch_records(session: requests.Session, limit: int | None):
    """Yield raw CollectionObject records, paging through the Tastypie API.

    Specify7 responds with {"meta": {"total_count": N, ...}, "objects": [...]}.
    We page on limit/offset until we've seen total_count (or hit --limit).
    """
    endpoint = f"{BASE_URL}/api/specify/collectionobject/"
    offset = 0
    yielded = 0

    while True:
        params = {"limit": PAGE_SIZE, "offset": offset}
        if COLLECTION_ID:
            params["collection"] = COLLECTION_ID   # scope to one collection
        payload = _get(session, endpoint, params)

        objects = payload.get("objects", [])
        if not objects:
            break

        for raw in objects:
            yield raw
            yielded += 1
            if limit and yielded >= limit:
                return

        total = payload.get("meta", {}).get("total_count")
        offset += len(objects)
        if total is not None and offset >= total:
            break


# --------------------------------------------------------------------------- #
# Map: Specify7 record -> our flat specimen dict  (THE SEAM)
# --------------------------------------------------------------------------- #

def _lineage(session: requests.Session, taxon_uri: str) -> list[dict]:
    """Fetch a taxon and all its ancestors, nearest-first (the node, then its
    parent, ... up to the root). [] if taxon_uri is falsy.

    Follows the `parent` URI on each node until it's null (the root). Nodes are
    memoized in _taxon_cache, and a `seen` set guards against a malformed parent
    cycle turning this into an infinite loop.
    """
    nodes: list[dict] = []
    seen: set[str] = set()
    uri = taxon_uri
    while uri and uri not in seen:
        seen.add(uri)
        node = _taxon_cache.get(uri)
        if node is None:
            node = _get(session, f"{BASE_URL}{uri}", {})
            _taxon_cache[uri] = node
        nodes.append(node)
        uri = node.get("parent")   # null on the root -> loop ends
    return nodes


def _ranks_by_id(nodes: list[dict]) -> dict[int, str]:
    """Collapse a lineage into {rankid: name}, keeping the NEAREST node per rank
    and dropping the tree root (rankid 0 -- the "Uploaded" node).

    Selecting by rankid is exactly what keeps "Uploaded" out of genus/family:
    the root is skipped here, so ranks.get(GENUS_RANK) is a real genus or absent."""
    ranks: dict[int, str] = {}
    for node in nodes:
        rid = node.get("rankid")
        if rid is None or rid == ROOT_RANK:   # skip the "Uploaded" root explicitly
            continue
        ranks.setdefault(rid, (node.get("name") or "").strip())
    return ranks


def _current_determination(session: requests.Session, raw: dict) -> dict | None:
    """The determination flagged iscurrent, else the first, else None.

    Determinations are a dependent to-many on the CollectionObject, so Specify7
    usually inlines them as a list. If your instance returns a URI instead, we
    fetch it. Field casing is Specify's lowercase (`iscurrent`, like `isaccepted`
    on the taxon record)."""
    dets = raw.get("determinations")
    if isinstance(dets, str):                 # a URI -> fetch the collection
        dets = _get(session, f"{BASE_URL}{dets}", {}).get("objects", [])
    dets = dets or []
    for det in dets:
        if det.get("iscurrent"):
            return det
    return dets[0] if dets else None


def _map_record(session: requests.Session, raw: dict) -> dict | None:
    """Turn one Specify7 CollectionObject into our specimen dict, or None.

    Target shape (what specimens.py / search.py expect):
        {id, scientificName, family, kingdom, locality, collector, description}

    Returns None to SKIP a record with no current determination or no taxon --
    it has no scientific name, so it can't be searched by name and would only
    pollute the index.

    VERIFIED against this instance: the taxon walk (scientificName / family /
    kingdom). Read one real determination + collectingEvent record before
    trusting the locality/collector paths below -- they follow Specify's standard
    structure but their exact field names depend on your schema config.
    """
    det = _current_determination(session, raw)
    if det is None or not det.get("taxon"):
        return None                            # nothing determined -> skip

    nodes = _lineage(session, det["taxon"])
    if not nodes or nodes[0].get("rankid") in (None, ROOT_RANK):
        return None                            # determination points at the root itself

    ranks = _ranks_by_id(nodes)
    det_node = nodes[0]

    # scientificName = the determined node's own fullname ("Mentzelia Micrantha"
    # at species, "Mentzelia" at genus). genus/species are searched as words
    # inside this string, so no separate fields are needed.
    scientific = (det_node.get("fullname") or det_node.get("name") or "").strip()
    if not scientific:
        return None

    locality, collector = _locality_and_collector(session, raw)

    specimen = {
        "id": raw["id"],
        "scientificName": scientific,
        "family": ranks.get(FAMILY_RANK, ""),
        "kingdom": ranks.get(KINGDOM_RANK, ""),   # warned by _report_kingdoms if off-vocab
        "locality": locality,
        "collector": collector,
        "description": (raw.get("remarks") or "").strip(),
    }
    specimen["content_hash"] = content_hash(specimen)
    return specimen


def _locality_and_collector(session: requests.Session, raw: dict) -> tuple[str, str]:
    """(locality, collector) off the CollectingEvent. ("", "") when absent --
    common and fine (fossils, purchased or legacy specimens have no event).

    STANDARD Specify path, not yet verified against this instance:
        collectingEvent -> locality.localityname
        collectingEvent -> collectors[] -> agent (lastName/firstName or fullName)
    Confirm the field names against a real collectingevent/agent record.
    """
    ce_uri = raw.get("collectingevent")
    if not ce_uri:
        return "", ""
    ce = _get(session, f"{BASE_URL}{ce_uri}", {})

    locality = ""
    loc_uri = ce.get("locality")
    if loc_uri:
        loc = _get(session, f"{BASE_URL}{loc_uri}", {})
        locality = (loc.get("localityname") or "").strip()

    names = []
    for col in ce.get("collectors") or []:     # dependent to-many, usually inlined
        agent = col.get("agent")
        if isinstance(agent, str):             # a URI -> fetch it
            agent = _get(session, f"{BASE_URL}{agent}", {})
        if isinstance(agent, dict):
            names.append(_agent_name(agent))
    return locality, ", ".join(n for n in names if n)


def _agent_name(agent: dict) -> str:
    """An agent's display name: fullName if present, else 'First Last'."""
    full = (agent.get("fullname") or "").strip()
    if full:
        return full
    parts = [(agent.get("firstname") or "").strip(), (agent.get("lastname") or "").strip()]
    return " ".join(p for p in parts if p)


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #

def sync(limit: int | None, dry_run: bool) -> None:
    session = make_session()

    specimens, skipped = [], 0
    for raw in fetch_records(session, limit):
        mapped = _map_record(session, raw)
        if mapped is None:
            skipped += 1
            continue
        mapped.setdefault("content_hash", content_hash(mapped))
        specimens.append(mapped)

    print(f"\nmapped {len(specimens)} specimen(s), skipped {skipped}")
    _report_kingdoms(specimens)

    if dry_run:
        print("\n--dry-run: nothing written. First mapped record:")
        print(json.dumps(specimens[0], indent=2) if specimens else "  (none)")
        return

    _write_atomic(specimens)
    print(f"\nwrote {len(specimens)} records to {OUT_PATH}")
    print("next: `python -m app.enrich` (re-enriches new/changed records), then restart the server.")


def _report_kingdoms(specimens: list[dict]) -> None:
    """Warn if the data holds kingdoms the query parser doesn't know about.

    KINGDOMS (and the Kingdom Literal in app/ai.py) are a fixed vocabulary. A
    kingdom in the data but not in that list can never be produced as a filter,
    so those specimens are unreachable by kingdom. Surface it loudly -- this is
    the single most likely silent breakage when moving off the demo data.
    """
    observed = sorted({s["kingdom"] for s in specimens if s.get("kingdom")})
    print(f"kingdoms observed: {observed}")
    unknown = [k for k in observed if k not in KINGDOMS]
    if unknown:
        print(f"  WARNING: {unknown} not in specimens.KINGDOMS. Add them there AND to")
        print(f"           the Kingdom = Literal[...] in app/ai.py, or they won't filter.")


def _write_atomic(specimens: list[dict]) -> None:
    """Write via a temp file + replace, so a crash mid-write can't corrupt the
    snapshot the running server reads."""
    OUT_PATH.parent.mkdir(exist_ok=True)
    tmp = OUT_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(specimens, indent=2), encoding="utf-8")
    tmp.replace(OUT_PATH)


def main() -> None:
    ap = argparse.ArgumentParser(description="Sync specimens from Specify7.")
    ap.add_argument("--limit", type=int, help="fetch only the first N records (test the mapping)")
    ap.add_argument("--dry-run", action="store_true", help="fetch + map, print, write nothing")
    args = ap.parse_args()
    sync(limit=args.limit, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
