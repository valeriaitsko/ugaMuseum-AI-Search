"""Offline enrichment pass. Run once, before serving:

    python -m app.enrich

Reads DEMO_SPECIMENS, asks Claude for search aliases per specimen, writes
app/data/enriched.json. Already-enriched specimens are skipped, so re-running
after adding records only pays for the new ones.

Why offline instead of at query time: enrichment cost scales with the size of
the collection (once), not with traffic. A 50,000-specimen catalog is one batch
job, and every search afterwards is a vector lookup. Enriching per request would
put an LLM call on the hot path for zero added recall.

At real collection size, swap the loop below for the Batches API
(client.messages.batches.create) -- same prompt, 50% of the cost, results within
the hour.

PROVENANCE: everything this script writes is model-generated. It exists to make
records findable and must never be rendered to a visitor as though the museum
asserted it. The API only ever returns the original catalog fields.
"""

import hashlib
import json
from pathlib import Path

import anthropic

from app.ai import enrich_specimen
from app.specimens import load_specimens

DATA_DIR = Path(__file__).parent / "data"
ENRICHED_PATH = DATA_DIR / "enriched.json"

# Fields sent to Claude for enrichment. The hash of these is what decides whether
# a specimen needs re-enriching -- change any of them and the aliases are stale.
ENRICHED_FIELDS = ("kingdom", "family", "genus", "species", "locality")


def content_hash(specimen: dict) -> str:
    """Fingerprint of the fields that feed enrichment.

    Enrichment is keyed by specimen id, but a curator editing a record in Specify7
    keeps the id and changes the content. Keying re-enrichment on this hash -- not
    the id -- means an edited species name triggers a fresh enrichment, while an
    untouched record is skipped and costs nothing. Computed here, from exactly the
    fields we send to Claude, so it stays correct no matter where the record came
    from (demo list or synced catalog).
    """
    payload = "\x00".join(str(specimen.get(f, "")) for f in ENRICHED_FIELDS)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_enriched() -> dict[str, dict]:
    """Existing enrichments, keyed by stringified specimen id (JSON has no int keys)."""
    if not ENRICHED_PATH.exists():
        return {}
    return json.loads(ENRICHED_PATH.read_text(encoding="utf-8"))


def main() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    specimens = load_specimens()
    enriched = load_enriched()

    # Re-enrich when new (no cache entry) or changed (content hash differs from
    # what we enriched against last time). Unchanged records are skipped for free.
    def needs_enriching(s: dict) -> bool:
        cached = enriched.get(str(s["id"]))
        return cached is None or cached.get("_content_hash") != content_hash(s)

    todo = [s for s in specimens if needs_enriching(s)]
    if not todo:
        print(f"All {len(specimens)} specimens already enriched and unchanged. Nothing to do.")
        return

    new = sum(1 for s in todo if str(s["id"]) not in enriched)
    changed = len(todo) - new
    print(f"Enriching {len(todo)} specimen(s) -- {new} new, {changed} changed "
          f"-- with {len(enriched)} cached.\n")

    for specimen in todo:
        # A short human label for progress output. The flattened record has no
        # scientificName -- build one from genus + species, falling back to a
        # catalog number or the id.
        label = (
            " ".join(p for p in (specimen.get("genus"), specimen.get("species")) if p)
            or specimen.get("altCatalogNumber")
            or specimen.get("catalogNumber")
            or f"id {specimen['id']}"
        )
        try:
            result = enrich_specimen(specimen)
        except anthropic.BadRequestError as exc:
            # 400 means the request itself is unacceptable -- a bad schema, or an
            # account that can't be billed. Every remaining specimen sends the
            # same shape to the same account, so they will all fail identically.
            # Skipping and continuing just burns 11 more round trips.
            print(f"  [{specimen['id']:>2}] {label}: {exc.message}")
            print("\nA 400 applies to every specimen, not just this one. Stopping.")
            raise SystemExit(1)
        except anthropic.APIStatusError as exc:
            # 429 and 5xx are transient (and the SDK already retried). Skip this
            # specimen; the next run picks it up from the cache.
            print(f"  [{specimen['id']:>2}] {label}: API error {exc.status_code} -- skipped")
            continue
        except anthropic.APIConnectionError:
            print(f"  [{specimen['id']:>2}] {label}: network error -- skipped")
            continue

        record = result.model_dump()
        record["_content_hash"] = content_hash(specimen)  # so we can detect edits next run
        enriched[str(specimen["id"])] = record

        aliases = ", ".join((result.abbreviations + result.common_names)[:3])
        print(f"  [{specimen['id']:>2}] {label:<24} {result.category:<8} {aliases}")

        # Write after every specimen. An interrupted run keeps its progress and
        # the next run resumes rather than re-paying for what already succeeded.
        ENRICHED_PATH.write_text(json.dumps(enriched, indent=2), encoding="utf-8")

    if enriched:
        print(f"\nWrote {len(enriched)} enrichments to {ENRICHED_PATH}")
    else:
        # The write only happens after a successful call, so claiming we "wrote 0"
        # to a file that was never created is a lie the next reader will chase.
        print("\nNothing succeeded. No file written.")


if __name__ == "__main__":
    main()
