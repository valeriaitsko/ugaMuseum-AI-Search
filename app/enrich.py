"""Offline enrichment pass. Run before serving:

    python -m app.enrich                 # enrich every new/changed record (default model)
    python -m app.enrich --limit 200     # a random sample -- check quality before the full run

Compare models before committing the full (expensive) pass. Send each model's
sample to its own scratch file so they don't collide in the real enriched.json,
then read the files side by side:

    python -m app.enrich --limit 200 --model claude-haiku-4-5 --out app/data/enriched-haiku.json
    python -m app.enrich --limit 200 --model claude-sonnet-5  --out app/data/enriched-sonnet.json
    python -m app.enrich --limit 200 --model claude-opus-4-8  --out app/data/enriched-opus.json

(app/data/enriched-*.json is gitignored; the real enriched.json is not.)

Reads the specimen snapshot (app/specimens.load_specimens -- the Specify snapshot
if present, else the demo list), asks Claude for search aliases per specimen, and
writes app/data/enriched.json. Already-enriched, unchanged records are skipped, so
re-running after adding or editing records only pays for what changed.

--limit enriches a random sample of that many records. Random, not the first N,
because catalog ids are grouped by upload -- the first 200 insects could be all
moths and tell you nothing about how the model handles beetles or flies. Each
sampled run enriches that many *more* records (the ones already done drop out of
the todo list), so you can sample, eyeball the output, then run without --limit to
finish the rest -- never re-paying for a record.

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

import argparse
import hashlib
import json
import random
from pathlib import Path

import anthropic

from app.ai import ENRICH_MODEL, enrich_specimen
from app.specimens import load_specimens

DATA_DIR = Path(__file__).parent / "data"
ENRICHED_PATH = DATA_DIR / "enriched.json"

# Fields that AFFECT the enrichment output, hashed to decide when a record needs
# re-enriching. Not every field enrich_specimen sends: catalogNumber and
# altCatalogNumber are also sent but are identifiers that don't change the aliases,
# so a change to them shouldn't trigger a re-enrich. `order` DOES shape the output
# (it's the group context Claude uses to write "a fly", "a beetle"), so it belongs
# here. Adding it re-enriches the already-done records once, since their stored
# hash predates it -- a one-time, harmless cost.
ENRICHED_FIELDS = ("kingdom", "order", "family", "genus", "species", "locality")


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


def load_enriched(path: Path) -> dict[str, dict]:
    """Existing enrichments, keyed by stringified specimen id (JSON has no int keys)."""
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    ap = argparse.ArgumentParser(description="Enrich specimen records with Claude.")
    ap.add_argument("--limit", type=int,
                    help="enrich a random sample of this many new/changed records (quality check)")
    ap.add_argument("--model",
                    help=f"model to enrich with (default {ENRICH_MODEL}); "
                         f"e.g. claude-haiku-4-5, claude-sonnet-5")
    ap.add_argument("--out",
                    help="output file (default app/data/enriched.json). Point comparison "
                         "runs at a scratch file so they don't touch the real one.")
    args = ap.parse_args()

    model = args.model or ENRICH_MODEL
    out_path = Path(args.out) if args.out else ENRICHED_PATH

    DATA_DIR.mkdir(exist_ok=True)
    specimens = load_specimens()
    enriched = load_enriched(out_path)

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
    print(f"model: {model}   ->  {out_path}")
    print(f"{len(todo)} specimen(s) need enriching -- {new} new, {changed} changed "
          f"-- with {len(enriched)} cached.")

    if args.limit and args.limit < len(todo):
        # Random (seeded, so a run is reproducible) rather than the first N, so the
        # sample spans orders instead of whatever the id order groups together.
        todo = random.Random(0).sample(todo, args.limit)
        print(f"--limit: enriching a random sample of {len(todo)}. "
              f"Run again without --limit to finish the rest.")
    print()

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
            result = enrich_specimen(specimen, model=model)
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
        record["_model"] = model                          # provenance: which model wrote this
        enriched[str(specimen["id"])] = record

        aliases = ", ".join((result.abbreviations + result.common_names)[:3])
        print(f"  [{specimen['id']:>2}] {label:<24} {result.category:<8} {aliases}")

        # Write after every specimen. An interrupted run keeps its progress and
        # the next run resumes rather than re-paying for what already succeeded.
        out_path.write_text(json.dumps(enriched, indent=2), encoding="utf-8")

    if enriched:
        print(f"\nWrote {len(enriched)} enrichments to {out_path}")
    else:
        # The write only happens after a successful call, so claiming we "wrote 0"
        # to a file that was never created is a lie the next reader will chase.
        print("\nNothing succeeded. No file written.")


if __name__ == "__main__":
    main()
