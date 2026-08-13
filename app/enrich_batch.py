"""Enrich the whole collection via the Batches API -- half the price of the
synchronous app.enrich, for the one-time full run.

Fire-and-forget flow (recommended for the full ~31k run):

    python -m app.enrich_batch submit        # send the job, save its id, exit
    python -m app.enrich_batch collect        # collect when ready; re-run until "Done"

`submit` returns immediately -- the batch runs on Anthropic's side for minutes to
hours. `collect` advances a small state machine (collect primary -> submit the
fallback batch on refusals -> collect fallback) and is safe to run as many times
as you like; each run does what it can and tells you whether to come back. The
batch id lives in a state file, so closing your terminal or a dropped connection
loses nothing.

One-shot flow (good for a small --limit test):

    python -m app.enrich_batch run --limit 5  # submit + poll + collect, blocking

Batches trade immediacy for a 50% discount: enrichment is offline with nobody
waiting, so that's free -- and it sidesteps the burst-refusal fragility of the
synchronous loop (Anthropic paces the batch server-side).

Refusals compose as a second batch: specimens the primary (Opus) safety classifier
declines (pest species like the fall armyworm) are resubmitted on the fallback
(Sonnet). Only records both models refuse are left un-enriched.
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request

from app.ai import ENRICH_MODEL, ENRICH_SYSTEM, Enrichment, get_client
from app.enrich import DATA_DIR, ENRICHED_PATH, FALLBACK_MODEL, content_hash, load_enriched
from app.specimens import load_specimens

POLL_SECONDS = 30
STATE_PATH = DATA_DIR / "enrich_batch_state.json"


# --------------------------------------------------------------------------- #
# Request construction (shared)
# --------------------------------------------------------------------------- #

def enrichment_schema() -> dict:
    """The Enrichment model as a strict JSON schema for output_config.format.

    Batches don't offer the messages.parse `output_format=` convenience, so we
    derive the schema from the Pydantic model and set additionalProperties: false
    (required for strict structured output). The trimmed field descriptions ride
    along, same as the synchronous path.
    """
    schema = Enrichment.model_json_schema()
    schema["additionalProperties"] = False
    return schema


def build_request(specimen: dict, model: str) -> Request:
    """One batch request for one specimen. custom_id is the specimen id, so results
    (which come back in any order) map straight back to the record."""
    fields = "\n".join(f"{k}: {v}" for k, v in specimen.items() if k != "id")
    return Request(
        custom_id=str(specimen["id"]),
        params=MessageCreateParamsNonStreaming(
            model=model,
            max_tokens=4096,
            system=ENRICH_SYSTEM,
            messages=[{"role": "user", "content": fields}],
            output_config={"format": {"type": "json_schema", "schema": enrichment_schema()}},
        ),
    )


def submit_batch(specimens: list[dict], model: str) -> str:
    batch = get_client().messages.batches.create(
        requests=[build_request(s, model) for s in specimens]
    )
    print(f"  submitted batch {batch.id} ({len(specimens)} requests on {model})")
    return batch.id


def batch_ended(batch_id: str) -> bool:
    """One status check (no polling loop). Prints progress and returns whether done."""
    b = get_client().messages.batches.retrieve(batch_id)
    c = b.request_counts
    print(f"  {batch_id}: {b.processing_status} -- "
          f"{c.succeeded} succeeded, {c.processing} processing, {c.errored} errored")
    return b.processing_status == "ended"


def wait_for(batch_id: str) -> None:
    """Blocking poll until ended (used only by the one-shot `run` mode)."""
    while not batch_ended(batch_id):
        time.sleep(POLL_SECONDS)


def collect_results(batch_id: str) -> tuple[dict[str, dict], list[str], list[str]]:
    """Read a finished batch. Returns (enrichments, refused_ids, errored_ids).

    A 'succeeded' result whose message stop_reason is 'refusal' is a safety decline,
    routed to `refused` for the fallback batch -- not an enrichment.
    """
    client = get_client()
    enrichments: dict[str, dict] = {}
    refused: list[str] = []
    errored: list[str] = []
    for result in client.messages.batches.results(batch_id):
        sid = result.custom_id
        if result.result.type != "succeeded":
            errored.append(sid)
            continue
        msg = result.result.message
        if msg.stop_reason == "refusal":
            refused.append(sid)
            continue
        text = next((b.text for b in msg.content if b.type == "text"), None)
        if text is None:
            errored.append(sid)
            continue
        try:
            enrichments[sid] = Enrichment.model_validate_json(text).model_dump()
        except Exception:
            errored.append(sid)
    return enrichments, refused, errored


def write_records(enriched: dict, new: dict[str, dict], by_id: dict, model: str, out_path: Path) -> None:
    """Merge a batch's enrichments into the on-disk file, stamping provenance."""
    for sid, record in new.items():
        record["_content_hash"] = content_hash(by_id[sid])
        record["_model"] = model
        enriched[sid] = record
    out_path.write_text(json.dumps(enriched, indent=2), encoding="utf-8")


# --------------------------------------------------------------------------- #
# State (so submit/collect survive across separate invocations)
# --------------------------------------------------------------------------- #

def save_state(state: dict) -> None:
    STATE_PATH.write_text(json.dumps(state, indent=2), encoding="utf-8")


def load_state() -> dict | None:
    return json.loads(STATE_PATH.read_text(encoding="utf-8")) if STATE_PATH.exists() else None


def compute_todo(specimens: list[dict], enriched: dict, limit: int | None) -> list[dict]:
    """Records that are new or whose content changed since last enrichment."""
    todo = [s for s in specimens if (c := enriched.get(str(s["id"]))) is None
            or c.get("_content_hash") != content_hash(s)]
    if limit and limit < len(todo):
        todo = random.Random(0).sample(todo, limit)
    return todo


# --------------------------------------------------------------------------- #
# Commands
# --------------------------------------------------------------------------- #

def cmd_submit(args) -> None:
    model = args.model or ENRICH_MODEL
    fallback = None if args.fallback.strip().lower() in ("none", "") else args.fallback
    out_path = Path(args.out) if args.out else ENRICHED_PATH
    DATA_DIR.mkdir(exist_ok=True)

    if load_state():
        sys.exit("A batch is already in progress (see enrich_batch_state.json). Run "
                 "`collect` to finish it, or delete that file to start over.")

    specimens = load_specimens()
    todo = compute_todo(specimens, load_enriched(out_path), args.limit)
    if not todo:
        print("Nothing to enrich -- all records already enriched and unchanged.")
        return

    print(f"submitting {len(todo)} record(s): primary {model}, fallback {fallback or 'none'} -> {out_path}")
    batch_id = submit_batch(todo, model)
    save_state({
        "out": str(out_path),
        "primary_model": model, "primary_batch": batch_id, "primary_done": False,
        "fallback_model": fallback, "fallback_batch": None, "fallback_done": False,
        "refused": [], "errored": [],
    })
    print(f"\nSubmitted. The batch runs on Anthropic's side (minutes to hours).")
    print(f"Run `python -m app.enrich_batch collect` when you want results -- re-run it "
          f"until it says Done. Nothing is lost if you close this terminal.")


def cmd_collect(args) -> None:
    state = load_state()
    if not state:
        sys.exit("No batch in progress. Run `submit` first.")

    out_path = Path(state["out"])
    enriched = load_enriched(out_path)
    by_id = {str(s["id"]): s for s in load_specimens()}

    # Phase 1: primary batch
    if not state["primary_done"]:
        if not batch_ended(state["primary_batch"]):
            print("Primary batch still processing -- run `collect` again later.")
            return
        got, refused, errored = collect_results(state["primary_batch"])
        write_records(enriched, got, by_id, state["primary_model"], out_path)
        state.update(primary_done=True, refused=refused, errored=errored)
        save_state(state)
        print(f"  primary: {len(got)} enriched, {len(refused)} refused, {len(errored)} errored")

    fallback = state.get("fallback_model")
    refused = state.get("refused", [])

    # Phase 2: submit fallback on the refusals (once)
    if fallback and refused and not state.get("fallback_batch"):
        fb = [by_id[sid] for sid in refused if sid in by_id]
        print(f"submitting fallback batch for {len(fb)} refused specimen(s) on {fallback}")
        state["fallback_batch"] = submit_batch(fb, fallback)
        save_state(state)
        print("Fallback submitted -- run `collect` again to finish.")
        return

    # Phase 3: collect fallback
    if state.get("fallback_batch") and not state["fallback_done"]:
        if not batch_ended(state["fallback_batch"]):
            print("Fallback batch still processing -- run `collect` again later.")
            return
        fb_got, still_refused, fb_err = collect_results(state["fallback_batch"])
        write_records(enriched, fb_got, by_id, state["fallback_model"], out_path)
        state.update(fallback_done=True, still_refused=still_refused,
                     errored=state.get("errored", []) + fb_err)
        save_state(state)
        print(f"  fallback: rescued {len(fb_got)} of {len(refused)} refused")

    _finish(state, enriched, out_path)


def _finish(state: dict, enriched: dict, out_path: Path) -> None:
    """Print the final summary and clear the state so the next submit starts fresh."""
    still_refused = state.get("still_refused") or (state.get("refused") if not state.get("fallback_model") else [])
    errored = state.get("errored", [])
    print(f"\nDone. {len(enriched)} total enrichments in {out_path}.")
    if still_refused:
        print(f"  {len(still_refused)} declined by all models -- un-enriched (findable by Latin name).")
    if errored:
        print(f"  {len(errored)} errored/expired -- run `submit` again to retry just those "
              f"(change detection skips everything already done).")
    STATE_PATH.unlink(missing_ok=True)


def cmd_run(args) -> None:
    """One-shot blocking flow: submit + poll + collect, incl. fallback. For --limit tests."""
    model = args.model or ENRICH_MODEL
    fallback = None if args.fallback.strip().lower() in ("none", "") else args.fallback
    out_path = Path(args.out) if args.out else ENRICHED_PATH
    DATA_DIR.mkdir(exist_ok=True)

    specimens = load_specimens()
    by_id = {str(s["id"]): s for s in specimens}
    enriched = load_enriched(out_path)
    todo = compute_todo(specimens, enriched, args.limit)
    if not todo:
        print("Nothing to enrich -- all records already enriched and unchanged.")
        return

    print(f"{len(todo)} record(s): primary {model}, fallback {fallback or 'none'} -> {out_path}")
    print(f"\nPRIMARY: {len(todo)} on {model}")
    pid = submit_batch(todo, model)
    wait_for(pid)
    got, refused, errored = collect_results(pid)
    write_records(enriched, got, by_id, model, out_path)
    print(f"  primary: {len(got)} enriched, {len(refused)} refused, {len(errored)} errored")

    still_refused, rescued = refused, 0
    if refused and fallback:
        fb = [by_id[sid] for sid in refused if sid in by_id]
        print(f"\nFALLBACK: {len(fb)} on {fallback}")
        fid = submit_batch(fb, fallback)
        wait_for(fid)
        fb_got, still_refused, fb_err = collect_results(fid)
        write_records(enriched, fb_got, by_id, fallback, out_path)
        rescued, errored = len(fb_got), errored + fb_err
        print(f"  fallback: rescued {rescued} of {len(refused)}")

    print(f"\nDone. {len(enriched)} total enrichments in {out_path}.")
    if still_refused:
        print(f"  {len(still_refused)} declined by all models -- un-enriched.")
    if errored:
        print(f"  {len(errored)} errored/expired -- re-run to retry them.")


def main() -> None:
    try:
        sys.stdout.reconfigure(line_buffering=True)  # flush every line -- ids visible immediately
    except Exception:
        pass

    ap = argparse.ArgumentParser(description="Enrich the collection via the Batches API (half price).")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def add_common(p):
        p.add_argument("--limit", type=int, help="enrich a random sample of N (test)")
        p.add_argument("--model", help=f"primary model (default {ENRICH_MODEL})")
        p.add_argument("--fallback", default=FALLBACK_MODEL,
                       help=f"model for refused specimens (default {FALLBACK_MODEL}); 'none' to disable")
        p.add_argument("--out", help="output file (default app/data/enriched.json)")

    add_common(sub.add_parser("submit", help="submit the batch and exit; collect later"))
    add_common(sub.add_parser("run", help="submit + poll + collect in one (blocking; good for --limit tests)"))
    sub.add_parser("collect", help="collect an in-progress batch; re-run until Done")

    args = ap.parse_args()
    {"submit": cmd_submit, "collect": cmd_collect, "run": cmd_run}[args.cmd](args)


if __name__ == "__main__":
    main()
