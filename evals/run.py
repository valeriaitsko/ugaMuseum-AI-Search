"""Compare query-parser models on the same set of visitor queries.

    python -m evals.run                       # opus vs haiku
    python -m evals.run --models claude-haiku-4-5
    python -m evals.run --json evals/out.json

Measures three things, in descending order of how much they should worry you:

  hallucinated   A filter invented from nothing. "T rex" -> locality="South Dakota"
                 silently excludes every other T. rex in the collection. A wrong
                 filter is worse than no filter, because no filter degrades to
                 semantic search while a wrong one excludes.

  missed         A filter the query plainly stated and the parser dropped. Costs
                 precision; the semantic half usually still finds the specimen.

  top1           Did the right specimen rank first, end to end. This exercises
                 enrichment and embeddings too, not just the parser -- so a low
                 score here with a clean parse means the problem is downstream.

Costs real money: len(CASES) calls per model. Nothing is cached.
"""

import argparse
import json
import sys
import time
from dataclasses import dataclass, field

import anthropic

from app.ai import ENRICH_MODEL, QUERY_MODEL, parse_search_query_with_usage
from app.search import SearchIndex
from evals.cases import ALL_FIELDS, CASES, CATEGORICAL_FIELDS

# $ per million tokens (input, output). Keep in sync with the pricing page.
PRICES = {
    "claude-opus-4-8": (5.00, 25.00),
    "claude-sonnet-5": (3.00, 15.00),
    "claude-haiku-4-5": (1.00, 5.00),
}


def field_matches(name: str, expected: str, actual) -> bool:
    """Categorical fields compare exactly; free-text compares by containment.

    The catalog stores "Madison County, Georgia". A parser that extracts
    "Georgia" from "collected in Georgia" is correct, not wrong.
    """
    if actual is None:
        return False
    if name in CATEGORICAL_FIELDS:
        return str(actual).lower() == expected.lower()
    return expected.lower() in str(actual).lower()


@dataclass
class ModelReport:
    model: str
    expected_total: int = 0
    expected_hit: int = 0
    top1_total: int = 0
    top1_hit: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    seconds: float = 0.0
    hallucinations: list[str] = field(default_factory=list)
    misses: list[str] = field(default_factory=list)
    top1_failures: list[str] = field(default_factory=list)
    parsed: dict[str, dict] = field(default_factory=dict)

    @property
    def cost(self) -> float:
        pin, pout = PRICES.get(self.model, (0.0, 0.0))
        return self.input_tokens / 1e6 * pin + self.output_tokens / 1e6 * pout


def run_model(model: str, index: SearchIndex, cases: list[dict]) -> ModelReport:
    report = ModelReport(model=model)
    started = time.monotonic()

    for case in cases:
        query = case["query"]
        try:
            filters, usage = parse_search_query_with_usage(query, model=model)
        except anthropic.BadRequestError as exc:
            # Same reasoning as app/enrich.py: a 400 is a property of the request
            # or the account, not of this one query. Every remaining case will
            # fail identically. Don't spend 31 more round trips proving it.
            print(f"\n  400 on {query!r}: {exc.message}")
            print("  A 400 applies to every case. Stopping.\n")
            raise SystemExit(1)

        report.input_tokens += usage.input_tokens
        report.output_tokens += usage.output_tokens
        report.parsed[query] = {f: getattr(filters, f) for f in ALL_FIELDS}

        for name, expected in case["expect"].items():
            report.expected_total += 1
            if field_matches(name, expected, getattr(filters, name)):
                report.expected_hit += 1
            else:
                got = getattr(filters, name)
                report.misses.append(f"{query!r}: {name} expected {expected!r}, got {got!r}")

        for name in case["forbid"]:
            got = getattr(filters, name)
            if got is not None:
                report.hallucinations.append(f"{query!r}: invented {name}={got!r}")

        if case["top1"] is not None:
            report.top1_total += 1
            results = index.search(filters)
            actual = results[0]["id"] if results else None
            if actual == case["top1"]:
                report.top1_hit += 1
            else:
                got = results[0]["scientificName"] if results else "(nothing)"
                want = next(s["scientificName"] for s in index.specimens if s["id"] == case["top1"])
                report.top1_failures.append(f"{query!r}: wanted {want}, got {got}")

    report.seconds = time.monotonic() - started
    return report


def pct(hit: int, total: int) -> str:
    return f"{hit}/{total} ({100 * hit / total:.0f}%)" if total else "n/a"


def print_report(reports: list[ModelReport], enriched: bool) -> None:
    print("\n" + "=" * 78)
    print(f"{'model':<20} {'extracted':>14} {'hallucinated':>14} {'top-1':>14} {'cost':>9}")
    print("-" * 78)
    for r in reports:
        print(f"{r.model:<20} {pct(r.expected_hit, r.expected_total):>14} "
              f"{len(r.hallucinations):>14} {pct(r.top1_hit, r.top1_total):>14} "
              f"${r.cost:>8.4f}")
    print("=" * 78)

    for r in reports:
        print(f"\n{r.model}  ({r.seconds:.1f}s, "
              f"{r.input_tokens:,} in / {r.output_tokens:,} out, "
              f"${r.cost / len(CASES):.5f}/query)")
        for label, items in (("HALLUCINATED", r.hallucinations),
                             ("MISSED", r.misses),
                             ("TOP-1 WRONG", r.top1_failures)):
            if items:
                print(f"  {label}:")
                for line in items:
                    print(f"    - {line}")
        if not (r.hallucinations or r.misses or r.top1_failures):
            print("  clean sweep")

    if not enriched:
        print("\nNOTE: enriched.json is missing, so top-1 measures the raw catalog text.")
        print("      Run `python -m app.enrich` before trusting those numbers.")

    if len(reports) == 2:
        a, b = reports
        diffs = [
            (q, f, a.parsed[q][f], b.parsed[q][f])
            for q in a.parsed for f in ALL_FIELDS
            if a.parsed[q][f] != b.parsed[q][f]
        ]
        print(f"\nDISAGREEMENTS ({len(diffs)}):  {a.model} vs {b.model}")
        for q, f, av, bv in diffs:
            print(f"  {q!r:<42} {f:<10} {av!r} vs {bv!r}")
        if not diffs:
            print("  none -- the models extracted identical filters on every case")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=f"{ENRICH_MODEL},{QUERY_MODEL}",
                    help="comma-separated model ids to compare")
    ap.add_argument("--limit", type=int, help="run only the first N cases")
    ap.add_argument("--json", help="write raw results here")
    args = ap.parse_args()

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    cases = CASES[: args.limit] if args.limit else CASES

    index = SearchIndex()
    if not index.semantic:
        print("WARNING: sentence-transformers unavailable; top-1 uses lexical fallback.\n")

    print(f"{len(cases)} cases x {len(models)} model(s) = {len(cases) * len(models)} API calls")
    estimate = sum(
        len(cases) * (250 / 1e6 * PRICES.get(m, (0, 0))[0] + 150 / 1e6 * PRICES.get(m, (0, 0))[1])
        for m in models
    )
    print(f"rough cost estimate: ${estimate:.3f}\n")

    reports = []
    for model in models:
        print(f"running {model}...", flush=True)
        reports.append(run_model(model, index, cases))

    print_report(reports, enriched=index.enriched)

    if args.json:
        payload = [
            {"model": r.model, "cost": r.cost, "seconds": r.seconds,
             "expected": [r.expected_hit, r.expected_total],
             "top1": [r.top1_hit, r.top1_total],
             "hallucinations": r.hallucinations, "misses": r.misses,
             "top1_failures": r.top1_failures}
            for r in reports
        ]
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        print(f"\nwrote {args.json}")

    # Non-zero exit on any hallucination: this is the failure worth gating CI on.
    if any(r.hallucinations for r in reports):
        sys.exit(1)


if __name__ == "__main__":
    main()
