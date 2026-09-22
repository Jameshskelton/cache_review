#!/usr/bin/env python3
"""
Turn the JSONL logs into the numbers and CSVs the article needs.

Outputs (to out/):
  cum_cost_by_turn.csv    one row per model/size/turn -- the main chart
  hit_rates.csv           cache hit magnitude and reliability per condition
  counterfactuals.csv     V4.1-Flash tokens repriced three ways
  effort_summary.csv      tokens, cost and accuracy per reasoning bucket

And prints the break-even analysis, which is the argument.

Usage:
    python analyze.py
"""

import csv
import json
import statistics as stats
from collections import defaultdict
from pathlib import Path

from rates import RATES, cost, full_breakeven, input_side_breakeven

RUNS = Path("runs")
OUT = Path("out")

PLATFORM_MEDIAN_CACHE_RATIO = 0.20  # typical cache_read/input across DO's catalog


def load(tag: str) -> list:
    p = RUNS / f"{tag}.jsonl"
    if not p.exists():
        return []
    rows = []
    for line in p.read_text().splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def write_csv(name: str, rows: list, fields: list) -> None:
    OUT.mkdir(exist_ok=True)
    with (OUT / name).open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"  wrote out/{name} ({len(rows)} rows)")


def analyze_sessions(rows: list) -> None:
    if not rows:
        print("No session data. Run run_sessions.py first.\n")
        return

    print("=" * 70)
    print("ARM 1: SESSION ECONOMICS")
    print("=" * 70)

    # Mean cumulative cost by (model, size, turn), averaged over corpora.
    buckets = defaultdict(list)
    for r in rows:
        buckets[(r["model"], r["target_size"], r["turn"])].append(r["cum_cost_usd"])

    cum_rows = [
        {
            "model": m,
            "label": RATES[m]["label"],
            "target_size": s,
            "turn": t,
            "mean_cum_cost_usd": round(stats.mean(v), 6),
            "n_sessions": len(v),
        }
        for (m, s, t), v in sorted(buckets.items())
    ]
    write_csv("cum_cost_by_turn.csv", cum_rows,
              ["model", "label", "target_size", "turn", "mean_cum_cost_usd", "n_sessions"])

    # Hit magnitude vs hit reliability -- these are different things and the
    # pilot showed the second one is where the surprise lives.
    hr = defaultdict(lambda: {"hits": 0, "n": 0, "rates": [], "prompt": [], "out": []})
    for r in rows:
        if r["turn"] == 1:
            continue  # turn 1 is cold by construction
        k = (r["model"], r["target_size"])
        d = hr[k]
        d["n"] += 1
        d["rates"].append(r["hit_rate"])
        d["prompt"].append(r["prompt_tokens"])
        d["out"].append(r["completion_tokens"])
        if r["cached_tokens"] > 0:
            d["hits"] += 1

    hit_rows = []
    print("\nCache behaviour, turns 2+:")
    for (m, s), d in sorted(hr.items()):
        reliability = d["hits"] / d["n"]
        mean_rate = stats.mean(d["rates"])
        blended = ((1 - mean_rate) * RATES[m]["input"]
                   + mean_rate * RATES[m]["cache_read"])
        hit_rows.append({
            "model": m, "label": RATES[m]["label"], "target_size": s,
            "turns": d["n"],
            "pct_turns_with_any_hit": round(reliability, 4),
            "mean_hit_rate": round(mean_rate, 4),
            "median_hit_rate": round(stats.median(d["rates"]), 4),
            "blended_input_rate_per_1m": round(blended, 4),
            "mean_prompt_tokens": round(stats.mean(d["prompt"]), 1),
            "mean_completion_tokens": round(stats.mean(d["out"]), 1),
        })
        print(f"  {RATES[m]['label']:<24} @{s:>7}: "
              f"{reliability:6.1%} of turns hit, mean coverage {mean_rate:6.1%}, "
              f"blended input ${blended:.4f}/M")

    write_csv("hit_rates.csv", hit_rows, list(hit_rows[0]))

    # Counterfactuals: same measured V4.1-Flash tokens, different cache pricing.
    cf = []
    running = defaultdict(float)
    for r in sorted([x for x in rows if x["model"] == "deepseek-v4.1-flash"],
                    key=lambda x: (x["target_size"], x["corpus"], x["turn"])):
        m = r["model"]
        actual = cost(m, r["prompt_tokens"], r["cached_tokens"], r["completion_tokens"])
        median = cost(m, r["prompt_tokens"], r["cached_tokens"], r["completion_tokens"],
                      cache_rate=RATES[m]["input"] * PLATFORM_MEDIAN_CACHE_RATIO)
        nocache = cost(m, r["prompt_tokens"], 0, r["completion_tokens"])

        k = (r["target_size"], r["corpus"])
        running[(k, "actual")] += actual
        running[(k, "median")] += median
        running[(k, "nocache")] += nocache

        cf.append({
            "target_size": r["target_size"], "corpus": r["corpus"], "turn": r["turn"],
            "cum_actual_usd": round(running[(k, "actual")], 6),
            "cum_at_median_cache_ratio_usd": round(running[(k, "median")], 6),
            "cum_no_cache_usd": round(running[(k, "nocache")], 6),
        })
    if cf:
        write_csv("counterfactuals.csv", cf, list(cf[0]))

    # The argument.
    print("\n" + "-" * 70)
    print("BREAK-EVEN: what cache hit rate does V4.1-Flash need to win?")
    print("-" * 70)
    a = "deepseek-v4.1-flash"
    for b in [m for m in RATES if m != a]:
        h = input_side_breakeven(a, b)
        print(f"\nvs {RATES[b]['label']}")
        print(f"  input side only:  ", end="")
        print(f"h = {h:.1%}" if h is not None else "no crossover")

        for (m, s), d in sorted(hr.items()):
            if m != a:
                continue
            pt, ct = stats.mean(d["prompt"]), stats.mean(d["out"])
            h2, reason = full_breakeven(a, b, pt, ct)
            observed = stats.mean(d["rates"])
            if h2 is None:
                print(f"  incl. output @{s:>7}: {reason}")
            else:
                verdict = "REACHED" if observed >= h2 else "not reached"
                print(f"  incl. output @{s:>7}: h = {h2:.1%} "
                      f"(observed {observed:.1%} -- {verdict})")


def analyze_effort(rows: list) -> None:
    if not rows:
        print("\nNo effort data. Run run_effort.py first.")
        return

    print("\n" + "=" * 70)
    print("ARM 2: REASONING EFFORT BUCKETS")
    print("=" * 70)

    by = defaultdict(list)
    for r in rows:
        by[r["bucket"]].append(r)

    order = ["low", "high", "xhigh", "max"]
    out = []
    baseline = None
    for bucket in [b for b in order if b in by] + [b for b in by if b not in order]:
        rs = by[bucket]
        reasoning = [r["reasoning_tokens"] or 0 for r in rs]
        comp = [r["completion_tokens"] for r in rs]
        costs = [r["cost_usd"] for r in rs]
        acc = sum(1 for r in rs if r["correct"]) / len(rs)
        est = any(r.get("reasoning_estimated") for r in rs)

        row = {
            "bucket": bucket, "n": len(rs),
            "accuracy": round(acc, 4),
            "median_reasoning_tokens": round(stats.median(reasoning), 1),
            "median_completion_tokens": round(stats.median(comp), 1),
            "mean_cost_usd": round(stats.mean(costs), 8),
            "median_latency_s": round(stats.median([r["elapsed_s"] for r in rs]), 2),
            "reasoning_tokens_estimated": est,
        }
        if baseline is None:
            baseline = row["mean_cost_usd"]
        row["cost_vs_low"] = round(row["mean_cost_usd"] / baseline, 2) if baseline else None
        out.append(row)

        print(f"  {bucket:>6}  n={row['n']:<4} acc={acc:6.1%}  "
              f"median_reasoning={row['median_reasoning_tokens']:>8}  "
              f"{row['cost_vs_low']:>5}x cost of 'low'")

    write_csv("effort_summary.csv", out, list(out[0]))
    if any(r["reasoning_tokens_estimated"] for r in out):
        print("\n  ! reasoning_tokens were estimated from reasoning_content "
              "length (chars/4). Say so in the article.")


def analyze_blocksize(rows: list) -> None:
    if not rows:
        return
    import math
    vals = [r["cached_tokens"] for r in rows if r["cached_tokens"] > 0]
    if not vals:
        print("\nBlock size: no hits recorded.")
        return
    g = 0
    for v in vals:
        g = math.gcd(g, v)
    print("\n" + "=" * 70)
    print(f"BLOCK SIZE: gcd of {len(vals)} nonzero values = {g}")
    print(f"  distinct values: {sorted(set(vals))[:20]}")


def main():
    analyze_sessions(load("sessions"))
    analyze_effort(load("effort"))
    analyze_blocksize(load("blocksize"))
    print("\nDone. CSVs in out/.")


if __name__ == "__main__":
    main()
