#!/usr/bin/env python3
"""
Arm 2 -- what the four reasoning_effort buckets actually cost.

The model card documents a continuous integer 1-100 knob. DigitalOcean's
launch post repeats that. The API accepts only low / high / xhigh / max --
no 'medium', no off switch. This measures where those four stops land.

Each prompt has one unambiguous numeric answer, so grading needs no judge
model. Accuracy is a sanity check, not the headline: with 20 easy items you
are measuring token spend, not capability. Say so in the writeup.

Usage:
    export MODEL_ACCESS_KEY=...
    python run_effort.py --reps 3
"""

import argparse
import re
import sys
import uuid

from do_client import append, chat, log_path, text_of, usage_fields
from rates import EFFORT_BUCKETS, RATES, cost

# (id, prompt, expected answer)
ITEMS = [
    ("q01", "What is 17 * 23?", 391),
    ("q02", "What is 2^12?", 4096),
    ("q03", "A train travels 180 km in 2.5 hours. What is its average speed in km/h?", 72),
    ("q04", "How many positive divisors does 36 have?", 9),
    ("q05", "What is the sum of the first 20 positive integers?", 210),
    ("q06", "What is 144 divided by 16?", 9),
    ("q07", "If 5x + 3 = 38, what is x?", 7),
    ("q08", "What is the greatest common divisor of 84 and 126?", 42),
    ("q09", "How many seconds are in 3 hours and 25 minutes?", 12300),
    ("q10", "What is 15 percent of 260?", 39),
    ("q11", "A rectangle measures 14 by 9. What is its perimeter?", 46),
    ("q12", "What is the next prime number after 89?", 97),
    ("q13", "What is 7 factorial?", 5040),
    ("q14", "How many edges does a cube have?", 12),
    ("q15", "What is the median of 3, 9, 4, 12, 7?", 7),
    ("q16", "Solve for x: 3x - 7 = 2x + 5.", 12),
    ("q17", "What is the area of a triangle with base 18 and height 7?", 63),
    ("q18", "How many minutes elapse between 09:42 and 13:17 on the same day?", 215),
    ("q19", "What is the sum of the interior angles of a hexagon, in degrees?", 720),
    ("q20", "A shirt costs $45 after a 25 percent discount. What was the original price?", 60),
]

SUFFIX = " Reply with only the number, no units and no explanation."

NUM = re.compile(r"-?\d[\d,]*\.?\d*")


def load_items(which: str):
    if which == "hard":
        from hard_items import ITEMS as HARD
        return HARD
    return ITEMS


def graded(reply: str, expected) -> bool:
    """Compare the last number in the reply against the expected value."""
    found = NUM.findall(reply.replace("$", ""))
    if not found:
        return False
    try:
        got = float(found[-1].replace(",", ""))
    except ValueError:
        return False
    return abs(got - float(expected)) < 1e-6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="deepseek-v4.1-flash")
    ap.add_argument("--reps", type=int, default=3,
                    help="repeats per (item, bucket); reasoning length is noisy")
    ap.add_argument("--max-tokens", type=int, default=8000,
                    help="must be generous or 'max' will truncate mid-thought")
    ap.add_argument("--buckets", nargs="+", default=EFFORT_BUCKETS)
    ap.add_argument("--items", default="easy", choices=["easy", "hard"],
                    help="'hard' loads hard_items.py; the easy set does not "
                         "load the effort knob at all")
    ap.add_argument("--tag", default="effort")
    args = ap.parse_args()

    items = load_items(args.items)

    logfile = log_path(args.tag)
    print(f"Logging to {logfile}")
    print(f"{len(items)} items x {len(args.buckets)} buckets x {args.reps} reps "
          f"= {len(items)*len(args.buckets)*args.reps} calls\n")

    for bucket in args.buckets:
        tot_out = tot_reason = correct = n = 0
        for qid, prompt, expected in items:
            for rep in range(args.reps):
                # Unique prefix per call: this arm must not benefit from cache.
                content = f"[{uuid.uuid4()}] {prompt}{SUFFIX}"
                msgs = [{"role": "user", "content": content}]
                try:
                    resp = chat(args.model, msgs,
                                max_tokens=args.max_tokens, effort=bucket)
                except Exception as e:
                    print(f"  {qid} rep{rep} FAILED: {type(e).__name__}: {e}",
                          file=sys.stderr)
                    continue

                u = usage_fields(resp)
                reply = text_of(resp)
                ok = graded(reply, expected)
                c = cost(args.model, u["prompt_tokens"],
                         u["cached_tokens"], u["completion_tokens"])

                append(logfile, {
                    "arm": "effort",
                    "model": args.model,
                    "bucket": bucket,
                    "item": qid,
                    "rep": rep,
                    "expected": expected,
                    "reply": reply[:200],
                    "correct": ok,
                    "cost_usd": c,
                    **u,
                    **resp["_meta"],
                })

                n += 1
                correct += int(ok)
                tot_out += u["completion_tokens"]
                tot_reason += u["reasoning_tokens"] or 0

        if n:
            print(f"{bucket:>6}  n={n:<4} acc={correct/n:6.1%}  "
                  f"mean_out={tot_out/n:8.1f}  mean_reasoning={tot_reason/n:8.1f}")
        else:
            print(f"{bucket:>6}  no successful calls")

    print("\nIf mean_reasoning is 0 everywhere, the deployment isn't reporting "
          "reasoning_tokens; check whether 'reasoning_estimated' is true in the "
          "log, and if not, fall back to comparing completion_tokens only.")


if __name__ == "__main__":
    main()
