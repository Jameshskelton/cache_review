#!/usr/bin/env python3
"""
Cache block granularity, done cleanly.

Your pilot suggested 64-token blocks: every observed cached value (192, 448,
832, 1728, 3328, 3392) is a multiple of 64, and 192/448/832/1728 are odd
multiples, which rules out 128. But that sweep reused one filler phrase at
every length, so the conditions shared prefixes and the 'cold' calls weren't
cold. This version builds a distinct random-word prefix per length, and
namespaces each condition with a UUID so nothing leaks between them.

It also repeats each length, because your ~512 case returned zero cached
tokens on both calls while ~256 worked. Whether that is a size threshold or
just best-effort flakiness is exactly the question, and one shot per length
cannot answer it.

Usage:
    export MODEL_ACCESS_KEY=...
    python run_blocksize.py --reps 5
"""

import argparse
import math
import random
import uuid
from collections import defaultdict

from do_client import append, chat, log_path, usage_fields

WORDS = """anchor ballast cadence damper ember fathom girder harbor ingot
jetty kelp lantern mizzen nautical oakum pennant quay rudder sextant tiller
undertow vessel windlass yardarm azimuth bollard capstan davit ensign fluke
gunwale halyard isobar keelson leeward mainsail netting orlop painter quadrant
ratline scuttle transom uphaul vang wharf""".split()


def distinct_filler(approx_tokens: int, seed: int) -> str:
    """A prefix that shares no leading tokens with any other length."""
    rng = random.Random(seed)
    # Roughly 1.3 tokens per word for this vocabulary; overshoot then trim.
    n = int(approx_tokens / 1.3) + 50
    return " ".join(rng.choice(WORDS) for _ in range(n))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="deepseek-v4.1-flash")
    ap.add_argument("--lengths", nargs="+", type=int,
                    default=[256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096])
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--tag", default="blocksize")
    args = ap.parse_args()

    logfile = log_path(args.tag)
    print(f"Logging to {logfile}\n")

    observed = []
    hits_by_len = defaultdict(lambda: [0, 0])

    for i, length in enumerate(args.lengths):
        prefix = distinct_filler(length, seed=1000 + i)
        sid = uuid.uuid4()  # namespaces this length; no cross-length leakage
        print(f"~{length} tokens")

        for rep in range(args.reps):
            content = f"[{sid}]\n{prefix}\nReply with the single word OK. #{rep}"
            resp = chat(args.model, [{"role": "user", "content": content}],
                        max_tokens=8, effort=None)
            u = usage_fields(resp)

            append(logfile, {
                "arm": "blocksize",
                "model": args.model,
                "target_length": length,
                "rep": rep,
                **u,
                **resp["_meta"],
            })

            hits_by_len[length][1] += 1
            if u["cached_tokens"] > 0:
                hits_by_len[length][0] += 1
                observed.append(u["cached_tokens"])

            print(f"  rep{rep} prompt={u['prompt_tokens']:>6} "
                  f"cached={u['cached_tokens']:>6} "
                  f"{resp['_meta']['elapsed_s']:>5.2f}s")

    print("\nHit rate by prefix length (rep 0 is expected to miss):")
    for length, (hits, total) in sorted(hits_by_len.items()):
        print(f"  ~{length:>5}: {hits}/{total}")

    if observed:
        g = 0
        for v in observed:
            g = math.gcd(g, v)
        print(f"\nGCD of all {len(observed)} nonzero cached_tokens values: {g}")
        print("That is your block size, assuming no condition is degenerate.")
        print(f"Distinct values seen: {sorted(set(observed))}")
    else:
        print("\nNo cache hits at all. Check that reps are close enough in time.")


if __name__ == "__main__":
    main()
