#!/usr/bin/env python3
"""
Probe DigitalOcean Serverless Inference prompt caching on deepseek-v4.1-flash.

Answers three questions:
  1. Does the usage object actually carry prompt_tokens_details.cached_tokens?
  2. Does a shared prefix produce a cache hit on the second call?
  3. Is the cache block-granular (DO's own docs show 128 of 214 tokens cached)?

Usage:
    export MODEL_ACCESS_KEY=...
    python cache_probe.py
"""

import json
import os
import time
import uuid

import requests

BASE = "https://inference.do-ai.run/v1/chat/completions"
MODEL = "deepseek-v4.1-flash"
KEY = os.environ["MODEL_ACCESS_KEY"]

# Filler that tokenizes predictably enough for a length sweep. Swap this for a
# real document once you've confirmed the mechanism works.
WORD = "lighthouse keeper inventory manifest entry "


def filler(approx_tokens: int) -> str:
    # ~6 tokens per repetition of WORD, rough but fine for a sweep.
    return WORD * max(1, approx_tokens // 6)


def call(content: str, label: str, max_tokens: int = 32) -> dict:
    body = {
        "model": MODEL,
        "max_tokens": max_tokens,
        "temperature": 1.0,
        "messages": [{"role": "user", "content": content}],
    }
    t0 = time.time()
    r = requests.post(
        BASE,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {KEY}",
        },
        json=body,
        timeout=300,
    )
    elapsed = time.time() - t0
    r.raise_for_status()
    usage = r.json().get("usage", {})

    prompt = usage.get("prompt_tokens")
    cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
    cache_read = usage.get("cache_read_input_tokens")

    print(
        f"{label:<28} prompt={prompt:<8} cached_tokens={cached} "
        f"cache_read={cache_read}  {elapsed:.2f}s"
    )
    return usage


def main():
    print("\n--- 1. Is the field present at all? ---")
    first = call("What is the capital of France?", "tiny prompt")
    if "prompt_tokens_details" not in first:
        print("\n!! prompt_tokens_details absent. Full usage object:")
        print(json.dumps(first, indent=2))
        print("Fall back to the reasoning_effort angle.")
        return

    print("\n--- 2. Miss then hit on a shared prefix ---")
    prefix = filler(4000)

    # Forced miss: a fresh UUID as the very first token breaks the prefix match.
    # This is the only way to get an uncached baseline, since DO-hosted models
    # cannot opt out of caching via X-Model-Affinity.
    call(f"{uuid.uuid4()}\n{prefix}\nSummarize in one line.", "forced miss (uuid first)")

    # Stable prefix, dynamic tail. First call populates.
    call(f"{prefix}\nQuestion A: how many entries?", "shared prefix, call 1")
    time.sleep(2)
    call(f"{prefix}\nQuestion B: name one entry.", "shared prefix, call 2")
    time.sleep(2)
    call(f"{prefix}\nQuestion B: name one entry.", "shared prefix, call 2 repeat")

    print("\n--- 3. Block granularity sweep ---")
    print("Look for cached_tokens landing on round numbers (128, 256, 512...)")
    for n in (256, 512, 1024, 2048, 4096, 8192):
        body = filler(n)
        call(body + "\nReply OK.", f"prefix ~{n} tok, warm")
        time.sleep(1)
        call(body + "\nReply OK again.", f"prefix ~{n} tok, reuse")
        time.sleep(1)


if __name__ == "__main__":
    main()
