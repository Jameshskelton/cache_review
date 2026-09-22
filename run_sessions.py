#!/usr/bin/env python3
"""
Arm 1 -- session economics.

Runs 15-turn conversations over a large static document and logs per-turn
token accounting for each model. The static prefix (system message + document)
never changes within a session; only the growing conversation tail does. That
is the shape real agents have, and the shape prompt caching is priced for.

Two design points worth carrying into the writeup:

1. Each session's system message opens with a unique session id. That gives the
   session its own cache lineage, so turn 1 is genuinely cold and sessions
   can't contaminate each other. Without it, a rerun looks suspiciously cheap.

2. Corpora must be mutually distinct. The pilot sweep used one repeated phrase
   at every length, so every condition shared a prefix with every other and
   'cold' calls came back partly cached. Use real, unrelated documents.

Usage:
    export MODEL_ACCESS_KEY=...
    mkdir -p corpora && cp your_docs/*.txt corpora/
    python run_sessions.py --sizes 50000 200000 --turns 15
"""

import argparse
import itertools
import sys
import uuid
from pathlib import Path

from do_client import append, chat, log_path, text_of, usage_fields
from rates import RATES, cost

CORPUS_DIR = Path("corpora")

SYSTEM_TEMPLATE = (
    "Session: {sid}\n"
    "You are a careful analyst. Answer questions about the document below "
    "using only its contents. Keep answers under 80 words.\n\n"
    "=== DOCUMENT ===\n{doc}\n=== END DOCUMENT ==="
)

# Deliberately document-agnostic so the same script runs over any corpus.
TURNS = [
    "Summarize the document in three sentences.",
    "What is the single most important claim it makes?",
    "List the main sections or topics it covers.",
    "Name one specific detail a careless reader would miss.",
    "What evidence does it offer for its central claim?",
    "Identify any numbers or quantities it reports.",
    "Who or what are the main entities discussed?",
    "What does it leave unexplained?",
    "Is there anything internally inconsistent in it?",
    "What would you need to verify it independently?",
    "Rewrite its opening idea in one plain sentence.",
    "What audience is it written for, and why do you think so?",
    "Name a question it implicitly raises but never answers.",
    "If you had to cut it in half, what would you remove?",
    "Give your overall assessment in two sentences.",
]


def load_corpora(target_tokens: int) -> list:
    """Truncate each corpus to roughly target_tokens (chars/4 heuristic)."""
    if not CORPUS_DIR.is_dir():
        raise SystemExit(f"Create {CORPUS_DIR}/ and add 3 distinct .txt or .md files.")
    files = sorted(
        p for p in CORPUS_DIR.iterdir()
        if p.suffix.lower() in (".txt", ".md") and p.is_file()
    )
    if len(files) < 2:
        raise SystemExit(f"Found {len(files)} corpus files in {CORPUS_DIR}/. Want 3.")

    target_chars = target_tokens * 4
    out = []
    for p in files:
        text = p.read_text(errors="replace")
        if len(text) < target_chars * 0.8:
            print(
                f"  ! {p.name} is ~{len(text)//4} tokens, short of {target_tokens}. "
                f"Using it as-is; actual prompt_tokens are logged either way.",
                file=sys.stderr,
            )
        out.append((p.stem, text[:target_chars]))
    return out


def run_session(model: str, corpus_name: str, doc: str, size: int,
                turns: int, logfile, max_tokens: int) -> None:
    sid = str(uuid.uuid4())
    effort = RATES[model].get("default_effort")
    messages = [{"role": "system", "content": SYSTEM_TEMPLATE.format(sid=sid, doc=doc)}]

    running = 0.0
    for i, question in enumerate(TURNS[:turns], start=1):
        messages.append({"role": "user", "content": question})
        resp = chat(model, messages, max_tokens=max_tokens, effort=effort)

        u = usage_fields(resp)
        reply = text_of(resp)
        messages.append({"role": "assistant", "content": reply})

        c = cost(model, u["prompt_tokens"], u["cached_tokens"], u["completion_tokens"])
        running += c
        hit = u["cached_tokens"] / u["prompt_tokens"] if u["prompt_tokens"] else 0.0

        append(logfile, {
            "arm": "sessions",
            "model": model,
            "corpus": corpus_name,
            "target_size": size,
            "session_id": sid,
            "turn": i,
            "cost_usd": c,
            "cum_cost_usd": running,
            "hit_rate": hit,
            **u,
            **resp["_meta"],
        })

        print(
            f"  t{i:>2} prompt={u['prompt_tokens']:>7} cached={u['cached_tokens']:>7} "
            f"({hit:6.1%}) out={u['completion_tokens']:>4} "
            f"${c:.6f} cum=${running:.5f} {resp['_meta']['elapsed_s']:>5.2f}s"
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=list(RATES))
    ap.add_argument("--sizes", nargs="+", type=int, default=[50_000, 200_000])
    ap.add_argument("--turns", type=int, default=15)
    ap.add_argument("--max-tokens", type=int, default=300)
    ap.add_argument("--tag", default="sessions")
    args = ap.parse_args()

    logfile = log_path(args.tag)
    print(f"Logging to {logfile}\n")

    for size in args.sizes:
        corpora = load_corpora(size)
        for model, (name, doc) in itertools.product(args.models, corpora):
            print(f"[{RATES[model]['label']}] {name} @ ~{size} tok")
            try:
                run_session(model, name, doc, size, args.turns, logfile, args.max_tokens)
            except Exception as e:
                print(f"  FAILED: {type(e).__name__}: {e}", file=sys.stderr)
            print()


if __name__ == "__main__":
    main()
