# DeepSeek-V4.1-Flash on DigitalOcean — measurement harness

Two arms and a side probe, built to support one argument: V4.1-Flash raised
its sticker price and cut its cache price, and whether that trade pays depends
entirely on a cache hit rate you may not actually achieve.

## Setup

```bash
pip install requests
export MODEL_ACCESS_KEY=...
mkdir -p corpora
# Add three mutually distinct .txt or .md files, each ≥800KB if you want the
# 200K-token condition to be real. Public-domain books, a docs export, and a
# concatenated codebase work well — different genres is a feature.
```

Record where the corpora came from. You will need to say in the article, and
a reader will ask.

## Run order

```bash
python run_blocksize.py --reps 5      # ~5 min, pennies
python run_sessions.py                 # ~40-60 min, the main cost
python run_effort.py --reps 3          # ~30 min
python analyze.py
```

Start with `run_blocksize.py`. It is cheap, it confirms the client works
end to end, and it settles the 64-token question before you commit budget.

### Cost

Arm 1 is 3 models × 2 sizes × 3 corpora × 15 turns = 270 calls. The 200K
condition dominates: each cold prefill is roughly $0.06 on V4.1-Flash, and
there are 9 of them per model. Expect **$5–15 total** across both arms. Check
your prepaid balance first — serverless inference is prepaid only and cuts
off at zero.

## What each script controls for

**Session-scoped cache namespaces.** Every session's system message opens with
a fresh UUID, so turn 1 is genuinely cold and reruns don't inherit yesterday's
cache. Without this the numbers drift downward every time you run.

**Distinct corpora.** Your pilot's length sweep reused one repeated phrase, so
the ~1024 "cold" call came back with 192 cached tokens it should not have had.
`run_blocksize.py` now generates a different random-word prefix per length.

**Cache excluded from Arm 2.** Effort-sweep prompts get a UUID prefix on every
call, so reasoning-token measurements aren't contaminated by cache hits.

**Effort fallback.** `do_client.chat` retries without `reasoning_effort` if the
server 400s on it, and records `effort_used: null`. That is how you find out
which models accept the knob without writing a separate probe.

## Known failure modes

- **`cached_tokens` is best-effort.** DO's docs say so explicitly and your
  pilot showed it — the ~512 condition missed twice, and ~4096 returned *fewer*
  cached tokens on reuse than on the warm call. `analyze.py` reports hit
  reliability (what fraction of turns hit at all) separately from hit magnitude
  (how much of the prompt was covered). Report both. The gap between them is
  the most interesting thing you have.
- **`reasoning_tokens` may be absent.** The client falls back to estimating
  from `reasoning_content` length and flags it. If both are missing, fall back
  to comparing `completion_tokens` and say that's what you did.
- **`max_tokens` truncation at high effort.** `run_effort.py` defaults to 8000.
  If `max` items come back with `finish_reason: length`, raise it — a truncated
  thought undercounts the bucket and inflates its apparent accuracy penalty.
- **Latency is not measured here.** Your pilot had a 460-token call take 3.97s
  and a 3445-token call take 1.28s. Serverless queueing swamps the signal at
  these sizes. Elapsed time is logged; don't build a chart on it.

## Honesty notes for the draft

V4 Flash 0731 is a weaker model. Cost per token is not cost per task, and this
harness measures the former. Either grade the session outputs or state plainly
that you are comparing price structures rather than value delivered.

Prompt caching for open-source models is in public preview, and the four-bucket
effort restriction is likely a gateway choice DO can change without notice.
Date-stamp every claim.
