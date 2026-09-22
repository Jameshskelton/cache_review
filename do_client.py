"""Thin client over DO Serverless Inference with retries and usage logging."""

import json
import os
import random
import time
from pathlib import Path

import requests

import rates as _rates

BASE = "https://inference.do-ai.run/v1/chat/completions"
LOG_DIR = Path("runs")

# Hard spend cap for this process. Override with BUDGET_USD=25 in the env.
# Every successful call is priced from the published rates and added to a
# running total; the next call raises instead of sending once the cap is hit.
# This is an estimate from list prices, not your actual invoice -- keep a
# margin and check the billing page afterwards.
BUDGET_USD = float(os.environ.get("BUDGET_USD", "15"))
_spent = 0.0


class BudgetExceeded(Exception):
    pass


class EffortRejected(Exception):
    pass


def spent() -> float:
    return _spent


def remaining() -> float:
    return BUDGET_USD - _spent


def _key() -> str:
    k = os.environ.get("MODEL_ACCESS_KEY")
    if not k:
        raise SystemExit("Set MODEL_ACCESS_KEY in your environment.")
    return k


def log_path(name: str) -> Path:
    LOG_DIR.mkdir(exist_ok=True)
    return LOG_DIR / f"{name}.jsonl"


def append(path: Path, record: dict) -> None:
    with path.open("a") as f:
        f.write(json.dumps(record) + "\n")


def chat(model: str, messages: list, max_tokens: int = 300,
         effort: str = None, temperature: float = 1.0,
         timeout: int = 600, max_retries: int = 5) -> dict:
    """
    One chat completion. Returns the parsed body plus a '_meta' block with
    wall-clock time and the effort value that was actually accepted.

    Retries on 429 and 5xx with jittered backoff. If the server rejects
    reasoning_effort with a 400, retries once without it and records that.
    """
    body = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if effort is not None:
        body["reasoning_effort"] = effort

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {_key()}",
    }

    global _spent
    if _spent >= BUDGET_USD:
        raise BudgetExceeded(
            f"Estimated spend ${_spent:.2f} has reached the ${BUDGET_USD:.2f} cap. "
            f"Raise it with BUDGET_USD=... if that's intended."
        )

    effort_used = effort
    attempt = 0
    while True:
        attempt += 1
        t0 = time.time()
        try:
            r = requests.post(BASE, headers=headers, json=body, timeout=timeout)
        except requests.RequestException as e:
            if attempt >= max_retries:
                raise
            time.sleep(min(60, 2 ** attempt) + random.random())
            continue
        elapsed = time.time() - t0

        if r.status_code == 400 and "reasoning_effort" in r.text and "reasoning_effort" in body:
            # Model or gateway won't take the knob. Drop it and note that.
            body.pop("reasoning_effort")
            effort_used = None
            continue

        if r.status_code in (429, 500, 502, 503, 504):
            if attempt >= max_retries:
                r.raise_for_status()
            time.sleep(min(60, 2 ** attempt) + random.random())
            continue

        r.raise_for_status()
        out = r.json()

        u = usage_fields(out)
        this_call = _rates.cost(
            model, u["prompt_tokens"], u["cached_tokens"], u["completion_tokens"]
        ) if model in _rates.RATES else 0.0
        _spent += this_call

        out["_meta"] = {
            "elapsed_s": round(elapsed, 3),
            "effort_requested": effort,
            "effort_used": effort_used,
            "attempts": attempt,
            "http_status": r.status_code,
            "call_cost_usd": this_call,
            "spent_so_far_usd": round(_spent, 6),
        }
        return out


def usage_fields(resp: dict) -> dict:
    """Flatten the bits of the usage object we care about."""
    u = resp.get("usage", {}) or {}
    ptd = u.get("prompt_tokens_details") or {}
    ctd = u.get("completion_tokens_details") or {}

    reasoning = ctd.get("reasoning_tokens")
    if reasoning is None:
        # Some deployments return the chain of thought as text instead of a
        # token count. Fall back to a rough char/4 estimate and flag it.
        msg = (resp.get("choices") or [{}])[0].get("message", {}) or {}
        rc = msg.get("reasoning_content")
        reasoning = (len(rc) // 4) if rc else None
        estimated = rc is not None
    else:
        estimated = False

    return {
        "prompt_tokens": u.get("prompt_tokens", 0),
        "cached_tokens": ptd.get("cached_tokens", 0) or 0,
        "cache_read_input_tokens": u.get("cache_read_input_tokens", 0) or 0,
        "cache_created_input_tokens": u.get("cache_created_input_tokens", 0) or 0,
        "completion_tokens": u.get("completion_tokens", 0),
        "reasoning_tokens": reasoning,
        "reasoning_estimated": estimated,
        "total_tokens": u.get("total_tokens", 0),
    }


def text_of(resp: dict) -> str:
    return ((resp.get("choices") or [{}])[0].get("message", {}) or {}).get("content") or ""
