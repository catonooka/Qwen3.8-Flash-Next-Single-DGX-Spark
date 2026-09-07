#!/usr/bin/env python3
"""Warm agent-turn bench: send a fixed long prompt (turn 1), then re-send
with each extra turn appended. Measures per-turn TTFT — the cost the
disable_eagle_block_drop flag targets (trailing-block keep vs re-prefill).

Usage: python3 bench/warmturn.py [--ctx 16000] [--turns 5] [--tag X]
"""
import argparse
import json
import time
import urllib.request

BASE = "http://127.0.0.1:8888"
MODEL = "qwen3.8-flash-next"
FILLER = (
    "The archive clerk catalogued each shipment manifest by harbor and date. "
    "Cargo manifests listed timber, salt, and printed maps bound for the capital. "
)


def build(ctx_tokens: int) -> str:
    target = int(ctx_tokens / 0.2243)
    parts = []
    n = 0
    while n < target:
        parts.append(FILLER)
        n += len(FILLER)
    return "".join(parts)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctx", type=int, default=16000)
    ap.add_argument("--turns", type=int, default=5)
    ap.add_argument("--tag", default="warmturn")
    args = ap.parse_args()

    base_prompt = build(args.ctx)
    msgs = [{"role": "user", "content": base_prompt + "\nSummarize the manifests in one sentence."}]

    for t in range(args.turns):
        body = json.dumps(
            {
                "model": MODEL,
                "max_tokens": 8,
                "temperature": 0,
                "chat_template_kwargs": {"enable_thinking": False},
                "messages": msgs,
            }
        ).encode()
        req = urllib.request.Request(
            f"{BASE}/v1/chat/completions", data=body,
            headers={"Content-Type": "application/json"},
        )
        t0 = time.time()
        with urllib.request.urlopen(req, timeout=600) as r:
            json.load(r)
        dt = time.time() - t0
        print(f"[{args.tag}] turn {t + 1}: ttft+8tok = {dt:.3f}s")
        # agent-style: append assistant ack + next user turn (reuses prefix)
        msgs.append({"role": "assistant", "content": "OK."})
        msgs.append({"role": "user", "content": f"Turn {t + 2}: list the harbors mentioned so far, briefly."})


if __name__ == "__main__":
    main()
