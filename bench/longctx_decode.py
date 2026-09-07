#!/usr/bin/env python3
"""Long-context decode bench: prefill a salted ~32k prompt, measure steady decode.

Reports: TTFT (prefill), mean step time over 300 tokens, acceptance proxy
(tokens/step via usage when available), and final tok/s.
Usage: python3 bench/longctx_decode.py [--ctx 32000] [--tokens 300]
"""
import argparse
import json
import random
import time
import urllib.request

BASE = "http://127.0.0.1:8888"
MODEL = "qwen3.8-flash-next"

FILLER = (
    "The lighthouse keeper recorded weather in a leather journal each morning. "
    "Gulls wheeled over the breakwater while fishing boats returned with the tide. "
)


def build_prompt(ctx_tokens: int, chars_per_tok: float = 1 / 0.2243) -> str:
    nonce = random.randrange(1 << 48)
    target_chars = int(ctx_tokens / 0.2243)
    parts = [f"[session {nonce}]\n"]
    n = 0
    while n < target_chars:
        parts.append(FILLER)
        n += len(FILLER)
    parts.append(
        "\n\nYou are a meticulous assistant. First, in ONE short sentence state "
        "how many journal entries the keeper likely wrote per week. Then write a "
        "detailed essay on the history of lighthouses."
    )
    return "".join(parts)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctx", type=int, default=32000)
    ap.add_argument("--tokens", type=int, default=300)
    ap.add_argument("--tag", default="longctx")
    args = ap.parse_args()

    prompt = build_prompt(args.ctx)
    body = json.dumps(
        {
            "model": MODEL,
            "max_tokens": args.tokens,
            "temperature": 0,
            "stream": True,
            "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"enable_thinking": False},
            "messages": [{"role": "user", "content": prompt}],
        }
    ).encode()

    req = urllib.request.Request(
        f"{BASE}/v1/chat/completions", data=body,
        headers={"Content-Type": "application/json"},
    )
    t0 = time.time()
    ttft = None
    chunks = 0
    completion_tokens = None
    with urllib.request.urlopen(req, timeout=600) as r:
        for line in r:
            if not line.startswith(b"data: "):
                continue
            payload = line[6:].strip()
            if payload == b"[DONE]":
                break
            d = json.loads(payload)
            u = d.get("usage")
            if u and u.get("completion_tokens"):
                completion_tokens = u["completion_tokens"]
            if d.get("choices") and d["choices"][0].get("delta", {}).get("content"):
                if ttft is None:
                    ttft = time.time() - t0
                chunks += 1
    total = time.time() - t0
    decode_time = total - (ttft or 0)
    print(
        f"[{args.tag}] ctx~{args.ctx} ttft={ttft:.2f}s "
        f"decode={decode_time:.2f}s chunks={chunks} "
        f"tok/s={((completion_tokens or chunks) / decode_time):.1f} "
        f"usage_completion={completion_tokens}"
    )


if __name__ == "__main__":
    main()
