#!/usr/bin/env python3
"""Prefill ladder: ctx tokens -> TTFT -> prefill tok/s. Matches overnight-2026-09-05 methodology (ctx/TTFT fit)."""
import argparse, json, statistics, sys, time, urllib.request

BASE, MODEL = "http://localhost:8888", "qwen3.8-flash-next"
FILLER = ("Entry {i:06d}: the quarterly logistics audit recorded a routine "
          "variance in the northbound depot inventory.\n")
TASK = "In one short sentence, what kind of document is the log above?"

def build_ctx(t):
    return "".join(FILLER.format(i=i) for i in range(max(1, int(t / 25))))

def post(payload, timeout=1800):
    req = urllib.request.Request(BASE + "/v1/chat/completions",
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        first = None
        n = 0
        for line in r:
            if line.startswith(b"data:"):
                n += 1
                if first is None and (b'"content"' in line or b'[DONE]' in line):
                    first = time.time() - t0
                    if b'[DONE]' in line:
                        break
        r.read()
    return first if first is not None else (time.time() - t0)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--ctx", type=int, nargs="+", default=[8192, 16384, 32768, 65536])
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--out", default="logs/prefill_ladder.jsonl")
    ap.add_argument("--note", default="")
    a = ap.parse_args()

    # warm-up: one mid-size prompt to eat PLE page-cache + JIT cold shapes
    post({"model": MODEL, "max_tokens": 2, "temperature": 0,
          "chat_template_kwargs": {"enable_thinking": False},
          "messages": [{"role": "user", "content": build_ctx(4096) + TASK}]}, timeout=600)

    for ctx in a.ctx:
        rows = []
        for rep in range(a.repeats):
            # unique salt at the START: prefix caching is ON, nested/repeated
            # fillers would all hit the radix cache and measure nothing
            import random
            nonce = random.randrange(1 << 48)
            salt = f"[session {nonce}]\n"
            prompt = salt + build_ctx(ctx) + TASK
            ttft = post({"model": MODEL, "max_tokens": 2, "temperature": 0,
                         "chat_template_kwargs": {"enable_thinking": False},
                         "messages": [{"role": "user", "content": prompt}]}, timeout=1800)
            tps = ctx / ttft
            rows.append({"ctx": ctx, "rep": rep, "ttft_s": round(ttft, 3),
                         "prefill_tps": round(tps, 1)})
            print(f"[{a.tag}] ctx={ctx:>6} rep{rep} ttft={ttft:7.2f}s  prefill={tps:7.1f} tok/s")
        rec = {"tag": a.tag, "note": a.note, "rows": rows,
               "ctx": ctx, "mean_tps": round(statistics.mean(r["prefill_tps"] for r in rows), 1)}
        with open(a.out, "a") as f:
            f.write(json.dumps(rec) + "\n")

if __name__ == "__main__":
    main()
