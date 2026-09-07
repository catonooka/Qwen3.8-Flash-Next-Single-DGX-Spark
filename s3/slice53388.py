#!/usr/bin/env python3
"""Slice pr53388.diff into per-file diffs under /s3/out53388/ and apply to
copies of the image's vllm sources. Runs INSIDE the skinny container."""
import os
import re
import shutil
import subprocess

VDIR = "/usr/local/lib/python3.12/dist-packages"
OUT = "/s3/out53388"

src = open("/s3/pr53388.diff").read()
parts = re.split(r"(?=^diff --git )", src, flags=re.M)
targets = []
os.makedirs(OUT, exist_ok=True)

for p in parts:
    m = re.match(r"diff --git a/(\S+) b/(\S+)", p)
    if not m:
        continue
    path = m.group(2)
    if not path.startswith("vllm/") or "/mooncake/" in path:
        continue
    dst = os.path.join(OUT, path)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy2(os.path.join(VDIR, path), dst)
    r = subprocess.run(
        ["patch", dst, "-i", "/dev/stdin"],
        input=p.encode(), capture_output=True,
    )
    status = "PATCHED" if r.returncode == 0 else "FAILED"
    print(f"{status:8s} {path}")
    if r.returncode != 0:
        print(r.stdout.decode()[-300:])
        print(r.stderr.decode()[-200:])
    targets.append(path)

print(f"--- {len(targets)} files")
for t in targets:
    fp = os.path.join(OUT, t)
    if t.endswith(".py"):
        compile(open(fp).read(), fp, "exec")
print("compile-OK")
print(open(os.path.join(OUT, "vllm/config/speculative.py")).read().count("disable_eagle_block_drop"), "flag mentions")
