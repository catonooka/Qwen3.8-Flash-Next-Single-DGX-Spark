import torch, time, sys

torch.manual_seed(0)
dev = "cuda"
V, H = 248320, 2560
B = 4  # verify batch with MTP-3 at 1 stream (1+3), pad to 4

Wbf = (torch.randn(V, H, device=dev) * 0.01318).to(torch.bfloat16)  # W5 measured sigma
h = torch.randn(B, H, device=dev, dtype=torch.bfloat16)

def bench(fn, n=50, warmup=10):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n): fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n * 1000  # ms

# 1) stock BF16 GEMV
t_bf = bench(lambda: torch.nn.functional.linear(h, Wbf))
gbps_bf = (V * H * 2) / (t_bf / 1000) / 1e9
print(f"BF16 F.linear  [{B},{H}]x[{V},{H}]: {t_bf:.3f} ms  = {gbps_bf:.0f} GB/s effective")

# 2) INT8 per-row via torch._int_mm
scale = Wbf.abs().amax(dim=1) / 127.0
scale = scale.float()
Wi8 = torch.round(Wbf.float() / scale[:, None]).clamp(-127, 127).to(torch.int8)
h16 = h.float()
def int8_path():
    out = torch._int_mm(h16.to(torch.int8) * 0 + 0, Wi8.t()) if False else None
    return out
# proper: activations also int8 (dynamic per-row)
ascale = h16.abs().amax(dim=1, keepdim=True) / 127.0
hi8 = torch.round(h16 / ascale).clamp(-127, 127).to(torch.int8)
def int8_path2():
    acc = torch._int_mm(hi8, Wi8.t())  # [B, V] int32
    return acc.float() * ascale * scale[None, :]
try:
    t_i8 = bench(int8_path2)
    gbps_i8 = (V * H) / (t_i8 / 1000) / 1e9
    print(f"INT8 _int_mm   [{B},{H}]x[{V},{H}]: {t_i8:.3f} ms  = {gbps_i8:.0f} GB/s effective (W bytes)")
except Exception as e:
    print("int_mm failed:", e)

# 3) top-64 rescore cost (on top of either)
def rescore(idx):
    w64 = Wbf[idx.reshape(-1)].view(B, 64, H)
    hh = h.unsqueeze(1).expand(-1, 64, -1)
    return torch.einsum('bkh,bkh->bk', hh.float(), w64.float())
idx = torch.randint(0, V, (B, 64), device=dev)
t_rs = bench(lambda: rescore(idx))
print(f"top-64 BF16 rescore: {t_rs:.3f} ms")

# 4) correctness probe: argmax match
with torch.no_grad():
    lg_bf = torch.nn.functional.linear(h, Wbf).float()
    lg_i8 = int8_path2()
    am_bf = lg_bf.argmax(-1)
    # screen top-64 from i8, rescore exact
    top64 = lg_i8.topk(64, dim=-1).indices
    w64 = Wbf[top64.reshape(-1)].view(B, 64, H)
    hh = h.unsqueeze(1).expand(-1, 64, -1)
    ex = torch.einsum('bkh,bkh->bk', hh.float(), w64.float())
    pick = top64.gather(1, ex.argmax(-1, keepdim=True)).squeeze(1)
    # is BF16 argmax inside i8-top64?
    in64 = (top64 == am_bf.unsqueeze(1)).any(dim=1).float().mean().item()
    match = (pick == am_bf).float().mean().item()
    margins = (lg_bf.topk(2, dim=-1).values[:, 0] - lg_bf.topk(2, dim=-1).values[:, 1])
    print(f"BF16-top1 in INT8-top64: {in64*100:.1f}%   argmax match after rescore: {match*100:.1f}%")
    print(f"min/median margin: {margins.min():.4f} / {margins.median():.4f} logits")
