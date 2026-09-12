import os
os.environ.setdefault('CUDA_MODULE_LOADING', 'LAZY')
import torch, sys, time
sys.path.insert(0, '/m')
torch.manual_seed(0)
dev = 'cuda'

from fused_sigmoid_gating_tree import fused_sigmoid_gating_delta_rule_update as f_tree

# Real model GDN dims: HV=48 value heads, K=V=128
HV, K, V = 48, 128, 128

def make_flat(T):
    q = torch.randn(1, T, HV, K, device=dev, dtype=torch.bfloat16)
    k = torch.randn(1, T, HV, K, device=dev, dtype=torch.bfloat16)
    v = torch.randn(1, T, HV, V, device=dev, dtype=torch.bfloat16)
    a = torch.randn(1, T, HV, device=dev, dtype=torch.float32)
    b = torch.randn(1, T, HV, device=dev, dtype=torch.float32)
    return q, k, v, a, b

A_log = torch.randn(HV, device=dev, dtype=torch.float32)
dt_bias = torch.randn(HV, device=dev, dtype=torch.float32)
SLOTS = 128
h = torch.zeros(SLOTS, HV, V, K, device=dev, dtype=torch.float32)

def bench(T, groups, init_idx, label, iters=200):
    q, k, v, a, b = make_flat(T)
    cu = torch.tensor([i * (T // groups) for i in range(groups + 1)], dtype=torch.int32, device=dev)
    width = T // groups
    idx = torch.zeros(groups, width, dtype=torch.int32, device=dev)
    row = torch.arange(1, groups + 1, device=dev)
    idx[:, :] = (row * width).unsqueeze(1) + torch.arange(width, device=dev).unsqueeze(0)
    idx.clamp_(max=SLOTS - 1)
    acc = torch.full((groups,), width, dtype=torch.int32, device=dev)
    def run():
        return f_tree(A_log, a, b, dt_bias, q, k, v,
                      initial_state=h, inplace_final_state=True,
                      cu_seqlens=cu, ssm_state_indices=idx,
                      num_accepted_tokens=acc,
                      initial_state_indices=init_idx,
                      use_qk_l2norm_in_kernel=True)
    for _ in range(20): run()
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(iters): run()
    torch.cuda.synchronize()
    us = (time.perf_counter() - t0) / iters * 1e6
    print(f'{label:38s} T={T:3d} groups={groups:2d} -> {us:7.1f} us')
    return us

# chain today: 1 request x 4 rows (MTP3+bonus)
bench(4, 1, torch.tensor([0], dtype=torch.int32, device=dev), 'CHAIN today (B=1, 4 rows)')
# tree proposals: same request as multiple sibling groups sharing parent slot 0
bench(8, 8, torch.zeros(8, dtype=torch.int32, device=dev), 'TREE 8 groups x 1 row (depth-1 fanout-8)')
bench(16, 8, torch.zeros(8, dtype=torch.int32, device=dev), 'TREE 8 groups x 2 rows (fanout-8 depth-2)')
bench(8, 4, torch.zeros(4, dtype=torch.int32, device=dev), 'TREE 4 groups x 2 rows (fanout-4 depth-2)')
bench(16, 4, torch.zeros(4, dtype=torch.int32, device=dev), 'TREE 4 groups x 4 rows (fanout-4 depth-4)')
# concurrency for scale
bench(32, 16, torch.zeros(16, dtype=torch.int32, device=dev), 'TREE 16 groups x 2 (2 reqs x fanout-8)')
