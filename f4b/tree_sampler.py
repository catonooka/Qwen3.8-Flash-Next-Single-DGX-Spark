"""Tree greedy rejection sampler — prototype for the Bole port (step 5).

Replaces vllm/v1/sample/rejection_sampler.py's rejection_greedy_sample_kernel
per-request linear scan with a tree-path accept scan. Layout: draft tokens of a
request form a tree in BFS order; node i has parent[i] (root's parent = -1).
Greedy/temp-0 semantics preserved exactly: walk from the deepest node whose
entire ancestor chain matched the target argmax; emit accepted chain + the
target's argmax at the rejection node (identical output distribution to chain
MTP at temp 0 — the ACCEPTED SET is a prefix-path by construction).

Kernel contract (drop-in for the greedy path):
  inputs per request: cu_num_draft_tokens offsets, draft_token_ids[num_tokens],
  target_argmax[num_tokens], tree_parent[num_tokens] (parent node index within
  request, -1 for root), tree_depth[num_tokens] (depth, root=0)
  output: output_token_ids[batch, max_spec_len+1] — same contract as stock.
"""
import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["max_spec_len"])
def tree_greedy_sample_kernel(
    output_token_ids_ptr,  # [batch_size, max_spec_len + 1]
    cu_num_draft_tokens_ptr,  # [batch_size]
    draft_token_ids_ptr,  # [num_tokens]
    target_argmax_ptr,  # [num_tokens]
    bonus_token_ids_ptr,  # [batch_size]
    tree_parent_ptr,  # [num_tokens] parent node idx within request, -1 = root
    tree_depth_ptr,  # [num_tokens]
    max_spec_len,
):
    req_idx = tl.program_id(0)
    start_idx = (
        tl.zeros([], dtype=cu_num_draft_tokens_ptr.dtype.element_ty)
        if req_idx == 0
        else tl.load(cu_num_draft_tokens_ptr + req_idx - 1)
    )
    end_idx = tl.load(cu_num_draft_tokens_ptr + req_idx)
    num_nodes = end_idx - start_idx

    # Pass 1: mark matched nodes (draft token == target argmax at that node).
    # A node is ACCEPTED iff itself and ALL ancestors are matched.
    # Compute per-node accept by walking parents (depth is small, <= 8).
    best_node = -1  # deepest accepted node
    best_depth = -1
    for i in range(num_nodes):
        node_matched = (
            tl.load(draft_token_ids_ptr + start_idx + i)
            == tl.load(target_argmax_ptr + start_idx + i).to(tl.int32)
        )
        if node_matched:
            # walk ancestor chain
            all_matched = True
            p = tl.load(tree_parent_ptr + start_idx + i)
            while p >= 0 and all_matched:
                p_matched = (
                    tl.load(draft_token_ids_ptr + start_idx + p)
                    == tl.load(target_argmax_ptr + start_idx + p).to(tl.int32)
                )
                all_matched = all_matched and p_matched
                p = tl.load(tree_parent_ptr + start_idx + p)
            if all_matched:
                d = tl.load(tree_depth_ptr + start_idx + i)
                if d > best_depth:
                    best_depth = d
                    best_node = i

    # Pass 2: emit tokens along the accepted path (root -> best_node).
    out_base = output_token_ids_ptr + req_idx * (max_spec_len + 1)
    if best_node < 0:
        # nothing accepted: emit target argmax at root
        tl.store(out_base, tl.load(target_argmax_ptr + start_idx))
    else:
        # collect path (depths are 0..best_depth, one node per depth on path)
        # emit in depth order: simplest = for each depth, find path node
        # path recovery: walk parents from best_node, store by depth, then
        # re-emit ascending. Triton can't index local arrays dynamically, so
        # emit via parent-walk into reversed positions: node at depth d goes
        # to slot d. We know each depth has exactly one path node.
        cur = best_node
        while cur >= 0:
            d = tl.load(tree_depth_ptr + start_idx + cur)
            tok = tl.load(target_argmax_ptr + start_idx + cur)
            tl.store(out_base + d, tok)
            cur = tl.load(tree_parent_ptr + start_idx + cur)
        # bonus token at slot best_depth+1
        tl.store(out_base + best_depth + 1, tl.load(bonus_token_ids_ptr + req_idx))


def tree_greedy_sample(
    draft_token_ids: torch.Tensor,   # [num_tokens] int32
    target_argmax: torch.Tensor,     # [num_tokens] int64
    bonus_token_ids: torch.Tensor,   # [batch]
    tree_parent: torch.Tensor,       # [num_tokens] int32 (-1 root)
    tree_depth: torch.Tensor,        # [num_tokens] int32
    num_draft_tokens: torch.Tensor,  # [batch]
    max_spec_len: int,
    device: str = "cuda",
) -> torch.Tensor:
    batch_size = num_draft_tokens.shape[0]
    cu = torch.zeros(batch_size, dtype=torch.int32, device=device)
    torch.cumsum(num_draft_tokens, dim=0, out=cu)
    out = torch.full(
        (batch_size, max_spec_len + 1), -1, dtype=torch.int32, device=device
    )
    tree_greedy_sample_kernel[(batch_size,)](
        out, cu, draft_token_ids.to(torch.int32),
        target_argmax, bonus_token_ids,
        tree_parent.to(torch.int32), tree_depth.to(torch.int32),
        max_spec_len,
    )
    return out
