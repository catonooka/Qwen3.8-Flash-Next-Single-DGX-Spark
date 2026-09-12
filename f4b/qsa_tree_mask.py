"""QSA tree ancestor-mask prototype — Bole port step 4.

Injection point: qsa.py `_indexer_score_kernel` writes per-token block scores
to logits[row, columns] (verified: tl.store at ~line 117, scores for columns
< visible). The top-k block selection then reads this matrix. For tree
verify, sibling nodes must only attend their OWN ancestors: mask non-ancestor
columns to -inf before top-k.

This prototype implements the mask pass + a correctness test: given a scores
matrix and per-token ancestor position sets, filtered top-k must equal the
top-k of the ancestor-restricted score matrix.
"""
import torch


def ancestor_block_mask(
    num_tokens: int,
    num_blocks: int,
    block_size: int,  # COMPRESS_RATIO: positions per block
    ancestors: list[list[int]],  # per token: logical positions visible (own chain)
    device: str = "cuda",
) -> torch.Tensor:
    """Boolean mask [num_tokens, num_blocks]: True = block intersects ancestors."""
    mask = torch.zeros(num_tokens, num_blocks, dtype=torch.bool, device=device)
    for i, positions in enumerate(ancestors):
        if positions:
            blocks = sorted({p // block_size for p in positions})
            mask[i, blocks] = True
    return mask


def apply_tree_mask(
    scores: torch.Tensor,  # [num_tokens, num_blocks] fp32
    mask: torch.Tensor,  # [num_tokens, num_blocks] bool
) -> torch.Tensor:
    """Set non-ancestor blocks to -inf (in-place semantics like the kernel)."""
    out = scores.clone()
    out[~mask] = float("-inf")
    return out


def _test():
    torch.manual_seed(0)
    dev = "cuda"
    T, NB, CR = 12, 64, 32
    scores = torch.randn(T, NB, device=dev)

    # tree: req has base position 100; nodes 0..5 laid out at logical positions
    # 100..105 (unique positions to avoid page-table collision). Ancestor sets:
    # node0 (root, pos100): {0..99 in cache + nothing new}   -> blocks 0..3
    # node1 (child A, pos101): cache + {100}
    # node2 (child B, pos101 sibling — but unique logical pos 102): cache + {100}
    # node3 (grandchild A1, pos103): cache + {100, 101}
    # node4 (grandchild A2, pos104): cache + {100, 101}
    # node5 (grandchild B1, pos105): cache + {100, 102}
    cache_positions = 100
    real_anc = [
        list(range(cache_positions)),                    # node0 root
        list(range(cache_positions)) + [100],            # node1 A
        list(range(cache_positions)) + [100],            # node2 B (unique pos 102)
        list(range(cache_positions)) + [100, 101],       # node3 A1
        list(range(cache_positions)) + [100, 101],       # node4 A2
        list(range(cache_positions)) + [100, 102],       # node5 B1
    ]
    anc = real_anc + [[] for _ in range(T - len(real_anc))]  # pad rows empty

    mask = ancestor_block_mask(T, NB, CR, anc, dev)
    masked = apply_tree_mask(scores, mask)

    # reference: top-k over ancestor-restricted scores
    for i in range(6):
        blocks = sorted({p // CR for p in anc[i]})
        ref_scores = torch.full((NB,), float("-inf"), device=dev)
        ref_scores[blocks] = scores[i, blocks]
        k = 8
        got = torch.topk(masked[i], k).indices
        ref = torch.topk(ref_scores, k).indices
        assert set(got.tolist()) == set(ref.tolist()), f"row {i} mismatch"
    # padded rows: fully masked -> all -inf (selection yields nothing valid)
    assert torch.isinf(masked[6:]).all()
    print("PASS: masked top-k == ancestor-restricted top-k; padded rows fully masked")


if __name__ == "__main__":
    _test()
