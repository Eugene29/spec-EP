# visualization/visualize_expert_routing.py
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

ap = argparse.ArgumentParser()
ap.add_argument("files", nargs="+", type=Path, help="experts_rank*.pt, each [L, T, K]")
ap.add_argument("--num_experts", type=int, default=256)
ap.add_argument("--num_gpus", type=int, default=8)
ap.add_argument("--row_h", type=float, default=1.8, help="inches of height per layer")
args = ap.parse_args()

E = torch.cat([torch.load(f).long() for f in args.files], dim=1)   # [L, T_total, K]
L = E.shape[0]
experts_per_gpu = args.num_experts // args.num_gpus

fig, axes = plt.subplots(
    L, 4, figsize=(20, args.row_h * L), squeeze=False,
    gridspec_kw={"width_ratios": [4, 4, 1.5, 1.5]},
)
col_titles = ["tokens per expert", "experts, sorted", "tokens per GPU", "GPUs, sorted"]

for l in range(L):
    ids = E[l].flatten()  # expert id per (token, pick)
    expert_hist = ids.bincount(minlength=args.num_experts)
    gpu_hist = (ids // experts_per_gpu).bincount(minlength=args.num_gpus)

    panels = [
        (expert_hist, args.num_experts, 1.0),
        (expert_hist.sort(descending=True).values, args.num_experts, 1.0),
        (gpu_hist, args.num_gpus, 0.8),
        (gpu_hist.sort(descending=True).values, args.num_gpus, 0.8),
    ]
    for c, (h, n, w) in enumerate(panels):
        ax = axes[l, c]
        ax.bar(range(n), h, width=w)
        ax.axhline(h.float().mean(), color="red", lw=0.8, ls="--")   # perfectly balanced level
        ax.tick_params(labelsize=6)
        if c >= 2:
            ax.set_xticks(range(n))
    # each histogram and its sorted twin share a y-range, so the skew is comparable
    for c0 in (0, 2):
        top = max(axes[l, c0].get_ylim()[1], axes[l, c0 + 1].get_ylim()[1])
        axes[l, c0].set_ylim(0, top)
        axes[l, c0 + 1].set_ylim(0, top)
    axes[l, 0].set_ylabel(f"L{l}", fontsize=8, rotation=0, labelpad=14, va="center")

for c, t in enumerate(col_titles):
    axes[0, c].set_title(t, fontsize=9)
axes[-1, 0].set_xlabel("expert id", fontsize=7)
axes[-1, 1].set_xlabel("rank by load", fontsize=7)
axes[-1, 2].set_xlabel("GPU", fontsize=7)
axes[-1, 3].set_xlabel("rank by load", fontsize=7)

fig.suptitle(f"Routing distribution ({len(args.files)} rank file(s), "
             f"{E.shape[1]} tokens, top-{E.shape[2]}); red dashed = balanced level", y=1.0)
fig.tight_layout()

out = args.files[0].with_suffix(".png")
fig.savefig(out, dpi=120)
print(f"saved {out}")