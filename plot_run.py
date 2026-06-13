"""
Plot EvoX run1 results for the TriMul kernel optimization.
"""
import re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

LOG = "trimul/skydiscover_runs/run1.log"

BASELINE_US = 11358.266

# (iter, geomean_us, kind)  kind: "keep" | "discard" | "fail"
iterations = [
    (1,  9344.256,  "keep"),
    (3,  9356.484,  "discard"),
    (4,  9291.110,  "keep"),
    (5,  9329.417,  "discard"),
    (6,  9266.186,  "keep"),
    (7,  9302.989,  "discard"),
    (8,  9281.608,  "discard"),
    (10, 0.0,       "fail"),
    (9,  6572.176,  "keep"),
    (13, 6562.477,  "keep"),
    (14, 0.0,       "fail"),
    (17, 6352.167,  "keep"),
    (18, 6480.952,  "discard"),
    (21, 6428.371,  "discard"),
    (23, 6377.658,  "discard"),
]
iterations.sort(key=lambda x: x[0])

SEARCH_EVOLUTIONS = [8, 21]  # iter at which search strategy was evolved

# best-over-time step line
def best_steps(rows):
    bx, by = [], []
    best = float("inf")
    for it, t, k in sorted(rows, key=lambda x: x[0]):
        if k == "keep" and t > 0 and t < best:
            best = t
        if best < float("inf"):
            bx.append(it)
            by.append(best)
    return bx, by

bx, by = best_steps(iterations)
best_final = min(t for _, t, k in iterations if k == "keep" and t > 0)

# clip outliers for y axis
CLIP_US = BASELINE_US * 1.1
all_valid = [t for _, t, k in iterations if k != "fail" and t > 0]
y_floor = -(CLIP_US * 1.08)
y_ceil  = -(min(all_valid) * 0.82)

def ny(t):
    return max(-t, y_floor) if t > 0 else y_floor

fig, ax = plt.subplots(figsize=(13, 7))
fig.subplots_adjust(top=0.75)

# scatter: keep vs discard
keep_x  = [it for it, t, k in iterations if k == "keep"]
keep_y  = [ny(t)  for it, t, k in iterations if k == "keep"]
disc_x  = [it for it, t, k in iterations if k == "discard"]
disc_y  = [ny(t)  for it, t, k in iterations if k == "discard"]
fail_x  = [it for it, t, k in iterations if k == "fail"]

if keep_x:
    ax.scatter(keep_x, keep_y, c="#3b82f6", s=80, zorder=5,
               edgecolors="white", linewidths=0.6, label="keep (new best)")
if disc_x:
    ax.scatter(disc_x, disc_y, c="#93c5fd", s=45, zorder=4,
               edgecolors="white", linewidths=0.3, alpha=0.85, label="discard")
if fail_x:
    ax.scatter(fail_x, [y_floor] * len(fail_x), c="#fbbf24", s=50, zorder=3,
               marker="x", linewidths=1.8, label=f"fail ({len(fail_x)})", alpha=0.9)

# best-over-time step line
if bx:
    ax.step(bx, [-t for t in by], where="post", color="#1d4ed8", linewidth=2.2,
            label="best so far", zorder=6)

# baseline dashed line
ax.axhline(y=-BASELINE_US, color="#94a3b8", linewidth=1.4, linestyle="--",
           alpha=0.8, zorder=2, label=f"baseline ({BASELINE_US:,.0f} µs)")

# search strategy evolution markers
for i, ev_iter in enumerate(SEARCH_EVOLUTIONS):
    ax.axvline(x=ev_iter, color="#a855f7", linewidth=1.5, linestyle=":",
               alpha=0.75, zorder=2)
    ax.annotate(f"search\nevolved", xy=(ev_iter + 0.2, y_ceil * 0.97),
                fontsize=8, color="#7c3aed", va="top")

ax.set_ylim(y_floor * 1.05, y_ceil)
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{-v:,.0f}"))
ax.set_xlabel("Iteration #", fontsize=12)
ax.set_ylabel("Latency (µs, lower is better →)", fontsize=12)
ax.set_xticks(range(0, 26, 2))
ax.grid(True, alpha=0.3)

ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=4,
          framealpha=0.9, fontsize=10, borderaxespad=0)

improvement = (BASELINE_US - best_final) / BASELINE_US * 100
fig.text(0.5, 0.92,
         f"Best: {best_final:,.1f} µs  (−{improvement:.1f}% vs baseline {BASELINE_US:,.0f} µs)",
         ha="center", va="top", fontsize=11, fontweight="bold", color="#1e3a5f",
         bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="#3b82f6", alpha=0.9))

fig.text(0.5, 0.995, "EvoX — TriMul kernel optimization (25 iterations)",
         ha="center", va="top", fontsize=14, fontweight="bold")

ax.annotate(f"(failures shown at floor)",
            xy=(0.01, 0.02), xycoords="axes fraction",
            ha="left", va="bottom", fontsize=9, color="#6b7280",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="#d1d5db", alpha=0.8))

out = "trimul/skydiscover_runs/run1_plot.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved {out}")
print(f"Best: {best_final:.1f} µs  ({improvement:.1f}% improvement)")
