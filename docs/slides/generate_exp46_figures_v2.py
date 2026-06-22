"""Generate figures for exp46 cross-dataset replication slides (v2 with real Steinmetz scatter)."""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from pathlib import Path
from scipy.stats import spearmanr

RESULTS = Path(__file__).parent.parent.parent / "results" / "exp46" / "exp46_20260622_150700.json"
STEINMETZ = Path(__file__).parent.parent.parent / "results" / "exp22" / "exp22_20260621_061444.json"
FIGURES = Path(__file__).parent / "figures"
FIGURES.mkdir(exist_ok=True)

with open(RESULTS) as f:
    data = json.load(f)
with open(STEINMETZ) as f:
    st_raw = json.load(f)

PINK = "#E91E63"
BLUE = "#2196F3"
GREEN = "#4CAF50"
RED = "#D32F2F"
NAVY = "#1565C0"
NEUTRAL = "#9E9E9E"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 13,
    "figure.facecolor": "white",
})

# Extract Steinmetz CKA/Procrustes
st_rp = st_raw["region_profiles"]
st_regions = [r for r in sorted(st_rp.keys()) if "cka_mean" in st_rp[r]]
st_cka = [st_rp[r]["cka_mean"] for r in st_regions]
st_proc = [st_rp[r]["proc_mean"] for r in st_regions]
st_rho, st_p = spearmanr(st_cka, st_proc)

# Extract IBL CKA/Procrustes
pairs = data["cka_procrustes_anticorrelation"]["pairs"]
ibl_regions = [p["region"] for p in pairs]
ibl_cka = [p["cka_mean"] for p in pairs]
ibl_proc = [p["procrustes_mean"] for p in pairs]
ibl_n = [p["n_sessions"] for p in pairs]
ibl_cka_ci = [p["cka_std"] / np.sqrt(p["n_sessions"]) for p in pairs]
ibl_proc_ci = [p["procrustes_std"] / np.sqrt(p["n_sessions"]) for p in pairs]
ibl_rho = data["cka_procrustes_anticorrelation"]["anti_correlation"]["spearman_rho"]
ibl_p = data["cka_procrustes_anticorrelation"]["anti_correlation"]["p_value"]


# ── Figure 1: Side-by-side Steinmetz vs IBL anti-correlation (MAIN FIGURE) ──
rng = np.random.default_rng(0)

# Shared y-axis, independent x-axis (Steinmetz is denser, IBL extends further)
ylim = (min(st_proc + ibl_proc) - 0.05, 1.05)
st_xlim = (-0.02, max(st_cka) + 0.05)
ibl_xlim = (-0.02, max(ibl_cka) + 0.08)

fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))

# Left: Steinmetz
ax = axes[0]
ax.scatter(st_cka, st_proc, c=PINK, s=50, alpha=0.6, edgecolors="white", linewidth=0.5, zorder=5)
matched_set = set(data["matched_regions"])
for i, r in enumerate(st_regions):
    if r in matched_set:
        ax.scatter(st_cka[i], st_proc[i], c=PINK, s=100, alpha=0.9, edgecolors="black", linewidth=1.5, zorder=6)
        ax.annotate(r, (st_cka[i], st_proc[i]), fontsize=7, fontweight="bold",
                    xytext=(5, 4), textcoords="offset points")
z_st = np.polyfit(st_cka, st_proc, 1)
x_st = np.linspace(st_xlim[0], st_xlim[1], 100)
ax.plot(x_st, np.polyval(z_st, x_st), "--", color=PINK, alpha=0.7, linewidth=2)
ax.text(0.05, 0.05, f"ρ = {st_rho:.2f}\nn = {len(st_regions)} regions",
        transform=ax.transAxes, ha="left", va="bottom", fontsize=12, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.4", facecolor=PINK, alpha=0.15))

# Permutation test inset on Steinmetz panel
st_cka_arr = np.array(st_cka)
st_proc_arr = np.array(st_proc)
st_null_rhos = []
for _ in range(10000):
    perm = rng.permutation(len(st_cka_arr))
    st_null_rhos.append(spearmanr(st_cka_arr[perm], st_proc_arr).statistic)
st_null_rhos = np.array(st_null_rhos)
st_perm_p = np.mean(st_null_rhos <= st_rho)

st_inset = ax.inset_axes([0.68, 0.72, 0.3, 0.22])
st_inset.hist(st_null_rhos, bins=40, color=NEUTRAL, alpha=0.5, edgecolor="white", linewidth=0.5)
st_inset.axvline(st_rho, color=PINK, linewidth=2.5, label=f"observed\nρ = {st_rho:.2f}")
st_inset.set_xlabel("ρ (null)", fontsize=5)
st_inset.set_ylabel("", fontsize=5)
st_inset.set_title(f"p < {max(st_perm_p, 1/10000):.4f}", fontsize=6, fontweight="bold")
st_inset.tick_params(labelsize=5)
st_inset.legend(fontsize=5, loc="upper left")
st_inset.spines["top"].set_visible(False)
st_inset.spines["right"].set_visible(False)
ax.set_xlabel("CKA (kernel alignment)", fontweight="bold")
ax.set_ylabel("Procrustes (subspace alignment)", fontweight="bold")
ax.set_title("Steinmetz 2019 (original)", fontweight="bold", color=PINK)
ax.set_xlim(st_xlim)
ax.set_ylim(ylim)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# Right: IBL
ax = axes[1]
ax.errorbar(ibl_cka, ibl_proc, xerr=ibl_cka_ci, yerr=ibl_proc_ci,
            fmt="o", color=BLUE, markersize=10, capsize=4,
            markeredgecolor="white", markeredgewidth=1.5, zorder=5,
            ecolor=BLUE, alpha=0.6)
for i, r in enumerate(ibl_regions):
    ax.annotate(r, (ibl_cka[i], ibl_proc[i]), fontsize=9, fontweight="bold",
                xytext=(7, 4), textcoords="offset points")
z_ibl = np.polyfit(ibl_cka, ibl_proc, 1)
x_ibl = np.linspace(ibl_xlim[0], ibl_xlim[1], 100)
ax.plot(x_ibl, np.polyval(z_ibl, x_ibl), "--", color=BLUE, alpha=0.7, linewidth=2)
ax.set_xlabel("CKA (kernel alignment)", fontweight="bold")
ax.set_title("IBL 2022 (replication)", fontweight="bold", color=BLUE)
ax.set_xlim(ibl_xlim)
ax.set_ylim(ylim)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# Permutation test inset on IBL panel
ibl_cka_arr = np.array(ibl_cka)
ibl_proc_arr = np.array(ibl_proc)
null_rhos = []
for _ in range(10000):
    perm = rng.permutation(len(ibl_cka_arr))
    null_rhos.append(spearmanr(ibl_cka_arr[perm], ibl_proc_arr).statistic)
null_rhos = np.array(null_rhos)
perm_p = np.mean(null_rhos <= ibl_rho)

inset = ax.inset_axes([0.68, 0.72, 0.3, 0.22])
inset.hist(null_rhos, bins=40, color=NEUTRAL, alpha=0.5, edgecolor="white", linewidth=0.5)
inset.axvline(ibl_rho, color=BLUE, linewidth=2.5, label=f"observed\nρ = {ibl_rho:.2f}")
inset.set_xlabel("ρ (null)", fontsize=5)
inset.set_ylabel("", fontsize=5)
inset.set_title(f"p < {max(perm_p, 1/10000):.4f}", fontsize=6, fontweight="bold")
inset.tick_params(labelsize=5)
inset.legend(fontsize=5, loc="upper left")
inset.spines["top"].set_visible(False)
inset.spines["right"].set_visible(False)

ax.text(0.05, 0.05, f"ρ = {ibl_rho:.2f}\np = {ibl_p:.1e}\nn = {len(ibl_regions)} regions",
        transform=ax.transAxes, ha="left", va="bottom", fontsize=12, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.4", facecolor=BLUE, alpha=0.15))

fig.suptitle("CKA–Procrustes dissociation replicates across datasets", fontweight="bold", fontsize=15, y=1.02)
fig.tight_layout()
fig.savefig(FIGURES / "cross_dataset_anticorrelation.pdf", bbox_inches="tight")
fig.savefig(FIGURES / "cross_dataset_anticorrelation.png", bbox_inches="tight", dpi=200)
plt.close(fig)
print("✓ cross_dataset_anticorrelation (with real Steinmetz scatter)")


# ── Figure 2: CKA-Procrustes bars for IBL regions (sorted) ──
fig, ax = plt.subplots(figsize=(7, 5))
sorted_idx = np.argsort(ibl_cka)
regions_sorted = [ibl_regions[i] for i in sorted_idx]
cka_sorted = [ibl_cka[i] for i in sorted_idx]
proc_sorted = [ibl_proc[i] for i in sorted_idx]

y = np.arange(len(regions_sorted))
ax.barh(y - 0.18, cka_sorted, 0.35, color=RED, alpha=0.8, label="CKA", edgecolor="white")
ax.barh(y + 0.18, proc_sorted, 0.35, color=NAVY, alpha=0.8, label="Procrustes", edgecolor="white")
ax.set_yticks(y)
ax.set_yticklabels(regions_sorted, fontweight="bold")
ax.set_xlabel("Similarity score", fontweight="bold")
ax.set_title("IBL: CKA vs Procrustes by region", fontweight="bold")
ax.legend(frameon=False, fontsize=11, loc="lower right")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.set_xlim(0, 1.05)
fig.tight_layout()
fig.savefig(FIGURES / "ibl_cka_proc_bars.pdf", bbox_inches="tight")
fig.savefig(FIGURES / "ibl_cka_proc_bars.png", bbox_inches="tight", dpi=200)
plt.close(fig)
print("✓ ibl_cka_proc_bars")


# ── Figure 3: Classification transfer side-by-side ──
ct = data["classification_transfer"]
st_types = ct["steinmetz_types"]
ibl_types = ct["ibl_types"]

fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True)
sorted_regions = sorted(st_types.keys())

for ax_idx, (title, types, color) in enumerate([
    ("Steinmetz", st_types, PINK),
    ("IBL", ibl_types, BLUE),
]):
    ax = axes[ax_idx]
    colors = [RED if types[r] == "CKA-type" else NAVY for r in sorted_regions]
    y_pos = np.arange(len(sorted_regions))
    ax.barh(y_pos, [1]*len(sorted_regions), color=colors, alpha=0.7,
            edgecolor="white", linewidth=1.5, height=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(sorted_regions, fontweight="bold", fontsize=10)
    ax.set_title(title, fontweight="bold", color=color, fontsize=13)
    ax.set_xlim(0, 1.5)
    ax.set_xticks([])
    for i, r in enumerate(sorted_regions):
        label = "CKA" if types[r] == "CKA-type" else "Proc"
        ax.text(0.5, i, label, ha="center", va="center", fontsize=10, fontweight="bold", color="white")
    for i, r in enumerate(sorted_regions):
        if st_types.get(r) != ibl_types.get(r):
            ax.text(1.15, i, "✗", ha="center", va="center", fontsize=14, color="#F44336", fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)

legend_elements = [
    Patch(facecolor=RED, alpha=0.7, label="CKA-type (high dim)"),
    Patch(facecolor=NAVY, alpha=0.7, label="Procrustes-type (low dim)"),
]
fig.legend(handles=legend_elements, loc="lower center", ncol=2, frameon=False, fontsize=10, bbox_to_anchor=(0.5, -0.02))
n_agree = ct["n_agree"]
n_total = ct["n_regions"]
fig.suptitle(f"Region type classification: {n_agree}/{n_total} agree across datasets", fontweight="bold", fontsize=13)
fig.tight_layout()
fig.savefig(FIGURES / "cross_dataset_classification.pdf", bbox_inches="tight")
fig.savefig(FIGURES / "cross_dataset_classification.png", bbox_inches="tight", dpi=200)
plt.close(fig)
print("✓ cross_dataset_classification")


# ── Figure 4: Alpha comparison grouped bar chart ──
matched = data["matched_regions"]
st_alphas = [data["steinmetz_alphas"][r] for r in matched]
ibl_alphas_list = [data["ibl_alphas"][r] for r in matched]

fig, ax = plt.subplots(figsize=(8, 5))
x = np.arange(len(matched))
w = 0.35
ax.bar(x - w/2, st_alphas, w, color=PINK, alpha=0.8, label="Steinmetz", edgecolor="white", linewidth=0.5)
ax.bar(x + w/2, ibl_alphas_list, w, color=BLUE, alpha=0.8, label="IBL", edgecolor="white", linewidth=0.5)
ax.set_xticks(x)
ax.set_xticklabels(matched, rotation=45, ha="right", fontweight="bold")
ax.set_ylabel("Power-law exponent (α)", fontweight="bold")
ax.set_title("Power-law exponent does not transfer across datasets", fontweight="bold")
ax.legend(frameon=False, fontsize=11)
rho_alpha = data["alpha_correlation"]["rho"]
p_alpha = data["alpha_correlation"]["p_value"]
ax.text(0.95, 0.95, f"Cross-dataset ρ = {rho_alpha:.2f}\np = {p_alpha:.2f} (n.s.)",
        transform=ax.transAxes, ha="right", va="top", fontsize=11,
        bbox=dict(boxstyle="round,pad=0.4", facecolor=NEUTRAL, alpha=0.15))
ax.set_yscale("log")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
fig.tight_layout()
fig.savefig(FIGURES / "cross_dataset_alpha.pdf", bbox_inches="tight")
fig.savefig(FIGURES / "cross_dataset_alpha.png", bbox_inches="tight", dpi=200)
plt.close(fig)
print("✓ cross_dataset_alpha")


# ── Figure 5: Session variability boxplots ──
fig, ax = plt.subplots(figsize=(8, 5))
ibl_per_session = data["ibl_alpha_per_session"]
regions_alpha = sorted(ibl_per_session.keys())
bp_data = [ibl_per_session[r] for r in regions_alpha]
labels = [f"{r}\n(n={len(ibl_per_session[r])})" for r in regions_alpha]

bp = ax.boxplot(bp_data, widths=0.6, patch_artist=True,
                boxprops=dict(facecolor=BLUE, alpha=0.3),
                medianprops=dict(color=BLUE, linewidth=2),
                whiskerprops=dict(color=BLUE),
                capprops=dict(color=BLUE),
                flierprops=dict(markeredgecolor=BLUE, markersize=4))
rng = np.random.default_rng(42)
for i, vals in enumerate(bp_data):
    jitter = rng.uniform(-0.15, 0.15, len(vals))
    ax.scatter([i + 1 + j for j in jitter], vals, c=BLUE, s=30, alpha=0.6, zorder=5, edgecolors="white", linewidth=0.5)

ax.set_xticklabels(labels, fontsize=9, fontweight="bold")
ax.set_ylabel("Power-law exponent (α)", fontweight="bold")
ax.set_title("IBL session variability in power-law exponent", fontweight="bold")
ax.axhline(1.5, color=NEUTRAL, linestyle="--", alpha=0.5, label="α = 1.5 threshold")
ax.legend(frameon=False, fontsize=10)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
fig.tight_layout()
fig.savefig(FIGURES / "ibl_alpha_variability.pdf", bbox_inches="tight")
fig.savefig(FIGURES / "ibl_alpha_variability.png", bbox_inches="tight", dpi=200)
plt.close(fig)
print("✓ ibl_alpha_variability")


# ── Figure 6: Matched regions highlighted on both datasets ──
# Show which regions overlap and their CKA/Proc positions
fig, ax = plt.subplots(figsize=(8, 6))

# Steinmetz (faded)
ax.scatter(st_cka, st_proc, c=PINK, s=30, alpha=0.2, edgecolors="none", label=f"Steinmetz (n={len(st_regions)})")

# IBL (with error bars)
ax.errorbar(ibl_cka, ibl_proc, xerr=ibl_cka_ci, yerr=ibl_proc_ci,
            fmt="s", color=BLUE, markersize=9, capsize=4,
            markeredgecolor="white", markeredgewidth=1.5, zorder=5,
            ecolor=BLUE, alpha=0.6, label=f"IBL (n={len(ibl_regions)})")

# Steinmetz matched regions (highlighted, connected to IBL counterpart)
for r in data["matched_regions"]:
    if r in st_regions and r in ibl_regions:
        si = st_regions.index(r)
        ii = ibl_regions.index(r)
        ax.scatter(st_cka[si], st_proc[si], c=PINK, s=100, alpha=0.9,
                   edgecolors="black", linewidth=1.5, zorder=6)
        # Connect with arrow
        ax.annotate("", xy=(ibl_cka[ii], ibl_proc[ii]),
                     xytext=(st_cka[si], st_proc[si]),
                     arrowprops=dict(arrowstyle="->", color=NEUTRAL, alpha=0.4, lw=1.5))
        # Label at midpoint
        mx = (st_cka[si] + ibl_cka[ii]) / 2
        my = (st_proc[si] + ibl_proc[ii]) / 2
        ax.annotate(r, (mx, my), fontsize=7, fontweight="bold", ha="center", color="#333333")

ax.set_xlabel("CKA", fontweight="bold")
ax.set_ylabel("Procrustes", fontweight="bold")
ax.set_title("Matched regions across Steinmetz and IBL", fontweight="bold")
ax.legend(frameon=False, fontsize=10)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
fig.tight_layout()
fig.savefig(FIGURES / "matched_regions_overlay.pdf", bbox_inches="tight")
fig.savefig(FIGURES / "matched_regions_overlay.png", bbox_inches="tight", dpi=200)
plt.close(fig)
print("✓ matched_regions_overlay")

print("\nAll 6 figures generated!")
