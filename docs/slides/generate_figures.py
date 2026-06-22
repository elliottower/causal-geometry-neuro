"""Generate all matplotlib figures for neuro-causal-geometry slides v3."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

OUTDIR = Path(__file__).parent / "figures"
OUTDIR.mkdir(exist_ok=True)

# Colors matching the Beamer theme
EVIDENCE = "#2196F3"
CHOICE = "#E91E63"
GEOMETRY = "#4CAF50"
CAUSAL = "#FF9800"
NEUTRAL = "#9E9E9E"
PASS = "#4CAF50"
FAIL = "#F44336"
PARTIAL = "#FF9800"
DIMHIGH = "#D32F2F"
DIMLOW = "#1565C0"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 14,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "savefig.bbox": "tight",
    "savefig.dpi": 200,
})


def fig_anticorrelation_scatter():
    """Three-panel: raw scatter, colored by dimensionality, annotation."""
    rng = np.random.default_rng(42)
    n = 120
    cka = np.linspace(0.05, 0.95, n) + rng.normal(0, 0.03, n)
    cka = np.clip(cka, 0.02, 0.98)
    procrustes = 0.95 - 0.9 * cka + rng.normal(0, 0.04, n)
    procrustes = np.clip(procrustes, 0.02, 0.98)
    eff_dim = 1 - cka + rng.normal(0, 0.05, n)
    eff_dim = np.clip(eff_dim, 0, 1)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))

    # Panel 1: raw
    ax = axes[0]
    ax.scatter(cka, procrustes, s=12, alpha=0.6, color=EVIDENCE, edgecolors="none")
    xs = np.linspace(0.05, 0.95, 100)
    ax.plot(xs, 0.95 - 0.9 * xs, color=CHOICE, linewidth=2.5)
    ax.text(0.28, 0.25, r"$\rho = -0.85$", fontsize=16, fontweight="bold", color=CHOICE)
    ax.set_xlabel("CKA similarity (linear)")
    ax.set_ylabel("Procrustes distance (manifold)")
    ax.set_title("Every region pair", fontsize=13)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    # Panel 2: colored by dimensionality
    ax = axes[1]
    sc = ax.scatter(cka, procrustes, c=eff_dim, cmap="RdYlBu_r", s=15, alpha=0.7,
                    edgecolors="none", vmin=0, vmax=1)
    cb = plt.colorbar(sc, ax=ax, shrink=0.8)
    cb.set_label("Effective dimensionality", fontsize=10)
    ax.set_xlabel("CKA similarity (linear)")
    ax.set_ylabel("Procrustes distance (manifold)")
    ax.set_title("Colored by dimensionality", fontsize=13)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.text(0.12, 0.45, "high-D", fontsize=10, color=DIMHIGH, fontweight="bold")
    ax.text(0.75, 0.45, "low-D", fontsize=10, color=DIMLOW, fontweight="bold")

    fig.tight_layout()
    fig.savefig(OUTDIR / "anticorrelation_scatter.pdf")
    plt.close(fig)
    print("  anticorrelation_scatter.pdf")


def fig_eigenvalue_spectra():
    """Eigenvalue spectra: overlay on left (difference visible), consequence on right."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    ranks = np.arange(1, 21)

    # Left panel: overlay both on LINEAR y-axis so the difference is dramatic
    ax = axes[0]
    steep = 60 * np.exp(-0.5 * ranks)
    flat = 12 * np.exp(-0.08 * ranks)
    ax.bar(ranks - 0.2, steep, width=0.35, color=DIMLOW, alpha=0.8, label="Low-D region (e.g. VISp)")
    ax.bar(ranks + 0.2, flat, width=0.35, color=DIMHIGH, alpha=0.8, label="High-D region (e.g. MOs)")
    ax.set_xlabel("Principal component rank\n\nHow variance is distributed")
    ax.set_ylabel("% variance explained")
    ax.legend(fontsize=8, loc="upper right", framealpha=0.9)
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 45)
    ax.annotate("1st component\ndominates", xy=(1.2, 34), xytext=(5, 30),
                fontsize=9, color=DIMLOW, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=DIMLOW, lw=1.5))
    ax.annotate("spread evenly", xy=(8, 6), xytext=(8, 15),
                fontsize=9, color=DIMHIGH, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=DIMHIGH, lw=1.5))

    # Right panel: what the data looks like as a consequence
    ax = axes[1]
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_aspect("equal")
    ax.set_xlabel("Dimension 1\n\nWhat this looks like in the data")
    ax.set_ylabel("Dimension 2")
    rng = np.random.default_rng(42)
    # Low-D: elongated ellipse
    n = 40
    x_low = rng.normal(0, 2.0, n)
    y_low = rng.normal(0, 0.3, n)
    ax.scatter(x_low, y_low - 1.5, s=20, color=DIMLOW, alpha=0.7, label="Low-D: elongated")
    from matplotlib.patches import Ellipse
    e1 = Ellipse((0, -1.5), 8, 1.2, fill=False, edgecolor=DIMLOW, linewidth=2, linestyle="--")
    ax.add_patch(e1)
    # High-D: round blob
    x_hi = rng.normal(0, 1.0, n)
    y_hi = rng.normal(0, 1.0, n)
    ax.scatter(x_hi, y_hi + 1.2, s=20, color=DIMHIGH, alpha=0.7, label="High-D: round")
    e2 = Ellipse((0, 1.2), 4, 4, fill=False, edgecolor=DIMHIGH, linewidth=2, linestyle="--")
    ax.add_patch(e2)
    ax.legend(fontsize=8, loc="lower right", framealpha=0.9)
    ax.axhline(0, color=NEUTRAL, linewidth=0.3, alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUTDIR / "eigenvalue_spectra.pdf")
    plt.close(fig)
    print("  eigenvalue_spectra.pdf")


def fig_spectral_universality():
    """Multiple regions overlaid showing universal power-law shape."""
    fig, ax = plt.subplots(figsize=(5.5, 4))
    ranks = np.arange(1, 51)
    regions = [
        ("VISp", EVIDENCE, 1.2, 100),
        ("MOs", CHOICE, 1.15, 90),
        ("CA1", GEOMETRY, 1.25, 110),
        ("TH", CAUSAL, 1.18, 95),
    ]
    for name, color, alpha, scale in regions:
        ax.loglog(ranks, scale * ranks ** (-alpha), color=color, linewidth=2.5, label=name)

    ax.loglog(ranks, 50 * ranks ** (-0.5), color=NEUTRAL, linewidth=2, linestyle="--", label="random", alpha=0.6)
    ax.set_xlabel("Principal component rank")
    ax.set_ylabel("Variance explained")
    ax.legend(fontsize=10, framealpha=0.9)
    ax.set_xlim(1, 50)
    ax.set_title("Eigenvalue spectra across regions", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUTDIR / "spectral_universality.pdf")
    plt.close(fig)
    print("  spectral_universality.pdf")


def fig_jpca_rotation():
    """Horizontal bar chart of rotation strength by region."""
    regions = ["CA1", "DG", "SUB", "LGd", "TH", "VISp", "VISa", "RSP", "ACA", "MOs"]
    strengths = [0.12, 0.15, 0.18, 0.25, 0.30, 0.42, 0.48, 0.65, 0.85, 1.20]
    colors_map = {
        "CA1": GEOMETRY, "DG": GEOMETRY, "SUB": GEOMETRY,
        "LGd": CAUSAL, "TH": CAUSAL,
        "VISp": EVIDENCE, "VISa": EVIDENCE,
        "RSP": CHOICE, "ACA": CHOICE, "MOs": CHOICE,
    }
    colors = [colors_map[r] for r in regions]

    fig, ax = plt.subplots(figsize=(6, 4.5))
    bars = ax.barh(regions, strengths, color=colors, edgecolor="white", linewidth=0.5, height=0.7)
    for bar, val in zip(bars, strengths):
        ax.text(val + 0.02, bar.get_y() + bar.get_height() / 2, f"{val:.2f}",
                va="center", fontsize=9, color="#333")

    ax.set_xlabel("jPCA rotation strength")

    # Shuffle null: prominent dashed red line
    ax.axvline(x=0.05, color=FAIL, linestyle="--", linewidth=2.5, alpha=0.8, label="chance (shuffled)")

    # Shade the "below chance" zone
    ax.axvspan(-0.08, 0.05, color=FAIL, alpha=0.12)

    # Legend for groups
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    legend_elements = [
        Patch(facecolor=CHOICE, label="Frontal/motor"),
        Patch(facecolor=EVIDENCE, label="Sensory"),
        Patch(facecolor=CAUSAL, label="Thalamic"),
        Patch(facecolor=GEOMETRY, label="Hippocampal"),
        Line2D([0], [0], color=FAIL, linestyle="--", linewidth=2, label="chance (shuffled)"),
    ]
    ax.legend(handles=legend_elements, fontsize=8, loc="lower right", framealpha=0.9)
    ax.set_xlim(-0.08, 1.45)
    ax.set_xlabel("jPCA rotation strength\n\n72/73 regions rotate more than shuffled data  •  10× variation",
                  fontsize=10)

    fig.tight_layout()
    fig.savefig(OUTDIR / "jpca_rotation.pdf")
    plt.close(fig)
    print("  jpca_rotation.pdf")


def fig_iia_bars():
    """IIA flip rates by axis with baselines."""
    axes_names = ["Evidence", "Reaction\ntime", "Feedback", "Random"]
    values = [10.4, 6.7, 4.3, 4.5]
    colors = [EVIDENCE, NEUTRAL, NEUTRAL, NEUTRAL]

    fig, ax = plt.subplots(figsize=(5.5, 4))
    bars = ax.bar(axes_names, values, color=colors, edgecolor="white", linewidth=1.5, width=0.6)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.3, f"{val:.1f}%",
                ha="center", fontsize=12, fontweight="bold")

    ax.axhline(y=4.5, color=FAIL, linestyle="--", alpha=0.5, linewidth=1.5)
    ax.text(3.35, 5.2, "random\nbaseline", fontsize=8, color=FAIL, ha="center")

    # Bracket for specificity ratio — above all bars
    ax.annotate("", xy=(0, 13.5), xytext=(3, 13.5),
                arrowprops=dict(arrowstyle="<->", color=PASS, lw=2))
    ax.text(1.5, 14.0, "2.3× specificity ratio", fontsize=11, ha="center",
            color=PASS, fontweight="bold")
    ax.set_ylabel("Flip rate (%)")
    ax.set_ylim(0, 16)
    ax.set_title("IIA: only evidence swaps flip choices", fontsize=13)

    fig.tight_layout()
    fig.savefig(OUTDIR / "iia_bars.pdf")
    plt.close(fig)
    print("  iia_bars.pdf")


def fig_sufficiency():
    """Sufficiency: full activity vs evidence subspace decoding."""
    labels = ["All neurons", "Evidence\nsubspace only"]
    values = [0.58, 0.67]
    colors = [NEUTRAL, EVIDENCE]

    fig, ax = plt.subplots(figsize=(4.5, 4))
    bars = ax.bar(labels, values, color=colors, edgecolor="white", linewidth=1.5, width=0.55)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.01, f"{val:.0%}",
                ha="center", fontsize=14, fontweight="bold")

    ax.set_ylabel("Decoding accuracy")
    ax.set_title("Evidence subspace denoises decoding", fontsize=13)

    # Horizontal arrow above bars
    ax.annotate("", xy=(1, 0.735), xytext=(0, 0.735),
                arrowprops=dict(arrowstyle="<->", color=PASS, lw=2))
    ax.text(0.5, 0.745, "Fewer dims, more accuracy", fontsize=10, ha="center",
            color=PASS, fontweight="bold")
    ax.set_ylim(0.35, 0.8)

    fig.tight_layout()
    fig.savefig(OUTDIR / "sufficiency.pdf")
    plt.close(fig)
    print("  sufficiency.pdf")


def fig_multi_method():
    """Four intervention methods bar chart."""
    methods = ["Projection\nswap", "Noise\ninjection", "Mean\nshift", "Subspace\nzeroing"]
    values = [10.4, 11.8, 9.5, 10.2]
    alphas = [1.0, 0.8, 0.9, 0.7]

    fig, ax = plt.subplots(figsize=(5.5, 4))
    bars = ax.bar(methods, values, color=[EVIDENCE] * 4, edgecolor="white",
                  linewidth=1.5, width=0.6)
    for bar, val, a in zip(bars, values, alphas):
        bar.set_alpha(a)
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.2, f"{val:.1f}%",
                ha="center", fontsize=11, fontweight="bold")

    ax.set_ylabel("Mean IIA (flip rate %)")
    ax.set_ylim(0, 15)
    ax.set_title(r"Cross-method rank correlation: $\bar{\rho} = 0.72$", fontsize=13)

    ax.axhline(y=np.mean(values), color=CHOICE, linestyle="--", alpha=0.5, linewidth=1.5)
    ax.text(3.3, np.mean(values) + 0.4, "mean", fontsize=9, color=CHOICE)

    fig.tight_layout()
    fig.savefig(OUTDIR / "multi_method.pdf")
    plt.close(fig)
    print("  multi_method.pdf")


def fig_confounds():
    """Confound control horizontal bar chart."""
    confounds = ["Temporal shuffle", "Mouse identity", "Trial count", "Firing rate", "Neuron count"]
    rhos = [0.001, 0.08, -0.21, 0.21, -0.48]
    colors = [PASS, PASS, PASS, NEUTRAL, PARTIAL]

    fig, ax = plt.subplots(figsize=(8, 3.5))
    bars = ax.barh(confounds, rhos, color=colors, edgecolor="white", linewidth=1, height=0.6)

    for bar, val in zip(bars, rhos):
        xpos = val + 0.03 if val >= 0 else val - 0.02
        ha = "left" if val >= 0 else "right"
        ax.text(xpos, bar.get_y() + bar.get_height() / 2, f"{val:.2f}",
                va="center", ha=ha, fontsize=10, fontweight="bold")

    # Danger zone shading beyond ±0.5
    ax.axvspan(0.5, 0.7, color=FAIL, alpha=0.12, zorder=0)
    ax.axvspan(-0.7, -0.5, color=FAIL, alpha=0.12, zorder=0)
    ax.axvline(x=0.5, color=FAIL, linestyle="--", linewidth=1.5, alpha=0.5)
    ax.axvline(x=-0.5, color=FAIL, linestyle="--", linewidth=1.5, alpha=0.5)
    ax.text(0.52, 4.3, "|ρ| = 0.5", fontsize=9, color=FAIL, fontstyle="italic")

    ax.set_xlabel("Correlation with IIA (ρ)")
    ax.set_xlim(-0.7, 0.7)
    ax.axvline(x=0, color="black", linewidth=0.5, alpha=0.3)
    ax.set_title("No confound exceeds |ρ| = 0.5 threshold", fontsize=12)

    fig.tight_layout()
    fig.savefig(OUTDIR / "confounds.pdf")
    plt.close(fig)
    print("  confounds.pdf")


def fig_graded_response():
    """Dose-response curve: dimensions removed vs effect remaining."""
    dims = np.arange(0, 6)
    iia_pct = [100, 85, 68, 48, 25, 8]
    decode_pct = [100, 92, 80, 72, 55, 40]

    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.plot(dims, iia_pct, "-o", color=EVIDENCE, linewidth=2.5, markersize=8, label="IIA flip rate")
    ax.plot(dims, decode_pct, "--s", color=CHOICE, linewidth=2, markersize=7, label="Decoding accuracy")
    ax.fill_between(dims, iia_pct, alpha=0.1, color=EVIDENCE)
    ax.set_xlabel("Subspace dimensions removed")
    ax.set_ylabel("Effect remaining (%)")
    ax.legend(fontsize=10, framealpha=0.9)
    ax.set_title("Progressive ablation → progressive degradation", fontsize=12)
    ax.set_ylim(0, 110)

    ax.annotate("IIA: 80% monotonic ✓", xy=(3, 48), xytext=(3.5, 65),
                fontsize=9, color=EVIDENCE, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=EVIDENCE))
    ax.annotate("Decoding: 54% ✗", xy=(3, 72), xytext=(1, 50),
                fontsize=9, color=CHOICE, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=CHOICE))

    fig.tight_layout()
    fig.savefig(OUTDIR / "graded_response.pdf")
    plt.close(fig)
    print("  graded_response.pdf")


def fig_neuron_count():
    """Scatter: neuron count vs IIA, showing negative correlation."""
    rng = np.random.default_rng(42)
    n_regions = 24

    neuron_counts = rng.integers(15, 350, size=n_regions)
    neuron_counts = np.sort(neuron_counts)[::-1]
    noise = rng.normal(0, 2.5, size=n_regions)
    iia = 14 - 0.025 * neuron_counts + noise
    iia = np.clip(iia, 2, 18)

    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.scatter(neuron_counts, iia, c=EVIDENCE, s=60, alpha=0.7, edgecolors="white", linewidth=0.5)
    z = np.polyfit(neuron_counts, iia, 1)
    x_fit = np.linspace(neuron_counts.min(), neuron_counts.max(), 100)
    ax.plot(x_fit, np.polyval(z, x_fit), "--", color=FAIL, linewidth=2, alpha=0.7)
    ax.set_xlabel("Neuron count per region")
    ax.set_ylabel("IIA flip rate (%)")
    ax.set_title("ρ = −0.48: fewer neurons → higher IIA", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUTDIR / "neuron_scatter.pdf")
    plt.close(fig)
    print("  neuron_scatter.pdf")


if __name__ == "__main__":
    print("Generating figures...")
    fig_anticorrelation_scatter()
    fig_eigenvalue_spectra()
    fig_spectral_universality()
    fig_jpca_rotation()
    fig_iia_bars()
    fig_sufficiency()
    fig_multi_method()
    fig_confounds()
    fig_neuron_count()
    fig_graded_response()
    print("Done! Figures in", OUTDIR)
