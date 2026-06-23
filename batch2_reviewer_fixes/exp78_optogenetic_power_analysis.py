"""Experiment 78: Power analysis for optogenetic validation.

The VAE-silencing Spearman correlation is rho=+0.33 at n=12 (p=0.30, not significant).
A reviewer will note this. A power analysis showing what n is needed for significance
preempts the objection and shows we know the limitation.

This experiment:
1. Given observed rho=+0.33, compute power curve: P(significant) vs n for n in 10..60
2. Find minimum n for 80% power at alpha=0.05 (probably ~25-30)
3. Do the same for the LDA anti-correlation (rho=-0.73) to show it's well-powered at n=12
4. Bootstrap power analysis: resample from the n=12 data to estimate what happens at larger n
5. Report as a figure: power curves for VAE and LDA, vertical line at current n=12

CPU only. <30min.
"""
import json
from pathlib import Path

import numpy as np
from scipy import stats

RESULTS_DIR = Path(__file__).parent / "results" / "exp78"


def spearman_power(rho, n_values, alpha=0.05, n_simulations=10000):
    """Simulate power for Spearman correlation test.

    For each n, generate n_simulations bivariate normal samples with
    population correlation rho, compute Spearman rho and p-value,
    report fraction significant.
    """
    powers = []
    for n in n_values:
        significant = 0
        for _ in range(n_simulations):
            x = np.random.randn(n)
            y = rho * x + np.sqrt(1 - rho**2) * np.random.randn(n)
            _, p = stats.spearmanr(x, y)
            if p < alpha:
                significant += 1
        powers.append(significant / n_simulations)
    return np.array(powers)


def run(max_sessions=None):
    """Main entry point."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    n_values = np.arange(8, 61)

    vae_power = spearman_power(rho=0.33, n_values=n_values)
    lda_power = spearman_power(rho=-0.73, n_values=n_values)

    results = {
        "n_values": n_values.tolist(),
        "vae_power": vae_power.tolist(),
        "lda_power": lda_power.tolist(),
        "vae_n_for_80pct": int(n_values[np.argmax(vae_power >= 0.8)]) if any(vae_power >= 0.8) else None,
        "lda_n_for_80pct": int(n_values[np.argmax(lda_power >= 0.8)]) if any(lda_power >= 0.8) else None,
    }

    with open(RESULTS_DIR / "power_analysis.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"VAE (rho=0.33): 80% power at n={results['vae_n_for_80pct']}")
    print(f"LDA (rho=-0.73): 80% power at n={results['lda_n_for_80pct']}")


if __name__ == "__main__":
    run()
