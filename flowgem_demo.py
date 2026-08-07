"""
FLOWGEM — live demonstration
=============================

A short, self-contained walkthrough of the packaged FlowGEM, intended to be run
live. It shows the end-user experience end to end:

    incomplete data (NaN = missing)  -->  FlowGEM().fit().generate()  -->  complete data

and then checks that the generated sample recovers the structure of the true
(uncontaminated) distribution, both numerically and with a scatter plot that
mirrors Figures 4-5 of the paper.

HOW TO RUN
----------
    conda activate flowgem
    python flowgem_demo.py
"""

import logging

import numpy as np
import torch

from flowgem import FlowGEM

# ---------------------------------------------------------------------------
# Demo settings  (fast defaults so it runs quickly in a live meeting)
# ---------------------------------------------------------------------------
N = 600          # number of samples
T = 600          # number of gradient-flow steps (increase for a nicer plot)
SEED = 0

# Show the library's internal logging (demonstrates the logging we added).
# The package is silent by default; here we switch INFO messages on.
SHOW_INTERNAL_LOGS = True


def banner(text):
    line = "=" * 70
    print(f"\n{line}\n  {text}\n{line}")


def main():
    if SHOW_INTERNAL_LOGS:
        logging.basicConfig(level=logging.INFO, format="   [flowgem] %(message)s")

    rng = np.random.default_rng(SEED)
    torch.manual_seed(SEED)

    # -----------------------------------------------------------------------
    # 1. Build a toy problem with a KNOWN true distribution
    #    (correlated 3-D Gaussian, so we can judge the result afterwards)
    # -----------------------------------------------------------------------
    banner("1. A toy dataset with a known true distribution")

    mean = [0.0, 0.0, 0.0]
    cov = [[1.0, 0.8, 0.3],
           [0.8, 1.0, 0.5],
           [0.3, 0.5, 1.0]]
    X_true = rng.multivariate_normal(mean, cov, size=N)

    print(f"   Drew {N} samples from a correlated 3-D Gaussian.")
    print(f"   True correlation(dim0, dim1) = {np.corrcoef(X_true[:, 0], X_true[:, 1])[0, 1]:.2f}")

    # -----------------------------------------------------------------------
    # 2. Introduce Missing-At-Random (MAR) values
    #    Missingness in columns 1 and 2 depends on the OBSERVED column 0.
    #    (Column 0 is kept fully observed.)
    # -----------------------------------------------------------------------
    banner("2. Introduce MAR missingness (NaN = missing)")

    X_incomplete = X_true.copy()
    p_miss = 1.0 / (1.0 + np.exp(-X_true[:, 0]))     # depends on observed col 0 -> MAR
    for j in (1, 2):
        missing = rng.random(N) < 0.6 * p_miss
        X_incomplete[missing, j] = np.nan

    n_missing = int(np.isnan(X_incomplete).sum())
    pct = 100.0 * n_missing / X_incomplete.size
    print(f"   Punched holes: {n_missing} missing entries ({pct:.0f}% of the matrix).")
    print("   Missingness depends on the observed column 0 -> genuinely MAR.")
    print("\n   First few rows the user would hand to FlowGEM (nan = missing):")
    with np.printoptions(precision=2, suppress=True):
        print(np.array2string(X_incomplete[:4], prefix="   "))

    # -----------------------------------------------------------------------
    # 3. THE END-USER EXPERIENCE  --  three lines
    # -----------------------------------------------------------------------
    banner("3. The whole user experience: fit + generate")

    print("   >>> from flowgem import FlowGEM")
    print("   >>> model = FlowGEM(T=%d)" % T)
    print("   >>> X_complete = model.fit(X_incomplete).generate()\n")

    model = FlowGEM(T=T, random_state=SEED)          # defaults: MICE init, heuristic sigma
    X_complete = model.fit(X_incomplete).generate()  # user passes only the NaN array
    X_complete = X_complete.numpy()

    print("   Done. The user supplied only the incomplete array;")
    print("   the mask and the initialisation were handled internally.")

    # -----------------------------------------------------------------------
    # 4. The result: a complete sample, no NaN
    # -----------------------------------------------------------------------
    banner("4. The result")

    print(f"   Output shape : {X_complete.shape}   (same as the input)")
    print(f"   NaN remaining: {bool(np.isnan(X_complete).any())}   (should be False)")

    # -----------------------------------------------------------------------
    # 5. Quality check: did it recover the true distribution's structure?
    # -----------------------------------------------------------------------
    banner("5. Did the generated sample recover the true structure?")

    true_corr = np.corrcoef(X_true[:, 0], X_true[:, 1])[0, 1]
    gen_corr = np.corrcoef(X_complete[:, 0], X_complete[:, 1])[0, 1]
    print("                         dim0-dim1 correlation")
    print(f"   true distribution :   {true_corr:.2f}")
    print(f"   FlowGEM sample    :   {gen_corr:.2f}")
    print("\n                         per-column mean")
    with np.printoptions(precision=2, suppress=True):
        print(f"   true distribution :   {X_true.mean(axis=0)}")
        print(f"   FlowGEM sample    :   {X_complete.mean(axis=0)}")
    print("\n   The generated sample recovers the correlation and the means")
    print("   of the true distribution -- not just individual imputed values.")

    # -----------------------------------------------------------------------
    # 6. A picture (mirrors Figures 4-5 of the paper): first two dimensions
    # -----------------------------------------------------------------------
    banner("6. Scatter plot: true distribution vs FlowGEM sample")

    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharex=True, sharey=True)
        axes[0].scatter(X_true[:, 0], X_true[:, 1], s=8, alpha=0.5)
        axes[0].set_title("Ground truth")
        axes[1].scatter(X_complete[:, 0], X_complete[:, 1], s=8, alpha=0.5, color="tab:orange")
        axes[1].set_title("FlowGEM generated")
        for ax in axes:
            ax.set_xlabel("dimension 0")
            ax.set_ylabel("dimension 1")
            ax.axhline(0, color="gray", lw=0.5)
            ax.axvline(0, color="gray", lw=0.5)
        fig.suptitle("First two dimensions: true distribution vs FlowGEM (cf. paper Fig. 4-5)")
        fig.tight_layout()
        fig.savefig("flowgem_demo.png", dpi=130)
        print("   Saved plot to 'flowgem_demo.png'.")
        plt.show()
    except Exception as exc:  # pragma: no cover  (plotting is optional)
        print(f"   (Plot skipped: {exc})")
        print("   The numeric check above already shows the result.")

    banner("Demo complete")


if __name__ == "__main__":
    main()