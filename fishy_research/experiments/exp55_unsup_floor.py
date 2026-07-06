"""EXP-55 — UNSUPERVISED k from the spectral signal-vs-noise floor (NO labels), then GT stripe check.

Pick k (low-frequency GFT modes kept per channel) using ONLY the unlabeled coefficients, then
score stripe with GT to ask: does the unsupervised floor capture stripe or cut it off too early?

WHAT THE DATA ACTUALLY LOOKS LIKE (measured, not assumed)
  The raw signed GFT coefficients are NOT scale-normalized per mode: a few modes carry energy
  ~1000x the rest, with NO clean low->high frequency ordering of magnitude. So absolute
  cross-specimen std is dominated by arbitrary per-mode energy. We therefore measure the
  across-specimen variation *relative to a per-mode noise null*.

UNSUPERVISED ESTIMATORS (all label-free)
  (A) Sign-permuted / phase-randomized NULL on energy-normalized coeffs (c/rms). For each mode we
      compare the real cross-specimen std to the std under per-entry random sign flips (which
      destroys cross-specimen coherence but preserves magnitude). |z| = how many null-std units the
      real std deviates -> a per-mode STRUCTURE score. Noise floor = median |z| over the high-mode
      plateau. k_floor = last mode whose structure stays above floor for a sustained window.
  (B) Eigenvalue ELBOW of the Laplacian spectrum (Kneedle: max distance from chord).
  (C) PCA SCREE: smallest #components of the standardized full descriptor reaching 95% variance
      (the conventional unsupervised dimensionality cut), reported also as modes/channel.

THEN (GT, scoring only): stripe 5-fold logistic accuracy at the unsupervised k vs k=40.

Run:  .venv/bin/python experiments/exp55_unsup_floor.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
RES = HERE.parent / "results" / "exp55"
RES.mkdir(parents=True, exist_ok=True)
RNG = np.random.default_rng(0)

KMAX = 300
PLATEAU_LO = 240        # modes >= this define the high-frequency "noise plateau"
N_NULL = 1000
WIN = 8                 # sustained-window length for the floor cutoff


def load():
    d = np.load(RES / "cache.npz", allow_pickle=True)
    C = {"L": d["CL"].astype(np.float64), "a": d["Ca"].astype(np.float64), "b": d["Cb"].astype(np.float64)}
    return C, d["eigvals"].astype(np.float64), d["F"].astype(int)


def structure_z(C, n=N_NULL, rng=RNG):
    """Per-mode structure score: |z| of energy-normalized cross-specimen std vs a sign-flip null.
    Returns (real_std, |z|)."""
    rms = np.sqrt((C ** 2).mean(0)) + 1e-12
    Cn = C / rms
    real = Cn.std(0, ddof=1)
    A = np.abs(Cn)
    N, K = Cn.shape
    nl = np.empty((n, K))
    for i in range(n):
        s = rng.integers(0, 2, size=(N, K)) * 2 - 1
        nl[i] = (A * s).std(0, ddof=1)
    z = np.abs(real - nl.mean(0)) / (nl.std(0) + 1e-12)
    return real, z


def k_from_floor(z, floor):
    """k = (last mode m such that the window [m-WIN+1, m] is entirely above floor) + 1.
    i.e. the deepest sustained run of above-floor structure. Returns 0 if none."""
    above = z > floor
    last = 0
    for m in range(WIN - 1, len(z)):
        if above[m - WIN + 1:m + 1].all():
            last = m + 1
    return last


def eigen_elbow(eigvals):
    x = np.arange(len(eigvals), dtype=float)
    y = eigvals.astype(float)
    xn = (x - x.min()) / (x.max() - x.min())
    yn = (y - y.min()) / (y.max() - y.min())
    d = np.abs((yn[-1] - yn[0]) * xn - (xn[-1] - xn[0]) * yn + xn[-1] * yn[0] - yn[-1] * xn[0])
    d /= np.hypot(yn[-1] - yn[0], xn[-1] - xn[0])
    return int(np.argmax(d))


def pca_scree(C, frac=0.95):
    X = StandardScaler().fit_transform(np.concatenate([C[ch] for ch in ("L", "a", "b")], 1))
    ev = PCA().fit(X).explained_variance_ratio_.cumsum()
    return int(np.searchsorted(ev, frac) + 1)


def cv(X, y):
    return float(cross_val_score(LogisticRegression(max_iter=3000),
                                 StandardScaler().fit_transform(X), y, cv=5).mean())


def main():
    C, eigvals, F = load()
    chans = ("L", "a", "b")
    stripe = F[:, 2]

    real, z, floor, kf = {}, {}, {}, {}
    for ch in chans:
        r, zz = structure_z(C[ch])
        real[ch], z[ch] = r, zz
        floor[ch] = float(np.median(zz[PLATEAU_LO:]))
        kf[ch] = k_from_floor(zz, floor[ch])

    k_elbow = eigen_elbow(eigvals)
    k_pca = pca_scree(C)

    print("=== UNSUPERVISED k selection (NO labels) ===")
    print(f"(A) sign-null structure floor (median |z| over modes>={PLATEAU_LO}):")
    for ch in chans:
        hi = int((z[ch] > floor[ch]).sum())
        print(f"    {ch}: floor|z|={floor[ch]:5.1f}  modes-above-floor={hi:3d}  "
              f"sustained-run k_floor={kf[ch]}")
    print(f"(B) eigenvalue elbow (Kneedle): k_elbow={k_elbow}")
    print(f"(C) PCA scree 95% var: {k_pca} components (~{k_pca/3:.0f} modes/channel equiv)")

    k_L = kf["L"]
    k_floor_cons = max(kf.values())
    print(f"\nNO-LABEL pick for L channel: k_L = {k_L}")
    print(f"NO-LABEL consensus floor k (max over chan) = {k_floor_cons}")
    print(f"NO-LABEL practical pick (PCA-scree modes/chan) = {k_pca//3}")

    def desc(k):
        return np.concatenate([C[ch][:, :max(k, 1)] for ch in chans], 1)

    picks = {
        "k_L(floor)": k_L,
        "k_floor_consensus": k_floor_cons,
        "k_elbow": k_elbow,
        "k_pca_per_chan": k_pca // 3,
        "k=40": 40,
    }
    accs = {name: cv(desc(k), stripe) for name, k in picks.items()}

    print("\n=== GT scoring (stripe 5-fold logistic acc) — scoring ONLY ===")
    for name, k in picks.items():
        print(f"  {name:<20s} k={k:<4d} stripe acc = {accs[name]:.3f}")
    a_floor, a40 = accs["k_floor_consensus"], accs["k=40"]
    verdict = "CAPTURES stripe (does NOT cut it off)" if a_floor >= a40 - 0.02 else "CUTS stripe off too early"
    print(f"\n  VERDICT: unsupervised floor {verdict}  (floor acc {a_floor:.3f} vs k=40 acc {a40:.3f})")
    a_pca = accs["k_pca_per_chan"]
    print(f"  (PCA-scree k={k_pca//3}/chan -> stripe {a_pca:.3f}; "
          f"{'still captures' if a_pca >= a40 - 0.03 else 'undershoots'} stripe)")

    ks_curve = [1, 2, 3, 5, 8, 12, 16, 20, 30, 40, 60, 100, 150, 200, 300]
    stripe_curve = [cv(desc(k), stripe) for k in ks_curve]

    # ================= PLOT =================
    fig = plt.figure(figsize=(22, 9))
    cm = {"L": "#333333", "a": "#e15759", "b": "#4e79a7"}
    # row 1: per-mode structure vs floor for L,a,b
    for j, ch in enumerate(chans):
        ax = fig.add_subplot(2, 3, j + 1)
        m = np.arange(KMAX)
        ax.plot(m, z[ch], color=cm[ch], lw=1.0, label="|z| cross-spec std vs sign-null")
        ax.axhline(floor[ch], color="#59a14f", ls=":", lw=1.4,
                   label=f"hi-mode noise floor |z|={floor[ch]:.0f}")
        above = z[ch] > floor[ch]
        ax.scatter(m[above], z[ch][above], s=8, color="#e15759", alpha=.5, zorder=4,
                   label="above floor (structure)")
        ax.axvline(kf[ch], color="#f28e2b", lw=2.2, label=f"sustained-run k={kf[ch]}")
        ax.axvline(40, color="#9c755f", lw=1.0, ls="--", alpha=.7, label="k=40 ref")
        ax.set_title(f"channel {ch}: per-mode structure (signal) vs noise floor")
        ax.set_xlabel("GFT mode index m  (0 = lowest spatial freq)")
        ax.set_ylabel("structure score |z|")
        ax.set_xlim(0, 300)
        ax.set_yscale("symlog")
        ax.grid(alpha=.2)
        ax.legend(fontsize=6.5, loc="upper right")

    # row 2 left: raw energy per mode (shows scale spikes, no freq ordering)
    axS = fig.add_subplot(2, 3, 4)
    for ch in chans:
        axS.plot(np.sqrt((C[ch] ** 2).mean(0)), color=cm[ch], lw=0.9, label=f"{ch} RMS energy")
    axS.set_yscale("log")
    axS.set_xlabel("GFT mode index m")
    axS.set_ylabel("per-mode RMS energy (log)")
    axS.set_title("per-mode energy: spikes, NO low->high ordering\n(why abs std is uninformative)")
    axS.grid(alpha=.2)
    axS.legend(fontsize=7)

    # row 2 mid: eigenvalue spectrum + elbow
    axV = fig.add_subplot(2, 3, 5)
    axV.plot(eigvals, color="#4e79a7", lw=1.3, label="Laplacian eigenvalues")
    axV.axvline(k_elbow, color="#4e79a7", ls="-.", lw=1.6, label=f"elbow k={k_elbow}")
    axV.axvline(k_pca // 3, color="#59a14f", ls=":", lw=1.6, label=f"PCA scree k={k_pca//3}/ch")
    axV.axvline(40, color="#9c755f", ls="--", lw=1.0, label="k=40")
    axV.set_xlabel("mode index m")
    axV.set_ylabel("eigenvalue (graph frequency)")
    axV.set_title("eigenvalue spectrum + elbow")
    axV.grid(alpha=.2)
    axV.legend(fontsize=7, loc="upper left")

    # row 2 right: stripe accuracy vs k with all unsupervised picks
    axE = fig.add_subplot(2, 3, 6)
    axE.plot(ks_curve, stripe_curve, "o-", color="#e15759", label="stripe logistic acc")
    base = 1 - stripe.mean()
    axE.axhline(base, color="#888", ls=":", label=f"majority {base:.2f}")
    axE.axvline(k_pca // 3, color="#59a14f", lw=1.6, ls=":", label=f"PCA scree k={k_pca//3}")
    axE.axvline(k_elbow, color="#4e79a7", lw=1.4, ls="-.", label=f"elbow k={k_elbow}")
    axE.axvline(k_floor_cons, color="#f28e2b", lw=2.0, label=f"floor k={k_floor_cons}")
    axE.axvline(40, color="#9c755f", lw=1.0, ls="--", label="k=40")
    axE.set_xscale("log")
    axE.set_xlabel("k (log)")
    axE.set_ylabel("stripe 5-fold acc")
    axE.set_ylim(0.6, 1.0)
    axE.set_title("does the unsupervised floor capture stripe?")
    axE.grid(alpha=.2)
    axE.legend(fontsize=7, loc="lower right")

    fig.suptitle("EXP-55  Unsupervised k from spectral signal-vs-noise floor (no labels), then GT stripe scoring",
                 y=1.0, fontsize=13)
    fig.tight_layout()
    out = RES / "unsup_floor.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
