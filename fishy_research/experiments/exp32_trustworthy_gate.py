"""EXP-32 — Trustworthy small-n gate: does SigClust + consensus-PAC FIND the real color
clusters and ABSTAIN on the (supervised-only) pattern structure — at n=250 AND at n=40?

This is the safety core for "find patterns unsupervised on small samples": a gate that
surfaces genuinely separated structure and refuses to fabricate. Validated against the
fishy ground truth (belly x tail = 4 real groups; stripe/cheeks = no density gap, must abstain).
"""
from __future__ import annotations

import json
import warnings
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler

from fishpipe import data, features, spectral, textons, gating

warnings.filterwarnings("ignore")


def descriptors(fcd):
    color = features.area_hist(fcd, n_clusters=24, color_space="lab")
    P = textons.diffusion_operator()
    jet = textons.local_jet(fcd, scales=(2, 4), P=P)
    sc, km = textons.build_codebook(jet, K=128, seed=0)
    pattern = textons.encode(fcd, jet, sc, km, mode="bow")
    return {"COLOR (area-hist)": color, "PATTERN (texton-BoW)": pattern}


def main():
    fcd = data.build_face_colors(verbose=False)
    gt = data.load_ground_truth()
    joint = data.joint_label(gt, ("belly", "tail"))     # 4 true color groups
    desc = descriptors(fcd)
    report = {}

    for n_total in (250, 40):
        if n_total < 250:
            rng = np.random.default_rng(0)
            ch = gt.labels["cheeks"]
            idx = list(rng.choice(np.where(ch == 1)[0], 3, replace=False))
            idx += list(rng.choice(np.where(ch == 0)[0], n_total - 3, replace=False))
            idx = np.array(sorted(idx))
        else:
            idx = np.arange(250)
        print(f"\n{'='*74}\n n = {len(idx)}\n{'='*74}")
        report[n_total] = {}
        for name, X in desc.items():
            Xi = X[idx]
            # reduce to the biologist-facing 2-D morphospace (where belly x tail = 4 quadrants)
            Xr = PCA(n_components=min(2, Xi.shape[1] - 1, len(idx) - 1),
                     random_state=0).fit_transform(StandardScaler().fit_transform(Xi))
            sc = gating.sigclust(Xr, n_sim=500)
            cg = gating.consensus_gate(Xr, ks=(2, 3, 4, 5), n_resample=60, n_null=25)
            kcons = cg["chosen_k"]
            # The clustering a user would actually see: the team's own GMM+BIC auto-cluster.
            from fishpipe.recommended import auto_cluster
            lab = auto_cluster(Xr, max_k=6)
            kgmm = len(np.unique(lab))
            ari_color = adjusted_rand_score(joint[idx], lab)
            ari_stripe = adjusted_rand_score(gt.labels["stripe"][idx], lab)
            jac = gating.cluster_jaccard(Xr, kgmm) if kgmm >= 2 else np.array([1.0])
            n_stable = int(np.sum(jac >= 0.75))
            report[n_total][name] = {
                "sigclust_p": round(sc["pvalue"], 4),
                "consensus_chosen_k": kcons,
                "gmm_bic_k": kgmm,
                "ARI_vs_colorgroups": round(ari_color, 3),
                "ARI_vs_stripe": round(ari_stripe, 3),
                "cluster_jaccard": [round(float(j), 2) for j in jac],
                "n_stable_clusters(jac>=.75)": n_stable,
            }
            print(f"\n  {name}")
            print(f"    SigClust p={sc['pvalue']:.4f} (>1 Gaussian?)   consensus k={kcons}   GMM+BIC k={kgmm}")
            print(f"    GMM clusters: ARI vs color-groups={ari_color:.3f}  vs stripe={ari_stripe:.3f}")
            print(f"    per-cluster bootstrap Jaccard: {[round(float(j),2) for j in jac]}  "
                  f"-> {n_stable}/{kgmm} stable (>=0.75)")

    out = data.config.RESULTS_DIR / "exp32"
    out.mkdir(exist_ok=True)
    (out / "trustworthy_gate.json").write_text(json.dumps(report, indent=2))
    print(f"\nsaved -> {out/'trustworthy_gate.json'}")


if __name__ == "__main__":
    main()
