"""EXP-38 — Purely UNSUPERVISED localized clustering: tile the body, cluster within each region,
gate each, surface the regions that carry real structure. (No expert / no labels in the loop;
ground truth only validates.) v2: adds a per-region TEXTURE (banding/edge) feature so the dorsal
scan can reach STRIPE, and a combined color+texture mode.

Per-region feature modes:
  * color    : area-wt mean Lab + the MAX-chroma face's (a*,b*) (catches rare vivid patches).
  * texture  : banding/edge energy from the local jet (|band-pass L*| = stripe edges, local L*
               contrast, dark fraction) — 'how striped/contrasty is this region'.
  * color+texture : both concatenated.
"""
from __future__ import annotations

import json, warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.cm as cm
from PIL import Image
from diptest import diptest
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score

from fishpipe import config, data, textons, render
from fishpipe.features import rgb_to_lab, _region_labels

warnings.filterwarnings("ignore")
RES = config.RESULTS_DIR / "exp38"; RES.mkdir(parents=True, exist_ok=True)
FACTORS = ["belly", "tail", "stripe", "cheeks"]


def color_desc(lab, chroma, faces):
    Lf, af, bf = lab[:, faces, 0], lab[:, faces, 1], lab[:, faces, 2]
    am = np.argmax(chroma[:, faces], axis=1)
    mx_a = np.take_along_axis(af, am[:, None], 1)[:, 0]
    mx_b = np.take_along_axis(bf, am[:, None], 1)[:, 0]
    return np.column_stack([Lf.mean(1), af.mean(1), bf.mean(1), mx_a, mx_b])


def texture_desc(jet, lab, faces):
    """Banding/edge energy in the region: band-pass L* (stripe edges, jet col 6), multiscale
    band-pass (col 7), local L* std (col 8), dark fraction. (N, 6)."""
    bp = np.abs(jet[:, faces, 6]); ms = np.abs(jet[:, faces, 7]); ls = jet[:, faces, 8]
    dark = (100.0 - lab[:, faces, 0])
    return np.column_stack([bp.mean(1), bp.std(1), np.percentile(bp, 90, 1),
                            ms.mean(1), ls.mean(1), (dark > 50).mean(1)])


def gmm2(X):
    Xs = StandardScaler().fit_transform(X)
    return GaussianMixture(2, covariance_type="full", random_state=0).fit_predict(Xs), Xs


def best_dip(Xs):
    A = PCA(min(4, Xs.shape[1]), random_state=0).fit_transform(Xs)
    dp = [diptest(A[:, j]) for j in range(A.shape[1])]
    j = int(np.argmax([d[0] for d in dp]))
    return dp[j][0], dp[j][1]


def scan(mode, reg, R, lab, chroma, jet, gt, rng):
    region_dip = np.zeros(R); region_dipp = np.ones(R)
    best = {f: {"ari": -1, "region": -1} for f in FACTORS}
    null = []
    for r in range(R):
        faces = np.where(reg == r)[0]
        if len(faces) < 20:
            continue
        if mode == "color":
            X = color_desc(lab, chroma, faces)
        elif mode == "texture":
            X = texture_desc(jet, lab, faces)
        else:
            X = np.hstack([color_desc(lab, chroma, faces), texture_desc(jet, lab, faces)])
        labp, Xs = gmm2(X)
        region_dip[r], region_dipp[r] = best_dip(Xs)
        for f in FACTORS:
            a = adjusted_rand_score(gt.labels[f], labp)
            if a > best[f]["ari"]:
                best[f] = {"ari": round(a, 3), "region": int(r)}
        Xn = np.column_stack([rng.permutation(X[:, j]) for j in range(X.shape[1])])
        null.append(best_dip(StandardScaler().fit_transform(Xn))[0])
    return best, region_dip, region_dipp, float(np.percentile(null, 95))


def main():
    fcd = data.build_face_colors(verbose=False); gt = data.load_ground_truth()
    mesh = data.load_mesh()
    R = 128; reg = _region_labels(R)
    lab = rgb_to_lab(fcd.colors); chroma = np.sqrt(lab[..., 1] ** 2 + lab[..., 2] ** 2)
    P = textons.diffusion_operator(); jet = textons.local_jet(fcd, scales=(2, 4), P=P)
    rng = np.random.default_rng(0)

    report = {"R": R, "modes": {}}
    for mode in ("color", "texture", "color+texture"):
        best, rdip, rdipp, null95 = scan(mode, reg, R, lab, chroma, jet, gt, rng)
        print(f"\n=== mode = {mode}  (dip null95 = {null95:.3f}) ===")
        print(f"  {'factor':<8}{'best-region ARI':>16}{'region#':>9}{'dip>null95?':>13}")
        report["modes"][mode] = {"null95": round(null95, 3), "best": {}}
        for f in FACTORS:
            r = best[f]["region"]; sig = rdip[r] > null95
            print(f"  {f:<8}{best[f]['ari']:>16}{r:>9}{('YES' if sig else 'no'):>13}")
            report["modes"][mode]["best"][f] = {**best[f], "dip": round(float(rdip[r]), 3),
                                                "significant": bool(sig)}
        # structure map for the combined mode
        if mode == "color+texture":
            score = -np.log10(np.clip(rdipp, 1e-6, 1))
            fs = np.zeros(len(reg))
            for r in range(R):
                fs[reg == r] = score[r]
            s = (fs - fs.min()) / (np.ptp(fs) + 1e-9)
            hm = (cm.viridis(s)[:, :3] * 255).astype(np.uint8)
            b = (mesh.vertices[:, 2].min(), mesh.vertices[:, 2].max(),
                 mesh.vertices[:, 1].min(), mesh.vertices[:, 1].max())
            Image.fromarray(render.render_side_view(mesh, hm, px=1000, bounds=b)).save(
                RES / "structure_map_colortexture.png")

    (RES / "localized_v2.json").write_text(json.dumps(report, indent=2))
    print(f"\nsaved -> {RES/'localized_v2.json'}  (+ structure_map_colortexture.png)")


if __name__ == "__main__":
    main()
