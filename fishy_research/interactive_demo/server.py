"""Zero-dependency web server for the interactive morphospace feedback demo.

Stdlib http.server only (no FastAPI/Flask) so it runs in the existing conda env with no install.
Holds Sessions (engine.py), serves the single-page frontend, renders specimen thumbnails +
PC->region heatmaps + an exemplar dendrogram as PNGs, and drives the feedback->morph loop.

Run:   .venv/bin/python interactive_demo/server.py [--port 8000] [--host 127.0.0.1]
Then open http://127.0.0.1:8000
"""
from __future__ import annotations

import argparse
import io
import json
import threading
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs

import sys as _sys
_HERE0 = Path(__file__).resolve().parent
for _p in (str(_HERE0.parent), str(_HERE0.parent / "src")):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from PIL import Image
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster

from interactive_demo import engine
from fishpipe import render

HERE = Path(__file__).resolve().parent
STATIC = HERE / "static"
THUMB_DIR = HERE / "_cache" / "thumbs"
THUMB_PX = 150

_LOCK = threading.Lock()
SESSIONS: dict[str, engine.Session] = {}
_THUMB_BYTES: dict[tuple, bytes] = {}        # (dataset, i) -> PNG bytes
_DENDRO_PALETTE = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e",
                   "#17becf", "#bcbd22", "#e377c2", "#8c564b", "#7f7f7f"]


# --------------------------------------------------------------------------- rendering
def _png_bytes(arr: np.ndarray) -> bytes:
    if arr.shape[-1] == 4:
        im = Image.fromarray(arr, "RGBA")
    else:
        im = Image.fromarray(arr[..., :3], "RGB")
    buf = io.BytesIO()
    im.save(buf, "PNG")
    return buf.getvalue()


def _render_specimen(ld: engine.LoadedDataset, face_rgb: np.ndarray, px: int = THUMB_PX) -> np.ndarray:
    if ld.spec.renderer == "valve":
        return render.render_textured_rgba(ld.mesh, face_rgb, px=px, supersample=2,
                                            crop=True, align=True)
    return render.render_side_view(ld.mesh, face_rgb, px=px, bounds=ld.bounds)


THUMB_HI_PX = 300        # high-res leaf renders for the (zoomable, vector) dendrogram
_HIRES: dict[tuple, np.ndarray] = {}


def _hires_arr(dataset: str, i: int) -> np.ndarray:
    """High-resolution render of specimen i for crisp dendrogram leaves; cached (memory + disk).
    Separate from the 250 canvas thumbnails so it doesn't bloat their memory."""
    key = (dataset, i)
    if key in _HIRES:
        return _HIRES[key]
    ld = engine.load_dataset(dataset)
    fp = THUMB_DIR / f"{dataset}_hi{THUMB_HI_PX}" / f"{ld.names[i]}.png"
    fp.parent.mkdir(parents=True, exist_ok=True)
    if not fp.exists():
        Image.fromarray(_render_specimen(ld, ld.fcd.colors[i], px=THUMB_HI_PX)).save(fp)
    arr = np.asarray(Image.open(fp).convert("RGBA"))
    _HIRES[key] = arr
    return arr


def _hires_png(dataset: str, i: int) -> bytes:
    """Hi-res specimen PNG bytes (rendered+cached), served to the dendrogram <image> elements."""
    ld = engine.load_dataset(dataset)
    fp = THUMB_DIR / f"{dataset}_hi{THUMB_HI_PX}" / f"{ld.names[i]}.png"
    if not fp.exists():
        _hires_arr(dataset, i)
    return fp.read_bytes()


def dendrogram_data(sess: engine.Session, positions=None) -> dict:
    """Tree structure for the CLIENT to draw as inline SVG (vector branches + hi-res <image> leaves).

    Uniform merge heights (always balanced) with the real Ward distance carried per-link as `w`
    (0..1, robust-scaled) so the client can map it to line thickness.
    """
    if positions is not None:
        idx, Q = sess.dendro_from_positions(positions)
    else:
        idx, Q = sess.dendro_subset()
    idx = list(idx)
    Z = linkage(Q, method="ward")
    true_d = Z[:, 2].astype(float).copy()
    m = len(Z); Nleaf = m + 1
    # DEPTH-ALIGNED levels: each merge sits at level = height of its subtree above the leaves
    # (leaf=0, merge-of-two-leaves=1, ...), so all "first merges" share a row, etc. Not monotonic
    # in merge order, so we draw with uniform heights (for a valid scipy layout + colouring) then
    # remap every link's y-coords to these levels.
    level = np.zeros(m)
    for k in range(m):
        a, b = int(Z[k, 0]), int(Z[k, 1])
        la = 0.0 if a < Nleaf else level[a - Nleaf]
        lb = 0.0 if b < Nleaf else level[b - Nleaf]
        level[k] = 1.0 + max(la, lb)
    Z[:, 2] = np.arange(1.0, m + 1.0)                      # uniform (monotonic) for scipy's layout
    kc = int(np.clip(6, 2, max(2, len(idx) - 1)))
    h = Z[:, 2]
    ct = float(0.5 * (h[-kc] + h[-(kc - 1)])) if len(h) >= kc else 0.0
    dd = dendrogram(Z, no_plot=True, color_threshold=ct, above_threshold_color="#b8b8b8")
    from matplotlib.colors import to_hex                    # scipy returns 'C1' etc.; SVG needs hex
    dmin = float(true_d.min()); dhi = float(np.percentile(true_d, 95))

    def lvl_of(uh):                                         # uniform child height -> its depth level
        return 0.0 if uh < 0.5 else float(level[int(round(uh)) - 1])
    # descendant leaf-position span per merge (for clicking a branch to select its clade)
    Nl = len(idx)
    desc = [None] * len(Z)

    def _leaves(cid):
        return [cid] if cid < Nl else desc[cid - Nl]
    for k in range(len(Z)):
        desc[k] = _leaves(int(Z[k, 0])) + _leaves(int(Z[k, 1]))
    pos_of = {leaf: p for p, leaf in enumerate(dd["leaves"])}
    links = []
    for xs, ys, col in zip(dd["icoord"], dd["dcoord"], dd["color_list"]):
        mi = max(0, min(len(true_d) - 1, int(round(ys[1])) - 1))
        w = float(np.clip((true_d[mi] - dmin) / (dhi - dmin + 1e-9), 0.0, 1.0))
        ly = [lvl_of(ys[0]), level[mi], level[mi], lvl_of(ys[3])]   # remap to depth levels
        ps = [pos_of[l] for l in desc[mi]]
        links.append({"x": [round(float(v), 2) for v in xs], "y": [round(float(v), 2) for v in ly],
                      "color": to_hex(col), "w": round(w, 3), "span": [int(min(ps)), int(max(ps))]})
    leaf_colors = dd.get("leaves_color_list", ["#444444"] * len(idx))
    ds = sess.dataset
    sample = _hires_arr(ds, idx[dd["leaves"][0]])          # for leaf aspect ratio
    leaves = [{"pos": pos, "i": int(idx[leaf]), "name": str(sess.ld.names[idx[leaf]]),
               "color": to_hex(leaf_colors[pos]) if pos < len(leaf_colors) else "#444444"}
              for pos, leaf in enumerate(dd["leaves"])]
    return {"n": len(idx), "ymax": float(level.max()), "step": 10.0,
            "leaf_aspect": round(float(sample.shape[1]) / float(sample.shape[0]), 3),
            "links": links, "leaves": leaves, "dataset": ds, "round": sess.round}


def prerender_thumbs(dataset: str):
    """Render every specimen's colour thumbnail once; cache to disk + memory."""
    ld = engine.load_dataset(dataset)
    out = THUMB_DIR / dataset
    out.mkdir(parents=True, exist_ok=True)
    n = len(ld.names)
    for i, name in enumerate(ld.names):
        key = (dataset, i)
        if key in _THUMB_BYTES:
            continue
        fp = out / f"{name}.png"
        if fp.exists():
            _THUMB_BYTES[key] = fp.read_bytes()
            continue
        img = _render_specimen(ld, ld.fcd.colors[i])
        b = _png_bytes(img)
        fp.write_bytes(b)
        _THUMB_BYTES[key] = b
        if i % 25 == 0 or i == n - 1:
            print(f"    [{dataset}] thumbnails {i + 1}/{n}", flush=True)


def _thumb_arr(dataset: str, i: int) -> np.ndarray:
    if (dataset, i) not in _THUMB_BYTES:
        prerender_thumbs(dataset)
    im = Image.open(io.BytesIO(_THUMB_BYTES[(dataset, i)])).convert("RGBA")
    return np.asarray(im)


def render_heatmap(sess: engine.Session, pc: int) -> bytes:
    face_rgb = sess.region_face_rgb(pc)
    img = _render_specimen(sess.ld, face_rgb)
    return _png_bytes(img)


def render_dendrogram(sess: engine.Session, positions=None, k_clusters: int = 6, n_leaves: int = 26) -> bytes:
    """Exemplar dendrogram on a diverse FPS subset of the CLIENT's live 2-D layout (WYSIWYG).

    Leaves carry specimen thumbnails (in dendrogram order); links + thumbnail borders share a
    per-cluster colour from a color_threshold cut yielding ~k_clusters groups. Redrawn on demand.
    """
    ds = sess.dataset
    if positions is not None:
        idx, Zsub = sess.dendro_from_positions(positions, n_leaves)
    else:
        idx, Zsub = sess.dendro_subset(n_leaves)
    idx = list(idx)
    Z = linkage(Zsub, method="ward")
    true_d = Z[:, 2].astype(float).copy()            # real Ward merge distances (encoded as line width)
    # UNIFORM merge heights: keep the Ward topology + merge order, but plot every merge at an
    # evenly-spaced height so the tree is always balanced and never crushed by outlier distances.
    Z[:, 2] = np.arange(1.0, len(Z) + 1.0)
    kc = int(np.clip(k_clusters, 2, len(idx) - 1))
    # cut between the kc-th and (kc-1)-th highest merges -> ~kc clusters
    heights = Z[:, 2]
    ct = float(0.5 * (heights[-kc] + heights[-(kc - 1)])) if len(heights) >= kc else 0.0

    nleaves = len(idx)
    figw = max(7.0, nleaves * 0.62)              # wider per-leaf -> roomy slots, no thumbnail overlap
    slot_pt = 0.98 * figw * 72.0 / nleaves       # points available per leaf slot
    # high dpi: matplotlib's SVG backend rasterizes embedded images at the figure dpi (vectors are
    # unaffected), so a low dpi bakes in tiny low-res thumbnails. 800 -> sharp leaves when zoomed.
    fig = plt.figure(figsize=(figw, 5.0), dpi=800)
    ax = fig.add_axes([0.01, 0.30, 0.98, 0.64])
    dd = dendrogram(Z, no_plot=True, color_threshold=ct, above_threshold_color="#b8b8b8")
    # draw each link by hand: vertical position is UNIFORM, but LINE WIDTH encodes the real Ward
    # merge distance (robust-scaled to the 95th pct so one outlier merge doesn't flatten the rest).
    dmin = float(true_d.min()); dhi = float(np.percentile(true_d, 95))
    for xs, ys, col in zip(dd["icoord"], dd["dcoord"], dd["color_list"]):
        mi = max(0, min(len(true_d) - 1, int(round(ys[1])) - 1))   # link's merge index from its height
        t = float(np.clip((true_d[mi] - dmin) / (dhi - dmin + 1e-9), 0.0, 1.0))
        ax.plot(xs, ys, color=col, lw=0.8 + 5.4 * t, solid_capstyle="round", solid_joinstyle="round")
    ymax = float(len(Z))
    ax.set_xlim(0, 10 * len(idx)); ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title(f"Dendrogram — {len(idx)} specimens (uniform spacing; thicker line = larger merge distance · round {sess.round})",
                 fontsize=9)
    leaf_colors = dd.get("leaves_color_list", ["#444444"] * len(idx))
    for pos, leaf in enumerate(dd["leaves"]):
        gi = idx[leaf]
        arr = _hires_arr(ds, gi)                            # high-res render -> crisp when zoomed in
        im = Image.fromarray(arr).convert("RGB")
        oi = OffsetImage(np.asarray(im), zoom=(0.86 * slot_pt) / im.width)   # fit slot -> no overlap
        xc = 5 + pos * 10                       # scipy spaces leaves at 10*pos + 5
        bc = leaf_colors[pos] if pos < len(leaf_colors) else "#444444"
        ab = AnnotationBbox(oi, (xc, 0), xybox=(xc, -0.05 * ymax),
                            box_alignment=(0.5, 1.0), frameon=True,
                            bboxprops=dict(edgecolor=bc, lw=2.2), pad=0.02)
        ax.add_artist(ab)
    ax.set_ylim(-0.16 * ymax, ymax * 1.04)
    buf = io.BytesIO()
    fig.savefig(buf, format="svg", bbox_inches="tight", facecolor="white")   # vector -> crisp at any zoom
    plt.close(fig)
    return buf.getvalue()


# --------------------------------------------------------------------------- HTTP handler
class Handler(BaseHTTPRequestHandler):
    server_version = "MorphoDemo/1.0"

    def log_message(self, *a):
        pass  # quiet

    # ---- response helpers
    def _send(self, code, body: bytes, ctype="application/octet-stream", cache=False):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        if cache:
            self.send_header("Cache-Control", "public, max-age=86400")
        else:
            self.send_header("Cache-Control", "no-store")
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _json(self, obj, code=200):
        self._send(code, json.dumps(obj).encode(), "application/json")

    def _err(self, code, msg):
        self._json({"error": msg}, code)

    def _body_json(self):
        n = int(self.headers.get("Content-Length", 0))
        if n == 0:
            return {}
        return json.loads(self.rfile.read(n).decode() or "{}")

    def _sess(self, sid) -> engine.Session | None:
        return SESSIONS.get(sid)

    # ---- GET
    def do_GET(self):
        u = urlparse(self.path)
        path = u.path
        q = parse_qs(u.query)
        try:
            if path == "/" or path == "":
                return self._send(200, (STATIC / "index.html").read_bytes(), "text/html; charset=utf-8")
            if path.startswith("/static/"):
                return self._serve_static(path[len("/static/"):])
            if path == "/api/datasets":
                return self._json({"datasets": engine.list_datasets()})
            if path.startswith("/thumb_hi/"):
                return self._serve_thumb_hi(path[len("/thumb_hi/"):])
            if path.startswith("/thumb/"):
                return self._serve_thumb(path[len("/thumb/"):])
            if path.startswith("/api/session/"):
                return self._session_get(path, q)
            return self._err(404, f"not found: {path}")
        except BrokenPipeError:
            pass
        except Exception as e:  # surface errors as JSON for the client console
            import traceback; traceback.print_exc()
            return self._err(500, f"{type(e).__name__}: {e}")

    def _serve_static(self, rel):
        fp = (STATIC / rel).resolve()
        if not str(fp).startswith(str(STATIC)) or not fp.exists():
            return self._err(404, "static not found")
        ctype = {"js": "application/javascript", "css": "text/css", "html": "text/html",
                 "png": "image/png", "svg": "image/svg+xml"}.get(fp.suffix[1:], "application/octet-stream")
        return self._send(200, fp.read_bytes(), ctype, cache=False)   # dev: always fresh JS/CSS

    def _serve_thumb(self, rel):
        # rel = "<dataset>/<name>.png"
        ds, _, fname = rel.partition("/")
        name = fname[:-4] if fname.endswith(".png") else fname
        ld = engine.load_dataset(ds)
        try:
            i = ld.names.index(name)
        except ValueError:
            return self._err(404, "specimen not found")
        if (ds, i) not in _THUMB_BYTES:          # requested before warm/session prerender
            prerender_thumbs(ds)
        return self._send(200, _THUMB_BYTES[(ds, i)], "image/png", cache=True)

    def _serve_thumb_hi(self, rel):
        ds, _, fname = rel.partition("/")
        name = fname[:-4] if fname.endswith(".png") else fname
        ld = engine.load_dataset(ds)
        try:
            i = ld.names.index(name)
        except ValueError:
            return self._err(404, "specimen not found")
        return self._send(200, _hires_png(ds, i), "image/png", cache=True)

    def _session_get(self, path, q):
        parts = path.split("/")        # ['', 'api', 'session', '<sid>', '<verb>']
        if len(parts) < 5:
            return self._err(404, "bad session path")
        sid, verb = parts[3], parts[4]
        sess = self._sess(sid)
        if sess is None:
            return self._err(404, "session not found")
        # fast JSON reads touch live state -> take the lock so they never observe a half-applied
        # mutation. (PNG renders below tolerate a snapshot and stay unlocked.)
        if verb in ("graph", "morphospace", "pc_info", "panel", "state"):
            with _LOCK:
                if verb == "graph":
                    return self._json(sess.graph())
                if verb == "morphospace":
                    return self._json(sess.morphospace())
                if verb == "pc_info":
                    pc = int(q.get("pc", [sess.active_pc])[0])
                    return self._json(sess.pc_info(pc))
                if verb == "panel":
                    anchor = q.get("anchor", [None])[0]
                    anchor = int(anchor) if anchor not in (None, "", "null") else None
                    if anchor is not None and not (0 <= anchor < sess.N):
                        return self._err(400, "anchor index out of range")
                    return self._json(sess.panel(anchor=anchor))
                return self._json({"dataset": sess.dataset, "K": sess.K, "active_pc": sess.active_pc,
                                   "axes": list(sess.axes), "round": sess.round,
                                   "has_gt": sess.ld.spec.has_gt})
        if verb == "heatmap.png":
            pc = int(q.get("pc", [sess.active_pc])[0])
            pc = int(np.clip(pc, 0, sess.K - 1))
            return self._send(200, render_heatmap(sess, pc), "image/png")
        if verb == "dendrogram.png":
            return self._send(200, render_dendrogram(sess), "image/svg+xml")
        if verb == "explain_heatmap.png":
            w = getattr(sess, "_explain_w", None)
            if w is None:
                return self._err(404, "no explanation computed yet")
            return self._send(200, _png_bytes(_render_specimen(sess.ld, sess.region_face_rgb_weighted(w))), "image/png")
        return self._err(404, f"unknown verb: {verb}")

    # ---- POST
    def do_POST(self):
        u = urlparse(self.path)
        path = u.path
        try:
            if path == "/api/session":
                return self._create_session()
            if path.startswith("/api/session/"):
                return self._session_post(path)
            return self._err(404, f"not found: {path}")
        except BrokenPipeError:
            pass
        except Exception as e:
            import traceback; traceback.print_exc()
            return self._err(500, f"{type(e).__name__}: {e}")

    def _create_session(self):
        body = self._body_json()
        ds = body.get("dataset", "fishy")
        if ds not in engine.DATASETS:
            return self._err(400, f"unknown dataset: {ds}")
        print(f"[session] loading dataset '{ds}' + thumbnails ...", flush=True)
        prerender_thumbs(ds)
        with _LOCK:
            sid = uuid.uuid4().hex[:12]
            sess = engine.Session(ds)
            SESSIONS[sid] = sess
        # hi-res leaf renders for the crisp vector dendrogram (one-time, disk-cached)
        leaves = sess.dendro_leaf_indices()
        print(f"[session] hi-res dendrogram leaves ({len(leaves)}) ...", flush=True)
        for i in leaves:
            _hires_arr(ds, i)
        spec = engine.DATASETS[ds]
        return self._json({"sid": sid, "dataset": ds, "title": spec.title, "K": sess.K,
                           "has_gt": spec.has_gt, "names": sess.ld.names,
                           "active_pc": sess.active_pc, "axes": list(sess.axes),
                           "evr": [float(v) for v in sess.evr]})

    def _session_post(self, path):
        parts = path.split("/")
        if len(parts) < 5:
            return self._err(404, "bad session path")
        sid, verb = parts[3], parts[4]
        sess = self._sess(sid)
        if sess is None:
            return self._err(404, "session not found")
        body = self._body_json()
        if verb == "dendrogram":                         # WYSIWYG: structure for the client SVG
            try:
                return self._json(dendrogram_data(sess, positions=body.get("positions")))
            except ValueError as e:
                return self._err(400, str(e))
        if verb == "explain":                            # which PCs explain the expert's grouping
            return self._json(sess.explain(body.get("pairs", []), positions=body.get("positions")))
        n = sess.N

        def _inrange(v):
            return 0 <= int(v) < n

        with _LOCK:
            if verb == "feedback":
                if "anchor" not in body:
                    return self._err(400, "missing required field: anchor")
                anchor = int(body["anchor"])
                ranked = [int(x) for x in body.get("ranked", [])]
                tied = bool(body.get("tied", False))
                remote = body.get("remote", None)
                remote = int(remote) if remote not in (None, "", "null") else None
                if not _inrange(anchor) or not all(_inrange(r) for r in ranked) \
                        or (remote is not None and not _inrange(remote)):
                    return self._err(400, "specimen index out of range")
                res = sess.feedback(anchor, ranked, tied, remote=remote)
                return self._json(res)
            if verb == "active_pc":
                if "pc" not in body:
                    return self._err(400, "missing required field: pc")
                sess.set_active_pc(int(body["pc"]))
                return self._json({"active_pc": sess.active_pc, "axes": list(sess.axes)})
            if verb == "axes":
                if "x" not in body or "y" not in body:
                    return self._err(400, "missing required fields: x, y")
                sess.set_axes(int(body["x"]), int(body["y"]))
                return self._json({"axes": list(sess.axes)})
            if verb == "reset":
                pc = body.get("pc", None)
                sess.reset(int(pc) if pc not in (None, "", "null") else None)
                return self._json({"round": sess.round})
        return self._err(404, f"unknown verb: {verb}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--warm", default="", help="comma-sep datasets to prerender thumbnails at boot")
    args = ap.parse_args()
    for ds in [d for d in args.warm.split(",") if d]:
        print(f"[warm] {ds}", flush=True)
        prerender_thumbs(ds)
    httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"\n  Morphospace demo running -> http://{args.host}:{args.port}\n", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nbye")


if __name__ == "__main__":
    main()
