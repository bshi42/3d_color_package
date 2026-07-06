"""Zero-dependency (stdlib http.server) API for the multi-label TAGGING morphospace demo.

Endpoints (all JSON unless noted):
  GET  /                                  -> index.html
  GET  /static/<f>                        -> static asset (uncached, dev)
  GET  /api/datasets                      -> [{name,title,has_gt}]
  POST /api/session            {dataset}  -> full state (+ an initial batch)
  POST /api/session/<sid>/tag         {name}                  -> {tags}
  POST /api/session/<sid>/remove_tag  {idx}                   -> state (recomputed)
  POST /api/session/<sid>/apply       {updates:{i:[tag]}}     -> state (retrain + re-project)
  POST /api/session/<sid>/batch       {n, only_unlabeled}     -> {batch}
  POST /api/session/<sid>/dendrogram                          -> latent Ward tree
  POST /api/session/<sid>/load        {data}                  -> state (restore tags+labels)
  GET  /thumb/<ds>/<name>.png   ·   /thumb_hi/<ds>/<name>.png -> specimen renders (cached)
"""
from __future__ import annotations

import io
import json
import sys
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

import numpy as np
from PIL import Image

import engine
from fishpipe import render

HERE = Path(__file__).resolve().parent
STATIC = HERE / "static"
# reuse the similarity demo's rendered thumbnails so nothing re-renders on first load
THUMB_DIR = Path(engine.FISHPIPE_ROOT) / "interactive_demo" / "_cache" / "thumbs"
THUMB_DIR.mkdir(parents=True, exist_ok=True)
THUMB_PX, THUMB_HI_PX = 150, 300
SESSIONS: dict[str, engine.TagSession] = {}


def _png_bytes(arr):
    im = Image.fromarray(arr, "RGBA") if arr.shape[-1] == 4 else Image.fromarray(arr[..., :3], "RGB")
    buf = io.BytesIO(); im.save(buf, "PNG"); return buf.getvalue()


def _render(ld, face_rgb, px):
    if ld.spec.renderer == "valve":
        return render.render_textured_rgba(ld.mesh, face_rgb, px=px, supersample=2, crop=True, align=True)
    return render.render_side_view(ld.mesh, face_rgb, px=px, bounds=ld.bounds)


def _thumb_png(dataset, name, hi=False):
    px = THUMB_HI_PX if hi else THUMB_PX
    sub = f"{dataset}_hi{THUMB_HI_PX}" if hi else dataset
    fp = THUMB_DIR / sub / f"{name}.png"
    if fp.exists():
        return fp.read_bytes()
    ld = engine.load_dataset(dataset)
    i = list(ld.names).index(name)
    fp.parent.mkdir(parents=True, exist_ok=True)
    b = _png_bytes(_render(ld, ld.fcd.colors[i], px))
    fp.write_bytes(b)
    return b


def _leaf_aspect(dataset, i):
    try:
        a = np.asarray(Image.open(io.BytesIO(_thumb_png(dataset, engine.load_dataset(dataset).names[i], hi=True))))
        return round(float(a.shape[1]) / float(a.shape[0]), 3)
    except Exception:
        return 2.0


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    # ---- helpers ----
    def _send(self, code, body, ctype="application/octet-stream"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code=200):
        self._send(code, json.dumps(obj).encode(), "application/json")

    def _err(self, code, msg):
        self._json({"error": msg}, code)

    def _body(self):
        n = int(self.headers.get("Content-Length", 0))
        return json.loads(self.rfile.read(n) or b"{}") if n else {}

    def _sess(self, sid):
        return SESSIONS.get(sid)

    # ---- GET ----
    def do_GET(self):
        path = unquote(urlparse(self.path).path)
        if path == "/" or path == "/index.html":
            return self._send(200, (STATIC / "index.html").read_bytes(), "text/html")
        if path.startswith("/static/"):
            fp = STATIC / path[len("/static/"):]
            if fp.is_file():
                ct = "text/css" if fp.suffix == ".css" else "application/javascript" if fp.suffix == ".js" else "text/plain"
                return self._send(200, fp.read_bytes(), ct)
            return self._err(404, "not found")
        if path == "/api/datasets":
            return self._json({"datasets": engine.list_datasets()})
        for pre, hi in (("/thumb_hi/", True), ("/thumb/", False)):
            if path.startswith(pre):
                rest = path[len(pre):]
                ds, name = rest.split("/", 1)
                name = name[:-4] if name.endswith(".png") else name
                try:
                    return self._send(200, _thumb_png(ds, name, hi=hi), "image/png")
                except Exception as e:
                    return self._err(404, str(e))
        return self._err(404, "not found")

    # ---- POST ----
    def do_POST(self):
        path = unquote(urlparse(self.path).path)
        try:
            body = self._body()
        except Exception as e:
            return self._err(400, f"bad json: {e}")

        if path == "/api/session":
            ds = body.get("dataset", "fishy")
            if ds not in engine.DATASETS:
                return self._err(400, "unknown dataset")
            s = engine.TagSession(ds)
            sid = uuid.uuid4().hex[:12]
            SESSIONS[sid] = s
            s.roll_batch(int(body.get("batch", 12)), only_unlabeled=True)
            out = s.state(); out["sid"] = sid
            return self._json(out)

        parts = path.strip("/").split("/")          # api session <sid> <verb>
        if len(parts) >= 4 and parts[0] == "api" and parts[1] == "session":
            s = self._sess(parts[2])
            if s is None:
                return self._err(404, "no session")
            verb = parts[3]
            try:
                if verb == "tag":
                    s.add_tag(body.get("name", ""))
                    return self._json({"tags": s.tags})
                if verb == "remove_tag":
                    s.remove_tag(int(body.get("idx", -1)))
                    return self._json(s.state())
                if verb == "apply":
                    s.apply_labels(body.get("updates", {}))
                    s.roll_batch(int(body.get("n", 12)), only_unlabeled=True, active=True)  # auto-roll most-informative
                    return self._json(s.state())
                if verb == "batch":
                    s.roll_batch(int(body.get("n", 12)), bool(body.get("only_unlabeled", True)),
                                 active=bool(body.get("active", True)))
                    return self._json({"batch": s.batch})
                if verb == "load":
                    s.import_state(body.get("data", {}))
                    return self._json(s.state())
                if verb == "dendrogram":
                    d = s.dendrogram()
                    d["leaf_aspect"] = _leaf_aspect(s.dataset, d["leaves"][0]["i"]) if d["leaves"] else 2.0
                    return self._json(d)
            except ValueError as e:
                return self._err(400, str(e))
            except Exception as e:
                return self._err(500, str(e))
        return self._err(404, "not found")


def main():
    port = 8770
    for i, a in enumerate(sys.argv):
        if a == "--port" and i + 1 < len(sys.argv):
            port = int(sys.argv[i + 1])
    print("warming descriptor + UMAP JIT…", flush=True)
    engine.TagSession("mussel")                       # warm numba/UMAP so the first click is fast
    srv = ThreadingHTTPServer(("127.0.0.1", port), H)
    print(f"\n  Tagging morphospace demo -> http://127.0.0.1:{port}\n", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
