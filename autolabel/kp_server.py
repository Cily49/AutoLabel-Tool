"""Keypoint model server for CVAT: human pose, face landmarks, hands (YOLOX + RTMW, CPU / onnxruntime).

    python -m autolabel.kp_server --port 8767

POST /detect {"image": b64, "threshold": 0.5, "kind": "body17|face68|face13|hand21",
              "auto_rotate": false, "mode": "performance", "kp_thr": 3.0, "out_thr": 1.0, "face_thr": 3.0}
             -> CVAT detector results: [{"label", "type": "skeleton", "elements": [...], "confidence"}]
GET  /health
Listens on 127.0.0.1 only (Docker Desktop reaches it as host.docker.internal).

Note: CVAT's Nuclio integration keeps only the "outside" flag of skeleton points (occluded is
always reset to false), so points below out_thr are sent as outside and the rest as visible.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
from PIL import Image

from autolabel import skeletons
from autolabel.rtmw import Keypointer

LOCK = threading.Lock()
MODELS: dict[str, Keypointer] = {}
F68 = 23
HAND = {"left": 91, "right": 112}


def model(mode: str) -> Keypointer:
    if mode not in MODELS:
        MODELS[mode] = Keypointer(mode)
    return MODELS[mode]


def skeleton(label, names, pts, scores, size, out_thr, conf, attributes=None):
    h, w = size
    elems = []
    for name, (x, y), s in zip(names, pts, scores):
        outside = bool(s < out_thr or not (0 <= x < w and 0 <= y < h))
        elems.append({"label": name, "type": "points", "outside": outside,
                      "points": [float(np.clip(x, 0, w - 1)), float(np.clip(y, 0, h - 1))]})
    if all(e["outside"] for e in elems):
        return None
    item = {"label": label, "type": "skeleton", "elements": elems, "confidence": f"{conf:.3f}"}
    if attributes:
        item["attributes"] = attributes
    return item


def detect(req: dict) -> list:
    kind = req.get("kind", "body17")
    if kind not in skeletons.KINDS:
        raise ValueError(f"kind must be one of {sorted(skeletons.KINDS)}")
    spec = skeletons.KINDS[kind]
    label, names = spec["label"], spec["names"]
    out_thr, kp_thr, face_thr = float(req.get("out_thr", 1.0)), float(req.get("kp_thr", 3.0)), float(req.get("face_thr", 3.0))
    rgb = np.asarray(Image.open(io.BytesIO(base64.b64decode(req["image"]))).convert("RGB"))
    size = rgb.shape[:2]
    people = model(req.get("mode", "performance"))(rgb, det_thr=float(req.get("threshold") or 0.5),
                                                   auto_rotate=bool(req.get("auto_rotate", False)))
    results = []
    for _box, kp, sc in people:
        conf = float(np.clip(sc[:17].mean() / 10, 0, 1))  # RTMW scores are unnormalised (~0-10)
        if kind == "body17":
            results.append(skeleton(label, names, kp[:17], sc[:17], size, out_thr, conf))
        elif kind in ("face68", "face13"):
            face, fsc = kp[F68:F68 + 68], sc[F68:F68 + 68]
            if fsc.mean() < face_thr:
                continue  # no clearly visible face: RTMW still guesses 68 points, skip them
            if kind == "face68":
                results.append(skeleton(label, names, face, fsc, size, out_thr, conf))
            else:
                pts = [face[i].mean(0) for i in skeletons.FACE13_FROM68.values()]
                s = [fsc[i].min() for i in skeletons.FACE13_FROM68.values()]
                results.append(skeleton(label, names, pts, s, size, out_thr, conf))
        else:  # hand21: one skeleton per visible hand, side = the person's own side
            for side, start in HAND.items():
                hs = sc[start:start + 21]
                if hs.mean() < kp_thr:
                    continue
                results.append(skeleton(label, names, kp[start:start + 21], hs, size, out_thr, conf,
                                        attributes=[{"name": "side", "value": side}]))
    return [r for r in results if r]


class Handler(BaseHTTPRequestHandler):
    def _send(self, code: int, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):  # noqa: N802
        self._send(200, {"ok": True, "model": "rtmw", "loaded": sorted(MODELS)})

    def do_POST(self):  # noqa: N802
        if self.path != "/detect":
            return self._send(404, {"error": "unknown endpoint"})
        t = time.time()
        try:
            req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
            with LOCK:
                res = detect(req)
            print(f"/detect {req.get('kind')} -> {len(res)} skeleton(s) in {time.time() - t:.2f}s", flush=True)
            self._send(200, res)
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            self._send(500, {"error": f"{type(e).__name__}: {e}"})

    def log_message(self, *args):
        pass


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8767)
    ap.add_argument("--preload", default="performance", help="model size to load at start ('' = lazy)")
    a = ap.parse_args()
    if a.preload:
        model(a.preload)
    print(f"Keypoint server ready; http://{a.host}:{a.port}", flush=True)
    ThreadingHTTPServer((a.host, a.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
