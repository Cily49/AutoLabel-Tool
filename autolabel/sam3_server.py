"""SAM3 model server for CVAT (runs on the host, inside the SAM3 Python environment).

CVAT's Nuclio functions (Docker) forward requests here through host.docker.internal. The model
is loaded once and shared by every SAM3 function.

    python -m autolabel.sam3_server --port 8766

POST /detect    {"image": b64, "threshold": 0.4, "labels": ["person", ...], "output": "box|mask|polygon"}
                -> CVAT detector results: [{"label", "type", "points" | "mask", "confidence"}]
POST /interact  {"image": b64, "pos_points": [[x,y]], "neg_points": [[x,y]], "obj_bbox": [[x1,y1],[x2,y2]]}
                -> {"shapes": [{"type": "mask", "points": <CVAT RLE>}]}
GET  /health
Listens on 127.0.0.1 only (Docker Desktop reaches it as host.docker.internal).
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import threading
import time
import traceback
import warnings
from collections import OrderedDict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

warnings.filterwarnings("ignore")

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from autolabel import vocab  # noqa: E402

LOCK = threading.Lock()  # the model is not thread-safe
CACHE: "OrderedDict[str, dict]" = OrderedDict()
MODEL = PROC = None


def state_for(image_b64: str):
    """Encode each image once: CVAT sends the same frame for every click / prompt."""
    key = hashlib.sha1(image_b64.encode()).hexdigest()
    if key in CACHE:
        CACHE.move_to_end(key)
        return CACHE[key]
    image = Image.open(io.BytesIO(base64.b64decode(image_b64))).convert("RGB")
    st = PROC.set_image(image)
    CACHE[key] = st
    while len(CACHE) > 4:
        CACHE.popitem(last=False)
    return st


def polygon(mask: np.ndarray) -> list:
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []
    c = cv2.approxPolyDP(max(contours, key=cv2.contourArea), 1.0, True)
    return c.reshape(-1).astype(float).tolist() if len(c) >= 3 else []


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


def detect(req: dict) -> list:
    thr = float(req.get("threshold") or 0.4)
    output = req.get("output", "box")
    st = state_for(req["image"])
    PROC.confidence_threshold = thr
    results = []
    for label in [l for l in req.get("labels", []) if l]:
        found = []  # (score, box, mask) merged over the label's prompts
        for prompt in vocab.prompts_for(label):
            PROC.reset_all_prompts(st)
            o = PROC.set_text_prompt(state=st, prompt=prompt)
            masks = o["masks"].squeeze(1).cpu().numpy().astype(bool) if len(o["scores"]) else []
            for m, b, s in zip(masks, o["boxes"].float().cpu().numpy(), o["scores"].float().cpu().numpy()):
                if s >= thr and m.any():
                    found.append((float(s), [float(v) for v in b], m))
        kept = []
        for s, b, m in sorted(found, key=lambda t: -t[0]):  # NMS across prompts of the same label
            if all(iou(b, k[1]) < 0.6 for k in kept):
                kept.append((s, b, m))
        for s, b, m in kept:
            item = {"confidence": f"{s:.4f}", "label": label}
            if output == "box":
                item.update(type="rectangle", points=b)
            elif output == "polygon":
                pts = polygon(m)
                if not pts:
                    continue
                item.update(type="polygon", points=pts)
            else:
                ys, xs = np.nonzero(m)
                x1, y1, x2, y2 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
                crop = m[y1:y2 + 1, x1:x2 + 1].astype(np.uint8).reshape(-1).tolist()
                item.update(type="mask", mask=crop + [x1, y1, x2, y2])
                pts = polygon(m)  # used when "Convert masks to polygons" is ticked in CVAT
                if pts:
                    item["points"] = pts
            results.append(item)
    PROC.reset_all_prompts(st)
    return results


def mask_to_rle(mask: np.ndarray) -> list:
    """Full-image mask -> CVAT interactor RLE."""
    h, w = mask.shape
    pixels = (mask.reshape(-1) != 0).astype(np.uint8)
    changes = np.flatnonzero(pixels[1:] != pixels[:-1]) + 1
    rle = np.diff(np.concatenate(([0], changes, [pixels.size]))).tolist()
    if pixels[0] == 1:
        rle.insert(0, 0)
    rle.extend([0, 0, w - 1, h - 1])
    return rle


def interact(req: dict) -> dict:
    st = state_for(req["image"])
    pos, neg, bbox = req.get("pos_points") or [], req.get("neg_points") or [], req.get("obj_bbox")
    kwargs = {"multimask_output": False}
    if pos or neg:
        kwargs["point_coords"] = np.array(pos + neg, dtype=np.float32)
        kwargs["point_labels"] = np.array([1] * len(pos) + [0] * len(neg), dtype=np.int32)
    if bbox:
        (x1, y1), (x2, y2) = bbox
        kwargs["box"] = np.array([min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)], dtype=np.float32)
    if len(kwargs) == 1:
        raise ValueError("click at least one point or draw a box")
    masks, _scores, _ = MODEL.predict_inst(st, **kwargs)
    mask = np.asarray(masks[0]) > 0
    return {"shapes": [{"type": "mask", "points": mask_to_rle(mask), "group": 0, "source": "semi-auto",
                        "attributes": [], "occluded": False, "rotation": 0}]}


class Handler(BaseHTTPRequestHandler):
    def _send(self, code: int, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):  # noqa: N802
        self._send(200, {"ok": PROC is not None, "model": "sam3", "device": str(getattr(PROC, "device", ""))})

    def do_POST(self):  # noqa: N802
        fn = {"/detect": detect, "/interact": interact}.get(self.path)
        if fn is None:
            return self._send(404, {"error": "unknown endpoint"})
        t = time.time()
        try:
            req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
            with LOCK:
                res = fn(req)
            n = len(res) if isinstance(res, list) else len(res["shapes"])
            print(f"{self.path} -> {n} shape(s) in {time.time() - t:.2f}s", flush=True)
            self._send(200, res)
        except Exception as e:  # noqa: BLE001 - report the error to CVAT instead of dropping the request
            traceback.print_exc()
            self._send(500, {"error": f"{type(e).__name__}: {e}"})

    def log_message(self, *args):
        pass


def main():
    global MODEL, PROC
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8766)
    a = ap.parse_args()
    from sam3.model.sam3_image_processor import Sam3Processor
    from sam3.model_builder import build_sam3_image_model

    t = time.time()
    MODEL = build_sam3_image_model(enable_inst_interactivity=True)
    PROC = Sam3Processor(MODEL)
    print(f"SAM3 ready on {PROC.device} ({time.time() - t:.1f}s); http://{a.host}:{a.port}", flush=True)
    ThreadingHTTPServer((a.host, a.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
