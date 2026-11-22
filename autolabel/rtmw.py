"""YOLOX person detection + RTMW whole-body keypoints (133 points) via rtmlib / onnxruntime (CPU)."""

from __future__ import annotations

import numpy as np

MODELS = {
    "performance": dict(
        det="https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/yolox_m_8xb8-300e_humanart-c2c7a14a.zip",
        det_size=(640, 640),
        pose="https://download.openmmlab.com/mmpose/v1/projects/rtmw/onnx_sdk/rtmw-dw-x-l_simcc-cocktail14_270e-384x288_20231122.zip",
        pose_size=(288, 384)),
    "lightweight": dict(
        det="https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/yolox_tiny_8xb8-300e_humanart-6f3252f9.zip",
        det_size=(416, 416),
        pose="https://download.openmmlab.com/mmpose/v1/projects/rtmw/onnx_sdk/rtmw-dw-l-m_simcc-cocktail14_270e-256x192_20231122.zip",
        pose_size=(192, 256)),
}


def _fwd(pts, w, h, k):  # original image coords -> np.rot90(k) image coords
    for _ in range(k):
        pts = np.stack([pts[:, 1], w - 1 - pts[:, 0]], 1)
        w, h = h, w
    return pts


def _inv(pts, w, h, k):  # np.rot90(k) image coords -> original image coords (w, h = original size)
    sizes, ww, hh = [], w, h
    for _ in range(k):
        sizes.append(ww)
        ww, hh = hh, ww
    for ww in reversed(sizes):
        pts = np.stack([ww - 1 - pts[:, 1], pts[:, 0]], 1)
    return pts


class Keypointer:
    def __init__(self, mode: str = "performance"):
        from rtmlib import YOLOX, RTMPose

        m = MODELS[mode]
        self.det = YOLOX(m["det"], model_input_size=m["det_size"], backend="onnxruntime", device="cpu")
        self.pose = RTMPose(m["pose"], model_input_size=m["pose_size"], backend="onnxruntime", device="cpu")

    def _run(self, bgr, det_thr):
        self.det.score_thr = det_thr
        boxes = self.det(bgr)
        boxes = [list(map(float, b)) for b in (boxes if len(boxes) else [])]
        if not boxes:
            return [], np.zeros((0, 133, 2)), np.zeros((0, 133))
        kp, sc = self.pose(bgr, bboxes=boxes)
        return boxes, kp, sc

    def __call__(self, rgb: np.ndarray, det_thr: float = 0.5, auto_rotate: bool = False):
        """Returns a list of (box_xyxy, keypoints[133,2], scores[133]) in original image coordinates."""
        bgr = np.ascontiguousarray(rgb[..., ::-1])
        h, w = bgr.shape[:2]
        rots = range(4) if auto_rotate else [0]
        best = None
        for k in rots:
            img = np.ascontiguousarray(np.rot90(bgr, k))
            boxes, kp, sc = self._run(img, det_thr)
            # score the orientation by how confident the body keypoints are (upright people score higher)
            q = float(sc[:, :17].mean()) * len(boxes) if len(boxes) else -1.0
            if best is None or q > best[0]:
                best = (q, k, boxes, kp, sc)
            if not auto_rotate:
                break
        _, k, boxes, kp, sc = best
        out = []
        for b, p, s in zip(boxes, kp, sc):
            pts = _inv(np.asarray(p, float), w, h, k)
            bb = _inv(np.array([[b[0], b[1]], [b[2], b[3]]], float), w, h, k)
            box = [bb[:, 0].min(), bb[:, 1].min(), bb[:, 0].max(), bb[:, 1].max()]
            out.append((box, pts, np.asarray(s, float)))
        return out
