"""Skeleton definitions (keypoint names, edges, CVAT SVG layout) for the RTMW keypoint functions.

Index ranges refer to RTMW's 133 COCO-WholeBody keypoints:
0-16 body, 17-22 feet, 23-90 face (68, iBUG/300W order), 91-111 left hand, 112-132 right hand.
left / right are always the person's own sides.
"""

from __future__ import annotations

import math

# ----------------------------------------------------------------------------- body (COCO-17)
BODY17 = ["nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_shoulder", "right_shoulder",
          "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_hip", "right_hip",
          "left_knee", "right_knee", "left_ankle", "right_ankle"]
BODY17_EDGES = [(15, 13), (13, 11), (16, 14), (14, 12), (11, 12), (5, 11), (6, 12), (5, 6), (5, 7),
                (6, 8), (7, 9), (8, 10), (1, 2), (0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6)]
# template drawn facing the viewer: the person's left side is on the right of the drawing
BODY17_XY = [(50, 10), (54, 7), (46, 7), (58, 9), (42, 9), (62, 25), (38, 25), (68, 42), (32, 42),
             (72, 57), (28, 57), (58, 58), (42, 58), (60, 77), (40, 77), (61, 95), (39, 95)]

# ----------------------------------------------------------------------------- face (68, iBUG order)
FACE68 = [f"p{i:02d}" for i in range(1, 69)]
FACE68_EDGES = ([(i, i + 1) for i in range(0, 16)] + [(i, i + 1) for i in range(17, 21)]
                + [(i, i + 1) for i in range(22, 26)] + [(i, i + 1) for i in range(27, 30)]
                + [(i, i + 1) for i in range(31, 35)]
                + [(i, i + 1) for i in range(36, 41)] + [(41, 36)]
                + [(i, i + 1) for i in range(42, 47)] + [(47, 42)]
                + [(i, i + 1) for i in range(48, 59)] + [(59, 48)]
                + [(i, i + 1) for i in range(60, 67)] + [(67, 60)])


def _face68_xy():
    pts = []
    for i in range(17):  # jaw: image-left to image-right
        a = math.pi * (1 - i / 16)
        pts.append((50 + 40 * math.cos(a), 45 + 45 * math.sin(a) * 0.95))
    pts += [(20 + 5 * i, 28 - (2 if 0 < i < 4 else 0)) for i in range(5)]   # brow (image left)
    pts += [(60 + 5 * i, 28 - (2 if 0 < i < 4 else 0)) for i in range(5)]   # brow (image right)
    pts += [(50, 38 + 5 * i) for i in range(4)]                                # nose bridge
    pts += [(42 + 4 * i, 60 - (2 if i == 2 else 0)) for i in range(5)]         # nostrils
    for cx in (33, 67):                                                        # eyes
        pts += [(cx - 8, 40), (cx - 3, 37), (cx + 3, 37), (cx + 8, 40), (cx + 3, 43), (cx - 3, 43)]
    for i in range(12):                                                        # outer lips
        a = math.pi + 2 * math.pi * i / 12
        pts.append((50 + 15 * math.cos(a), 74 + 7 * math.sin(a)))
    for i in range(8):                                                         # inner lips
        a = math.pi + 2 * math.pi * i / 8
        pts.append((50 + 10 * math.cos(a), 74 + 3 * math.sin(a)))
    return pts


FACE68_XY = _face68_xy()

# ----------------------------------------------------------------------------- face13 (Cabin v1.0)
# name -> 68-point indices (0-based) averaged; iBUG 36-41 is the person's RIGHT eye.
FACE13_FROM68 = {
    "left_eye_outer": [45], "left_eye_inner": [42], "left_eye_upper": [43, 44], "left_eye_lower": [47, 46],
    "right_eye_outer": [36], "right_eye_inner": [39], "right_eye_upper": [37, 38], "right_eye_lower": [41, 40],
    "nose_tip": [30], "mouth_left": [54], "mouth_right": [48], "lip_upper": [51], "lip_lower": [57],
}
FACE13 = list(FACE13_FROM68)
FACE13_EDGES = [(0, 2), (0, 3), (1, 2), (1, 3), (4, 6), (4, 7), (5, 6), (5, 7), (9, 11), (9, 12), (10, 11),
                (10, 12), (8, 11)]
FACE13_XY = [(80, 30), (60, 30), (70, 25), (70, 35), (20, 30), (40, 30), (30, 25), (30, 35), (50, 55),
             (65, 75), (35, 75), (50, 70), (50, 82)]

# ----------------------------------------------------------------------------- hand (21)
HAND21 = ["wrist"] + [f"{f}_{j}" for f in ("thumb", "index", "middle", "ring", "pinky") for j in range(1, 5)]
HAND21_EDGES = [(0, 1 + 4 * f) for f in range(5)] + [(1 + 4 * f + j, 2 + 4 * f + j) for f in range(5) for j in range(3)]


def _hand_xy():
    pts = [(50, 92)]
    for f, ang in enumerate((-60, -25, -5, 15, 35)):
        a = math.radians(ang - 90)
        base = 18 if f else 12
        for j in range(4):
            r = base + 16 * (j + 1) * (0.8 if f == 0 else 1)
            pts.append((50 + r * math.cos(a), 92 + r * math.sin(a)))
    return [(min(98, max(2, x)), min(98, max(2, y))) for x, y in pts]


HAND21_XY = _hand_xy()

KINDS = {
    "body17": dict(label="pose", names=BODY17, edges=BODY17_EDGES, xy=BODY17_XY),
    "face68": dict(label="face68", names=FACE68, edges=FACE68_EDGES, xy=FACE68_XY),
    "face13": dict(label="face13", names=FACE13, edges=FACE13_EDGES, xy=FACE13_XY),
    "hand21": dict(label="hand", names=HAND21, edges=HAND21_EDGES, xy=HAND21_XY),
}


def svg(kind: str, ids: list[int] | None = None) -> str:
    """CVAT skeleton SVG. data-label-id must match the sublabel ids used in the spec."""
    k = KINDS[kind]
    ids = ids or list(range(len(k["names"])))
    parts = []
    for a, b in k["edges"]:
        (x1, y1), (x2, y2) = k["xy"][a], k["xy"][b]
        parts.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" data-type="edge" '
                     f'data-node-from="{a + 1}" data-node-to="{b + 1}"></line>')
    for i, (x, y) in enumerate(k["xy"]):
        parts.append(f'<circle r="0.75" cx="{x:.1f}" cy="{y:.1f}" data-type="element node" '
                     f'data-element-id="{i + 1}" data-node-id="{i + 1}" data-label-id="{ids[i]}"></circle>')
    return "".join(parts)


def cvat_raw_label(kind: str) -> dict:
    """Label entry for CVAT's Raw label editor (Project/Task > Labels > Raw)."""
    k = KINDS[kind]
    entry = dict(name=k["label"], type="skeleton", attributes=[],
                 sublabels=[dict(name=n, type="points", attributes=[]) for n in k["names"]],
                 svg=svg(kind).replace('data-label-id="', 'data-label-name="'))
    # Raw editor resolves nodes by name for new labels
    names = k["names"]
    for i, n in enumerate(names):
        entry["svg"] = entry["svg"].replace(f'data-element-id="{i + 1}" data-node-id="{i + 1}" data-label-name="{i}"',
                                            f'data-element-id="{i + 1}" data-node-id="{i + 1}" data-label-name="{n}"')
    if kind == "hand21":
        entry["attributes"] = [dict(name="side", mutable=False, input_type="select",
                                    values=["left", "right"], default_value="left")]
    return entry
