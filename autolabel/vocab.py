"""Label vocabularies for the SAM3 functions.

A CVAT function declares its labels up front; in CVAT's "Automatic annotation" dialog you map
your task's labels to these (same names are matched automatically). SAM3 runs one text prompt per
label, so smaller label sets are faster: register one function per label set you use.
"""

from __future__ import annotations

COCO80 = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog",
    "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
    "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
    "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
    "teddy bear", "hair drier", "toothbrush",
]

# Cabin monitoring (DMS / OMS), names follow the Cabin v1.0 schema where it has them.
CABIN = [
    "person", "face", "hand", "phone", "child_seat", "infant_seat", "object",
    "seatbelt", "steering_wheel", "cigarette", "bottle", "cup", "glasses", "sunglasses", "mask", "bag",
]

DRIVING = ["car", "truck", "bus", "motorcycle", "bicycle", "person", "traffic light", "traffic sign"]

PRESETS = {"coco": COCO80, "cabin": CABIN, "driving": DRIVING}

# Text prompts SAM3 understands better than the raw label name. A label can have several prompts;
# their results are merged.
PROMPTS = {
    "face": ["human face"],
    "phone": ["mobile phone", "smartphone"],
    "cell phone": ["mobile phone"],
    "child_seat": ["child car seat"],
    "infant_seat": ["infant car seat carrier"],
    "object": ["bag", "box", "cup"],
    "seatbelt": ["seat belt"],
    "steering_wheel": ["steering wheel"],
    "glasses": ["eyeglasses"],
    "mask": ["face mask"],
    "tv": ["television"],
    "hair drier": ["hair dryer"],
    "traffic sign": ["traffic sign", "road sign"],
}


def resolve(labels: str = "", preset: str = "coco") -> list[str]:
    """Comma-separated labels win over the preset."""
    if labels.strip():
        return [x.strip() for x in labels.split(",") if x.strip()]
    if preset not in PRESETS:
        raise ValueError(f"unknown preset {preset!r}; choose one of {sorted(PRESETS)} or pass labels=...")
    return list(PRESETS[preset])


def prompts_for(label: str) -> list[str]:
    return PROMPTS.get(label, [label.replace("_", " ")])
