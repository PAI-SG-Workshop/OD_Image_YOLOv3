"""
YOLOv3 builders for the three tasks.

Everything here is algorithm-specific:
    - which .cfg (backbone + neck + head layout)
    - how [yolo] anchors are parsed
    - which detection layer (loss / target assignment) is used
    - how many box attributes the head regresses

Everything else (Darknet parsing, training, evaluation, demo) is common.
"""

from pathlib import Path

from common.darknet import Darknet
from detectors.yolov3.image_yolo_layer import YoloLayer as ImageYoloLayer

CFG_DIR = Path(__file__).resolve().parent / "cfg"


def _floats(text):
    return [float(v) for v in text.split(",")]


def _mask(block):
    return [int(v) for v in block["mask"].split(",")]


def image_factory(block, hyperparams):
    """Image YOLOv3: (w, h) anchors, global 9-anchor matching (legacy)."""
    v = _floats(block["anchors"])
    all_anchors = [(v[i], v[i + 1]) for i in range(0, len(v), 2)]
    mask = _mask(block)
    return ImageYoloLayer(
        [all_anchors[i] for i in mask],
        int(block["classes"]),
        int(hyperparams["height"]),
        all_anchors=all_anchors,
        anchor_mask=mask,
    )


SPECS = {
    #          cfg file                 factory          box attributes regressed
    "image":   ("image_yolov3.cfg",   image_factory,   5),   # x y w h conf
}


def build_model(task, num_classes):
    cfg_name, factory, num_box_attr = SPECS[task]
    return Darknet(CFG_DIR / cfg_name, factory, num_classes, num_box_attr)
