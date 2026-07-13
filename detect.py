"""
detect.py
Core detection logic for VisionNote.

Wraps a trained YOLOv8 model and exposes a simple detect() function that
returns the best detection above a confidence threshold, plus helpers for
drawing boxes and mapping class names to spoken phrases.

Usage:
    from detect import CurrencyDetector

    detector = CurrencyDetector(model_path="model/trained_model.pt")
    result = detector.detect(frame)
    if result:
        print(result["label"], result["confidence"])
"""

from dataclasses import dataclass
from typing import Optional, List, Dict
import numpy as np
from ultralytics import YOLO
import cv2

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

# Minimum confidence to trust a detection enough to announce it out loud.
# Below this, we ask the user to reposition instead of guessing.
CONFIDENCE_THRESHOLD = 0.85

# Map your trained model's class names -> spoken phrases per language.
# Update the keys to match whatever class names you used while labeling
# your dataset (e.g. in Roboflow / LabelImg).
DENOMINATION_PHRASES: Dict[str, Dict[str, str]] = {
    "10":  {"en": "Ten Rupees",           "hi": "दस रुपये",        "ta": "பத்து ரூபாய்"},
    "20":  {"en": "Twenty Rupees",        "hi": "बीस रुपये",       "ta": "இருபது ரூபாய்"},
    "50":  {"en": "Fifty Rupees",         "hi": "पचास रुपये",      "ta": "ஐம்பது ரூபாய்"},
    "100": {"en": "One Hundred Rupees",   "hi": "एक सौ रुपये",     "ta": "நூறு ரூபாய்"},
    "200": {"en": "Two Hundred Rupees",   "hi": "दो सौ रुपये",     "ta": "இருநூறு ரூபாய்"},
    "500": {"en": "Five Hundred Rupees",  "hi": "पांच सौ रुपये",   "ta": "ஐநூறு ரூபாய்"},
}


@dataclass
class Detection:
    label: str          # raw class name, e.g. "500"
    confidence: float    # 0.0 - 1.0
    box: List[int]       # [x1, y1, x2, y2]


class CurrencyDetector:
    def __init__(self, model_path: str = "model/trained_model.pt", conf: float = CONFIDENCE_THRESHOLD):
        """
        model_path: path to your fine-tuned YOLOv8 .pt weights.
                    For quick testing before you finish training on your own
                    dataset, you can temporarily point this at a pretrained
                    yolov8n.pt just to confirm the pipeline runs end-to-end
                    (it won't know currency classes, but boxes will draw).
        conf: confidence threshold below which we treat detection as "unsure".
        """
        self.model = YOLO(model_path)
        self.conf_threshold = conf

    def detect(self, frame: np.ndarray) -> Optional[Detection]:
        """
        Run detection on a single BGR frame (as read by OpenCV).
        Returns the single highest-confidence detection above threshold,
        or None if nothing confident was found.
        """
        results = self.model.predict(frame, verbose=False)[0]

        if results.boxes is None or len(results.boxes) == 0:
            return None

        best = None
        for box in results.boxes:
            conf = float(box.conf[0])
            if best is None or conf > best.confidence:
                cls_id = int(box.cls[0])
                label = self.model.names[cls_id]
                xyxy = box.xyxy[0].tolist()
                best = Detection(label=label, confidence=conf, box=[int(v) for v in xyxy])

        if best is None or best.confidence < self.conf_threshold:
            return None

        return best

    def all_detections(self, frame: np.ndarray) -> List[Detection]:
        """
        Return every detection above threshold (for multi-note counting).
        """
        results = self.model.predict(frame, verbose=False)[0]
        detections = []
        if results.boxes is None:
            return detections

        for box in results.boxes:
            conf = float(box.conf[0])
            if conf < self.conf_threshold:
                continue
            cls_id = int(box.cls[0])
            label = self.model.names[cls_id]
            xyxy = box.xyxy[0].tolist()
            detections.append(Detection(label=label, confidence=conf, box=[int(v) for v in xyxy]))

        return detections


def get_phrase(label: str, lang: str = "en") -> str:
    """
    Map a raw class label (e.g. '500') to a spoken phrase in the requested
    language. Falls back to English, then to a generic phrase if the label
    is unrecognized.
    """
    entry = DENOMINATION_PHRASES.get(label)
    if entry is None:
        return f"Detected a note, denomination unclear"
    return entry.get(lang, entry.get("en", f"{label} Rupees"))


def draw_detection(frame: np.ndarray, detection: Detection, lang: str = "en") -> np.ndarray:
    """
    Draw the bounding box + label + confidence on the frame for the
    on-screen preview (separate from the voice announcement).
    """
    x1, y1, x2, y2 = detection.box
    color = (0, 200, 0) if detection.confidence >= CONFIDENCE_THRESHOLD else (0, 165, 255)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    text = f"{get_phrase(detection.label, lang)} ({detection.confidence*100:.1f}%)"
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw + 4, y1), color, -1)
    cv2.putText(frame, text, (x1 + 2, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    return frame


def total_amount(detections: List[Detection]) -> int:
    """
    Sum up denominations for the 'multiple note detection' bonus feature.
    Ignores labels that aren't valid integers (e.g. unrecognized classes).
    """
    total = 0
    for d in detections:
        try:
            total += int(d.label)
        except ValueError:
            continue
    return total
