# scripts/utils.py
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO

def load_model(weights: str = "yolov8n.pt"):
    """
    Load a YOLOv8 model. Default: tiny 'yolov8n.pt' (fast).
    Place custom weights in models/ and pass the path.
    """
    model = YOLO(weights)
    return model

def annotate_frame(frame, boxes, scores, class_ids, class_names):
    """
    Draw bounding boxes + labels on an OpenCV BGR frame.
    boxes: list of [x1, y1, x2, y2]
    scores: list of confidences
    class_ids: list of ints
    class_names: list/dict mapping id->name
    """
    for box, score, cid in zip(boxes, scores, class_ids):
        x1, y1, x2, y2 = map(int, box)
        label = f"{class_names[int(cid)]} {score:.2f}"
        color = (0, 160, 255)  # BGR
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - 18), (x1 + tw + 6, y1), color, -1)
        cv2.putText(frame, label, (x1 + 3, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return frame
