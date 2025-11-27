# scripts/detect_image.py
import sys
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO
from utils import annotate_frame, load_model

def detect_image(input_path: str, output_path: str, weights: str = "yolov8n.pt", conf: float = 0.25):
    model = load_model(weights)
    img = cv2.imread(input_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {input_path}")

    results = model.predict(source=input_path, conf=conf, imgsz=640)  # returns Results object
    res = results[0]  # single image
    names = model.model.names

    boxes = []
    scores = []
    class_ids = []
    if res.boxes is not None and len(res.boxes) > 0:
        for b in res.boxes:
            xyxy = b.xyxy.tolist()[0]  # [x1,y1,x2,y2]
            conf_score = float(b.conf[0])
            cls = int(b.cls[0])
            boxes.append(xyxy)
            scores.append(conf_score)
            class_ids.append(cls)

    annotated = annotate_frame(img.copy(), boxes, scores, class_ids, names)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(output_path, annotated)
    print(f"Saved annotated image to: {output_path}")

if __name__ == "__main__":
    # example: python scripts/detect_image.py samples/input.jpg outputs/detected.jpg
    if len(sys.argv) < 3:
        print("Usage: python detect_image.py <input_image> <output_image> [weights] [conf]")
        sys.exit(1)
    detect_image(sys.argv[1], sys.argv[2], weights=(sys.argv[3] if len(sys.argv) > 3 else "yolov8n.pt"),
                 conf=float(sys.argv[4]) if len(sys.argv) > 4 else 0.25)
