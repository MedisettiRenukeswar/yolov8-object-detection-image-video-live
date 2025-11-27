# scripts/detect_video.py
import sys
from pathlib import Path
import cv2
from ultralytics import YOLO
from utils import load_model, annotate_frame

def detect_video(input_video: str, output_video: str, weights: str = "yolov8n.pt", conf: float = 0.25):
    model = load_model(weights)
    cap = cv2.VideoCapture(input_video)
    assert cap.isOpened(), f"Cannot open video {input_video}"
    fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    Path(output_video).parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_video, fourcc, fps, (w, h))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(source=frame, conf=conf, imgsz=640)
        res = results[0]
        boxes, scores, class_ids = [], [], []
        names = model.model.names
        if res.boxes is not None and len(res.boxes) > 0:
            for b in res.boxes:
                boxes.append(b.xyxy.tolist()[0])
                scores.append(float(b.conf[0]))
                class_ids.append(int(b.cls[0]))

        annotated = annotate_frame(frame, boxes, scores, class_ids, names)
        out.write(annotated)

    cap.release()
    out.release()
    print(f"Saved annotated video to: {output_video}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python detect_video.py <input_video> <output_video> [weights] [conf]")
        sys.exit(1)
    detect_video(sys.argv[1], sys.argv[2], weights=(sys.argv[3] if len(sys.argv) > 3 else "yolov8n.pt"),
                 conf=float(sys.argv[4]) if len(sys.argv) > 4 else 0.25)
