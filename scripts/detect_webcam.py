# scripts/detect_webcam.py
import cv2
import time
from scripts.utils import load_model, annotate_frame

def detect_webcam(source=0, weights="yolov8n.pt", conf=0.25, output_path: str = None, show_window: bool = True):
    """
    Run YOLOv8 on webcam and optionally save annotated output to MP4.

    Args:
        source: webcam index (0) or path to video file
        weights: YOLO weights file (e.g., "yolov8n.pt" or "models/custom.pt")
        conf: confidence threshold (0-1)
        output_path: if provided (e.g. "outputs/webcam_out.mp4"), the annotated video will be written
        show_window: if True, display live window; set False for headless recording
    """
    model = load_model(weights)
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video source: {source}")

    # Query capture properties (fallback to defaults if invalid)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 0 or fps != fps:  # also check NaN
        fps = 20.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)

    writer = None
    if output_path:
        # Ensure parent directory exists
        import os
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
        if not writer.isOpened():
            writer = None
            print(f"[WARN] Could not open VideoWriter for {output_path}. Continuing without saving.")

    names = model.model.names

    print("[INFO] Running YOLOv8 webcam. Press 'q' to quit. Recording:" , bool(writer))
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[INFO] End of stream or cannot read frame.")
                break

            # Optionally resize for speed (uncomment if needed)
            # frame_small = cv2.resize(frame, (int(w*0.75), int(h*0.75)))

            # Run model on the frame
            results = model.predict(source=frame, conf=conf, imgsz=640)
            res = results[0]

            boxes, scores, class_ids = [], [], []
            if res.boxes is not None and len(res.boxes) > 0:
                for b in res.boxes:
                    boxes.append(b.xyxy.tolist()[0])
                    scores.append(float(b.conf[0]))
                    class_ids.append(int(b.cls[0]))

            annotated = annotate_frame(frame.copy(), boxes, scores, class_ids, names)

            # Show window
            if show_window:
                cv2.imshow("YOLOv8 Webcam", annotated)

            # Write to file if recording
            if writer is not None:
                # Ensure frame is same size as writer expects
                if (annotated.shape[1], annotated.shape[0]) != (w, h):
                    annotated_resized = cv2.resize(annotated, (w, h))
                    writer.write(annotated_resized)
                else:
                    writer.write(annotated)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user.")
    finally:
        cap.release()
        if writer is not None:
            writer.release()
            print(f"[INFO] Saved annotated video to: {output_path}")
        cv2.destroyAllWindows()
        print("[INFO] Clean exit.")

if __name__ == "__main__":
    # Example usage:
    # 1) Live view only:
    #    python -m scripts.detect_webcam
    #
    # 2) Live + record to outputs/webcam_out.mp4:
    #    python -m scripts.detect_webcam outputs/webcam_record.mp4 -- (see below)
    #
    # You can also call programmatically:
    # detect_webcam(source=0, weights="yolov8n.pt", conf=0.25, output_path="outputs/webcam_out.mp4")

    # Minimal CLI parsing to let you pass output path via sys.argv (keeps it simple)
    import sys
    out = None
    if len(sys.argv) > 1:
        # first arg can be output path
        out = sys.argv[1]
    detect_webcam(0, "yolov8n.pt", 0.25, output_path=out, show_window=True)
