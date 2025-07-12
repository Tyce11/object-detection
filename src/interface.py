import numpy as np
from ultralytics import YOLOE  # Ultralytics Python API for YOLOE

# Load YOLOE model
# This variant is fine-tuned for instance segmentation but will also give boxes.
# No prompts needed—it uses internal embeddings to detect 1,200+ categories zero-shot.
model = YOLOE("yoloe-11s-seg-pf.pt")  # Prompt-Free small model :contentReference[oaicite:0]{index=0}
model.eval()

def predict(frame: np.ndarray, conf_thresh: float = 0.25):
    """
    Runs prompt-free open-vocabulary detection on a BGR OpenCV frame.
    Returns a list of (xmin, ymin, xmax, ymax, confidence, class_id, label).
    """
    # convert BGR → RGB
    rgb_frame = frame[..., ::-1]

    # run inference (no text or visual prompts needed)
    results = model.predict(rgb_frame, conf=conf_thresh)  # :contentReference[oaicite:1]{index=1}

    detections = []
    # results is a list (one entry per image); here we only passed one frame
    r = results[0]
    for box in r.boxes:
        # box.xyxy is a 1×4 tensor, box.conf and box.cls are 1×1 tensors
        xmin, ymin, xmax, ymax = box.xyxy[0].tolist()
        conf_score = float(box.conf[0])
        cls_id     = int(box.cls[0])
        label      = model.names[cls_id]
        detections.append((xmin, ymin, xmax, ymax, conf_score, cls_id, label))

    return detections
