# src/inference.py
import torch
import numpy as np

# load the small, pretrained YOLOv5s model
# first call will download weights (~14 MB)
_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
_model.eval()

def predict(frame: np.ndarray, conf_thresh: float = 0.25):
    """
    Runs object detection on a BGR OpenCV frame.
    Returns a list of detections: [ (xmin, ymin, xmax, ymax, conf, class_id, label), … ]
    """
    # BGR → RGB, convert to list of frames
    results = _model(frame[..., ::-1])  
    detections = []
    for *box, conf, cls in results.xyxy[0].cpu().numpy():
        if conf < conf_thresh:
            continue
        xmin, ymin, xmax, ymax = map(int, box)
        label = results.names[int(cls)]
        detections.append((xmin, ymin, xmax, ymax, float(conf), int(cls), label))
    return detections
