import cv2

CLASS_COLORS = {

}

def draw_detections(frame, detections, inst_fps, avg_fps):
    """
    Overlay bounding boxes, labels, and FPS on the frame.

    frame       : BGR image (numpy array)
    detections  : list of (xmin, ymin, xmax, ymax, conf, cls, label)
    inst_fps    : instantaneous FPS (float)
    avg_fps     : average FPS (float)
    """
    for xmin, ymin, xmax, ymax, conf, cls, label in detections:
        # cast coordinates to int for OpenCV
        x1, y1, x2, y2 = map(int, (xmin, ymin, xmax, ymax))
        color = CLASS_COLORS.get(label, (255,0,0))  # default to blue
        # draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        # prepare label text
        text = f"{label} {conf:.2f}"
        # calculate text size
        (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        # draw filled rectangle background for text
        cv2.rectangle(frame, (x1, y1 - h - 6), (x1 + w, y1), color, -1)
        # draw text itself
        cv2.putText(frame, text, (x1, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
    
    # overlay FPS stats if provided
    if inst_fps is not None:
        cv2.putText(frame, f"FPS: {inst_fps:.1f}", (10, 20),
                    cv2.FONT_HERSHEY_PLAIN, 1.0, (200,200,200), 1)
    if avg_fps is not None:
        cv2.putText(frame, f"AVG: {avg_fps:.1f}", (10, 40),
                    cv2.FONT_HERSHEY_PLAIN, 1.0, (200,200,200), 1)
