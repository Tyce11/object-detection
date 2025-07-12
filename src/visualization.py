import cv2

# Map some class names to custom colors
CLASS_COLORS = {
    'person': (0,0,200),
    'dog':    (200,0,0),
    'cat':    (0,200,0),
}

# Handles drawing the overlay
def draw_detections(frame, detections, inst_fps, avg_fps):
    """
    frame       BGR image
    detections  list of (xmin,ymin,xmax,ymax,conf,cls,label)
    inst_fps    instantaneous FPS
    avg_fps     average FPS
    """
    for xmin, ymin, xmax, ymax, conf, cls, label in detections:
        color = CLASS_COLORS.get(label, (0,255,255))
        # draw box
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
        # assemble label text
        text = f"{label} {conf:.2f}"
        # compute text size so background can scale
        (w,h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        # filled rectangle behind text for readability
        cv2.rectangle(frame, (xmin, ymin - h - 6), (xmin + w, ymin), color, -1)
        # draw the label
        cv2.putText(frame, text, (xmin, ymin - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
    # draw FPS
    cv2.putText(frame, f"FPS: {inst_fps:.1f}", (10,20),
                cv2.FONT_HERSHEY_PLAIN, 1.0, (200,200,200), 1)
    cv2.putText(frame, f"AVG: {avg_fps:.1f}", (10,40),
                cv2.FONT_HERSHEY_PLAIN, 1.0, (200,200,200), 1)