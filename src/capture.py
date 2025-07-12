import cv2
import time
import logging
from interface import predict
from visualization import draw_detections

logging.basicConfig(
    filename='logs/performance.log',      # log file path
    filemode='w',                    # overwrite (use 'a' to append)
    level=logging.INFO,              # capture INFO and above
    format='%(asctime)s %(message)s',# include timestamp
    datefmt='%Y-%m-%d %H:%M:%S'      # human-readable time
)

import warnings
warnings.filterwarnings("ignore", category=FutureWarning) # get rid of annoying warnings


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logging.error("Cannot open camera")
        return

    
    start_time = time.time()
    prev_time  = start_time
    frame_count = 0

    while True:
        t0 = time.time() # Track time before first frame
        ret, frame = cap.read()
        if not ret:
            break
        t1 = time.time() # Track time after first frame

        now = time.time()
        delta = now - prev_time
        prev_time = now

        # instantaneous FPS
        inst_fps = 1.0 / delta if delta > 0 else 0.0

        # average FPS
        frame_count += 1
        elapsed = now - start_time
        avg_fps = frame_count / elapsed if elapsed > 0 else 0.0

        # Object detection
        dets = predict(frame, conf_thresh=0.3)
        t2 = time.time() # Track time after detection

        # Overlay for detections
        draw_detections(frame, dets, inst_fps, avg_fps)
        t3 = time.time() # Track time after overlay

        cv2.imshow("Camera Feed", frame)
        t4 = time.time() # Track time after showing

        # Logs time to see time it takes
        logging.info(
            f"capture {(t1-t0)*1000:5.1f}ms | "
            f"infer {(t2-t1)*1000:5.1f}ms | "
            f"draw {(t3-t2)*1000:5.1f}ms | "
            f"show {(t4-t3)*1000:5.1f}ms | "
            f"FPS {inst_fps:.1f}/{avg_fps:.1f}"
        )

        prev_time = t1 # use our timing measure to set prev time

        # Ends the program when q is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
