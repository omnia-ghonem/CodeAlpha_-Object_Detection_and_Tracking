import cv2
import os
import cv2
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# Load pretrained YOLOv8 model
model = YOLO("yolov8l.pt")
tracker = DeepSort(max_age=30)
cwd = os.getcwd() 
video_path = os.path.join(cwd, "traffic-video.mp4")

#################################q
# Open Video
#################################

cap = cv2.VideoCapture(video_path)
# or For webcam: cap = cv2.VideoCapture(0)


#################################
#Process Video Frames
#################################

while True:

    ret, frame = cap.read()

    if not ret:
        break

    results = model(frame)

    detections = []

    for result in results:

        boxes = result.boxes

        for box in boxes:

            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

            confidence = float(box.conf[0])

            class_id = int(box.cls[0])

            label = model.names[class_id]

            detections.append(
                (
                    [x1, y1, x2 - x1, y2 - y1],
                    confidence,
                    label
                )
            )

    tracks = tracker.update_tracks(detections, frame=frame)

    for track in tracks:

        if not track.is_confirmed():
            continue

        track_id = track.track_id

        ltrb = track.to_ltrb()

        x1, y1, x2, y2 = map(int, ltrb)

        label = track.get_det_class()

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)

        cv2.putText(
            frame,
            f"{label} ID:{track_id}",
            (x1, y1-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0,255,0),
            2
        )

    cv2.imshow("Object Detection and Tracking", frame)

    if cv2.waitKey(2) & 0xFF == ord('q'):
        break



##############################################
# to Activate the virtual environment, run the following command in your terminal:
##############################################

# source venv/bin/activate  # On macOS/Linux
# venv\Scripts\activate  # On Windows

##############################################
# To install the required packages, run the following command in your terminal:
# pip install -r requirements.txt
##############################################

##############################################
#Release Resources
##############################################

cap.release()
cv2.destroyAllWindows()