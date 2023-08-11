import cv2
import cvzone
import math
import argparse
import numpy as np
import supervision as sv
# import pandas as pd
# import torch
# import time

from ultralytics import YOLO

model = YOLO('Yolo-Weights/yolov8n.pt')

def Display(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        colorsBGR = [x, y]
        print(colorsBGR)


cv2.namedWindow('Display')
cv2.setMouseCallback('Display', Display)

# Class name object
my_file = open("coco.txt", "r")
data = my_file.read()
classNames = data.split("\n")


# Frame set
def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Camera")
    parser.add_argument(
        "--camera-resolution",
        default=[1280, 720],
        nargs=2,
        type=int
    )
    args = parser.parse_args()
    return args


# polygon zone
ZONE_POLYGON = np.array([
    [450, 250],
    [700, 250],
    [770, 550],
    [100, 550]
])

def main():
    args = parse_arguments()
    frame_width, frame_height = args.camera_resolution

    cap = cv2.VideoCapture('Videos/cars.mp4')  # Video
    # cap = cv2.VideoCapture(0)  # Camera
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

    zone = sv.PolygonZone(polygon=ZONE_POLYGON, frame_resolution_wh=tuple(args.camera_resolution))
    zone_annotator = sv.PolygonZoneAnnotator(zone=zone, color=sv.Color.red())
    while True:
        ret, frame = cap.read()
        frame = zone_annotator.annotate(scene=frame)
        if not ret:
            break
        results = model(frame, stream=True)
        frame = zone_annotator.annotate(scene=frame)
        for r in results:
            boxes = r.boxes
            for box in boxes:
                # Bounding Box
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                # for CV2
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 1)
                # for cvzone
                w, h = x2 - x1, y2 - y1
                # Confidence
                conf = math.ceil((box.conf[0] * 100)) / 100
                # Class Name
                cls = int(box.cls[0])
                currentClass = classNames[cls]
                # Detect an object
                if currentClass == "car" and conf > 0.3:
                    cvzone.putTextRect(frame,
                                       f'{currentClass} {conf}',
                                       (max(0, x1),
                                        max(35, y1)),
                                       scale=1,
                                       thickness=1,
                                       offset=5)
                    cvzone.cornerRect(frame, (x1, y1, w, h), l=10)
        cv2.imshow("Display", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
