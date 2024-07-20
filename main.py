import torch
import numpy as np
import cv2
import sys
import json
from ultralytics import YOLO

# load model
model = YOLO("./yolov5su.pt")
if model is None:
    print("Error: Model loading failed.")
    sys.exit(1)

# model config
model.conf = 0.25 # NMS confidence threshold
model.iou = 0.45 # NMS IoU threshold
model.agnostic = False # NMS class-agnostic
model.multi_label = False # NMS multiple labels per box
model.classes = None # (optional list) filter by class, i.e. = [0, 15, 16] for COCO persons, cats and dogs
model.max_det = 20 # maximum number of detections per image
model.amp = False # Automatic Mixed Precision (AMP) inference

def draw_bounding_box(output_image, detections):
    for detection in detections:
        xmin, ymin, xmax, ymax, confidence, label = detection[:6] 

        # Draw bounding box
        cv2.rectangle(output_image, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 255, 0), 2)

        # Centroid Coordinates of detected object
        cx = int((xmin + xmax) / 2)
        cy = int((ymin + ymax) / 2)

        # Draw centroid
        cv2.circle(output_image, (cx, cy), 2, (0, 0, 255), 2, cv2.FILLED)
        cv2.putText(output_image, f"({cx}, {cy})", (cx - 40, cy + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
        cv2.putText(output_image, f"{label}, {confidence}", (int(xmin), int(ymin) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

    return output_image

# Start reading camera feed
cap = cv2.VideoCapture("http://192.168.43.1:4747/video")
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture image")
        break

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # OpenCV image (BGR to RGB)

    # Inference
    results = model(image)  # includes NMS

    all_detections = []
    for res in results:
        # Accessing bounding boxes and confidence scores directly from the Results object
        boxes = res.boxes.xyxy.cpu().numpy()  # Extract bounding box coordinates
        confidences = res.boxes.conf.cpu().numpy()  # Extract confidence scores
        labels = res.boxes.cls.cpu().numpy()

        for box, confidence, label in zip(boxes, confidences, labels):
            xmin, ymin, xmax, ymax = box
            all_detections.append([xmin, ymin, xmax, ymax, confidence, model.names[int(label)]])

    # Render detections on image
    output_image = draw_bounding_box(frame.copy(), all_detections)

    cv2.imshow("Output", output_image)  # Show the output image after rendering

    # Exit by pressing 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows() 