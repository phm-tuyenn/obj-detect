import torch
import numpy as np
import cv2
import sys
import json, requests
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

# code config
CENTER_RANGE = 30
BOT_HOST = ""
CAM_HOST = 1

# Get the target object
print("This is the list of objects: ")
for index in model.names.keys():
    print(f"Index {index}: {model.names[index]}")
target = int(input("Type the index number of target object: "))
print(f"You choose {model.names[target]}. Ready for a journey!!!")

def draw_bounding_box(output_image, detections):
    centered_obj = 0
    center_x = int(output_image.shape[1] / 2)
    y = output_image.shape[0]

    cv2.line(output_image, (center_x - CENTER_RANGE, 0), (center_x - CENTER_RANGE, y), (255, 0, 0), 2)
    cv2.line(output_image, (center_x + CENTER_RANGE, 0), (center_x + CENTER_RANGE, y), (255, 0, 0), 2)

    for detection in detections:
        xmin, ymin, xmax, ymax, confidence, label = detection[:6] 

        # Draw bounding box
        cv2.rectangle(output_image, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 255, 0), 2)

        # Centroid Coordinates of detected object
        cx = int((xmin + xmax) / 2)
        cy = int((ymin + ymax) / 2)

        # Check if object in centered position (for debug)
        if center_x - CENTER_RANGE <= cx <= center_x + CENTER_RANGE:
            centered_obj += 1
        
        # Check if target object in centered position or not to give move
        if target == label:
            if center_x - CENTER_RANGE <= cx <= center_x + CENTER_RANGE:
                move_the_bot(1)
            elif cx < center_x - CENTER_RANGE:
                move_the_bot(3)
            elif center_x - CENTER_RANGE < cx:
                move_the_bot(4)

        # Draw centroid
        cv2.circle(output_image, (cx, cy), 2, (0, 0, 255), 2, cv2.FILLED)
        cv2.putText(output_image, f"({cx}, {cy})", (cx - 40, cy + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
        
        # Draw object name and confidence
        cv2.putText(output_image, f"{model.names[label]}, {confidence}", (int(xmin), int(ymin) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

    return [output_image, centered_obj]

def move_the_bot(movement_code):
    # 1 forward, 2 backward, 3 left, 4 right
    print(movement_code)
    # requests.post(BOT_HOST, json = { 'move': movement_code })

# Start reading camera feed
cap = cv2.VideoCapture(CAM_HOST)
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
            all_detections.append([xmin, ymin, xmax, ymax, confidence, int(label)])

    # Render detections on image
    output_image, centered_objs = draw_bounding_box(frame.copy(), all_detections)

    frame_name = f"{len(list(zip(boxes, confidences, labels)))} objects found. {centered_objs} objects in centered position."
    cv2.imshow("frame", output_image)  # Show the output image after rendering
    cv2.setWindowTitle("frame", frame_name)

    # Exit by pressing 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows() 