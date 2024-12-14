import platform
from gpiozero import LED
import cv2
import numpy as np
import time

# Initialize the buzzer (LED) pin for simulation if on Raspberry Pi
if platform.system() == "Linux":
    buzzer_pin = LED(17)  # Use GPIO pin 17 for testing
else:
    print("Running on a non-Raspberry Pi system, GPIO functionality is skipped.")
    buzzer_pin = None  # No GPIO functionality on Windows

# Load YOLO model
net = cv2.dnn.readNet("D:/VEHICLE_DOOR_SAFETY/yolo_model/yolov3-tiny.weights", 
                      "D:/VEHICLE_DOOR_SAFETY/yolo_model/yolov3-tiny.cfg")

# Load the COCO names
with open("D:/VEHICLE_DOOR_SAFETY/yolo_model/coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Initialize video capture from the webcam
vs = cv2.VideoCapture(0)

while True:
    ret, frame = vs.read()
    if not ret:
        break

    # Prepare the frame for YOLO
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(net.getUnconnectedOutLayersNames())

    # Process the outputs
    class_ids = []
    confidences = []
    boxes = []
    height, width, channels = frame.shape

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:  # Only consider detections with confidence > 0.5
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Coordinates for rectangle
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Simulate buzzer control (if GPIO is available)
    if len(indexes) > 0 and buzzer_pin:
        buzzer_pin.on()  # Turn buzzer (LED) ON
        time.sleep(0.5)  # Beep duration
        buzzer_pin.off()  # Turn buzzer (LED) OFF

    # Draw bounding boxes and labels on the frame
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            color = (0, 255, 0)  # Green color for bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Show the live feed
    cv2.imshow("Image", frame)

    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the resources
vs.release()
cv2.destroyAllWindows()
