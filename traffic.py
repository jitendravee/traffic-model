


import cv2
import numpy as np
import os

# Load YOLO
cfg_path = '/Volumes/sandisk_1/python/traffic-ai/cfg/yolov3.cfg'
weights_path = '/Volumes/sandisk_1/python/traffic-ai/bin/yolov3.weights'

# Load the YOLO network
net = cv2.dnn.readNetFromDarknet(cfg_path, weights_path)
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Load class labels from coco.names
with open('/Volumes/sandisk_1/python/traffic-ai/cfg/coco.names', 'r') as f:
    classes = [line.strip() for line in f.readlines()]

def detectVehiclesInVideo(video_path):
    # Open video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video file.")
        return
    
    vehicle_count = 0  # Initialize vehicle count
    detected_boxes = []  # List to store previously detected vehicle boxes

    while True:
        # Read each frame of the video
        ret, frame = cap.read()
        if not ret:
            break  # Exit the loop if the video ends

        height, width, channels = frame.shape

        # Prepare the image for YOLO
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(output_layers)

        # Information to store detected vehicles
        class_ids = []
        confidences = []
        boxes = []

        # Loop through detections
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.3:  # Confidence threshold
                    # Get bounding box
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    # Rectangle coordinates
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        # Perform non-maxima suppression to reduce overlapping boxes
        indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.3, 0.4)

        # Draw bounding boxes and labels
        if len(indices) > 0:
            for i in indices.flatten():
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]])

                if label in ["car", "bus", "bike", "truck", "person"]:  # Filter for vehicle labels
                    color = (0, 255, 0)  # Green color for bounding boxes

                    # Check if the vehicle (bounding box) is already detected
                    vehicle_detected = False
                    for detected_box in detected_boxes:
                        # Calculate intersection over union (IoU) to check if the vehicle is the same
                        x1, y1, w1, h1 = detected_box
                        iou = compute_iou([x, y, w, h], [x1, y1, w1, h1])

                        # If IoU is high, we consider it the same vehicle
                        if iou > 0.5:
                            vehicle_detected = True
                            break
                    
                    if not vehicle_detected:
                        # If the vehicle is not detected already, count it and add its box to the list
                        detected_boxes.append([x, y, w, h])
                        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 3)
                        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                        vehicle_count += 1  # Increment the vehicle count

        # Display vehicle count on the frame
        cv2.putText(frame, f"Vehicles: {vehicle_count}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Show the frame in real-time
        cv2.imshow("Detected Vehicles", frame)

        # Exit loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release video capture and close all windows
    cap.release()
    cv2.destroyAllWindows()
    print(f"Total vehicles detected in video: {vehicle_count}")

def compute_iou(box1, box2):
    # Compute the Intersection over Union (IoU) for two bounding boxes
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    # Calculate the (x, y)-coordinates of the intersection rectangle
    x_intersection = max(x1, x2)
    y_intersection = max(y1, y2)
    w_intersection = max(0, min(x1 + w1, x2 + w2) - x_intersection)
    h_intersection = max(0, min(y1 + h1, y2 + h2) - y_intersection)

    # Area of intersection
    intersection_area = w_intersection * h_intersection

    # Area of both bounding boxes
    box1_area = w1 * h1
    box2_area = w2 * h2

    # IoU is the intersection area divided by the union area
    iou = intersection_area / float(box1_area + box2_area - intersection_area)
    return iou

# Define path to the video files
inputPath = '/Volumes/sandisk_1/python/traffic-ai/test_images/'
outputPath = '/Volumes/sandisk_1/python/traffic-ai/output_images/'

# Process all videos in the input directory
for filename in os.listdir(inputPath):
    if filename.endswith((".jpg",  ".avi")):  # Process video files
        video_path = os.path.join(inputPath, filename)
        print(f"Processing video: {filename}")
        detectVehiclesInVideo(video_path)

print("Done!")
