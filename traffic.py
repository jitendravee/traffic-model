

import math
import time
import cv2
import numpy as np
import requests
import threading

# Default traffic light settings
DEFAULT_YELLOW = 5
DEFAULT_GREEN = 15
DEFAULT_MINIMUM = 10
DEFAULT_MAXIMUM = 60
DEFAULT_RED = 130

# Vehicle time allocation per type
VEHICLE_TIMES = {
    "cars": 2,
    "bikes": 1,
    "buses": 3,
    "trucks": 4,
    "rickshaws": 2,
}

# Video feeds (replace with actual video file paths)
VIDEO_FEEDS = [
    "/Volumes/sandisk_1/python/traffic-ai/test_images/1.jpg",
    "/Volumes/sandisk_1/python/traffic-ai/test_images/2.jpg",
    "/Volumes/sandisk_1/python/traffic-ai/test_images/3.jpg",
]

# Load YOLO model
net = cv2.dnn.readNet("/Volumes/sandisk_1/python/traffic-ai/bin/yolov3.weights", "/Volumes/sandisk_1/python/traffic-ai/cfg/yolov3.cfg")
layer_names = net.getLayerNames()
out_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]


# Load COCO class labels
with open("/Volumes/sandisk_1/python/traffic-ai/cfg/coco.names", "r") as f:
    CLASSES = f.read().strip().split("\n")

# TrafficSignal class
class TrafficSignal:
    def __init__(self, red, yellow, green):
        self.red = red
        self.yellow = yellow
        self.green = green

    def update(self, red=None, yellow=None, green=None):
        if red is not None:
            self.red = red
        if yellow is not None:
            self.yellow = yellow
        if green is not None:
            self.green = green

# Initialize traffic signals
signals = [TrafficSignal(DEFAULT_RED, DEFAULT_YELLOW, DEFAULT_GREEN) for _ in range(len(VIDEO_FEEDS))]
current_signal = 0

# Detect vehicles in a video feed
def detect_vehicles(video):
    cap = cv2.VideoCapture(video)
    vehicle_counts = {key: 0 for key in VEHICLE_TIMES.keys()}

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        height, width, _ = frame.shape
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outputs = net.forward(out_layers)

        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    label = str(CLASSES[class_id])
                    if label in VEHICLE_TIMES.keys():
                        vehicle_counts[label + "s"] += 1

        break
      # Process only the first frame for faster detection

    cap.release()
    return vehicle_counts

# Calculate green light duration based on vehicle counts
def calculate_green_time(video_index):
    vehicle_counts = detect_vehicles(VIDEO_FEEDS[video_index])
    print(f"Vehicle counts for Signal {video_index + 1}: {vehicle_counts}")
    green_time = sum(vehicle_counts[vehicle] * VEHICLE_TIMES[vehicle] for vehicle in vehicle_counts)
    return max(DEFAULT_MINIMUM, min(math.ceil(green_time / 2), DEFAULT_MAXIMUM))

# Send signal data to API
def send_to_api(signal_data):
    print("called")
    api_url = "http://localhost:8080/v1/signal/6773b57ae9eed75638f07f86/update-count"
    payload = {"signals": signal_data}
    print("Sending to API:", payload)

    try:
        response = requests.patch(api_url, json=payload)
        if response.status_code == 200:
            print("Successfully updated signals")
        else:
            print(f"Failed to update signals: {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"API request error: {e}")

# Control traffic flow
def control_traffic():
    global current_signal

    while True:
        signal_data = []

        print(f"\n==> Processing Signal {current_signal + 1}")

        # Start vehicle detection for the next signal in a separate thread
        next_signal_index = (current_signal + 1) % len(VIDEO_FEEDS)
        detection_thread = threading.Thread(target=calculate_green_time, args=(next_signal_index,))
        detection_thread.start()

        # Green light phase for the current signal
        green_time = calculate_green_time(current_signal)
        print(f"Signal {current_signal + 1}: Green for {green_time} seconds")
        time.sleep(green_time - 2)  # Ensure real-time API call happens before green phase ends

        # Reset the current signal to default times and prepare API data
        signal_data.append({
            "signal_id": f"6773b57ae9eed75638f07f8{3 + current_signal}",
            "red_duration": DEFAULT_RED,
            "yellow_duration": DEFAULT_YELLOW,
            "green_duration": DEFAULT_GREEN
        })

        # Prepare and send signal data for the next signal
        next_signal_green_time = calculate_green_time(next_signal_index)
        signal_data.append({
            "signal_id": f"6773b57ae9eed75638f07f8{3 + next_signal_index}",
            "red_duration": 0,
            "yellow_duration": DEFAULT_YELLOW,
            "green_duration": next_signal_green_time
        })

        send_to_api(signal_data)

        # Yellow light phase for the current signal
        print(f"Signal {current_signal + 1}: Yellow for {DEFAULT_YELLOW} seconds")
        time.sleep(DEFAULT_YELLOW)

        # Update current signal
        current_signal = (current_signal + 1) % len(signals)


# Start the program
if __name__ == "__main__":
    control_traffic()

# import math
# import time
# import cv2
# import numpy as np
# import requests
# import threading

# # Default traffic light settings
# DEFAULT_YELLOW = 5
# DEFAULT_GREEN = 15
# DEFAULT_MINIMUM = 10
# DEFAULT_MAXIMUM = 60
# DEFAULT_RED = 130

# # Vehicle time allocation per type
# VEHICLE_TIMES = {
#     "car": 2,
#     "bike": 1,
#     "bus": 3,
#     "truck": 4,
#     "rickshaw": 2,
# }

# # Image paths (replace with actual image file paths)
# IMAGE_PATHS = [
#     "/Volumes/sandisk_1/python/traffic-ai/test_images/1.jpg",
#     "/Volumes/sandisk_1/python/traffic-ai/test_images/2.jpg",
#     "/Volumes/sandisk_1/python/traffic-ai/test_images/3.jpg",
# ]

# # Load YOLO model
# net = cv2.dnn.readNet("/Volumes/sandisk_1/python/traffic-ai/bin/yolov3.weights", "/Volumes/sandisk_1/python/traffic-ai/cfg/yolov3.cfg")
# layer_names = net.getLayerNames()
# out_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]

# # Load COCO class labels
# with open("/Volumes/sandisk_1/python/traffic-ai/cfg/coco.names", "r") as f:
#     CLASSES = f.read().strip().split("\n")

# # Detect vehicles in an image
# def detect_vehicles(image_path):
#     frame = cv2.imread(image_path)
#     if frame is None:
#         print(f"Failed to load image: {image_path}")
#         return {key: 0 for key in VEHICLE_TIMES.keys()}

#     height, width, _ = frame.shape
#     blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
#     net.setInput(blob)
#     outputs = net.forward(out_layers)

#     vehicle_counts = {key: 0 for key in VEHICLE_TIMES.keys()}

#     for output in outputs:
#         for detection in output:
#             scores = detection[5:]
#             class_id = np.argmax(scores)
#             confidence = scores[class_id]
#             if confidence > 0.5:
#                 label = str(CLASSES[class_id])
#                 if label in VEHICLE_TIMES:
#                     vehicle_counts[label] += 1

#     print(f"Detected vehicles in {image_path}: {vehicle_counts}")
#     return vehicle_counts

# # Calculate green light duration based on vehicle counts
# def calculate_green_time(image_index):
#     vehicle_counts = detect_vehicles(IMAGE_PATHS[image_index])
#     green_time = sum(vehicle_counts[vehicle] * VEHICLE_TIMES[vehicle] for vehicle in vehicle_counts)
#     return max(DEFAULT_MINIMUM, min(math.ceil(green_time / 2), DEFAULT_MAXIMUM))

# # Send signal data to API
# def send_to_api(signal_data):
#     print("called")
#     api_url = "http://localhost:8080/v1/signal/6773b57ae9eed75638f07f86/update-count"
#     payload = {"signals": signal_data}
#     print("Sending to API:", payload)

#     try:
#         response = requests.patch(api_url, json=payload)
#         if response.status_code == 200:
#             print("Successfully updated signals")
#         else:
#             print(f"Failed to update signals: {response.status_code}")
#     except requests.exceptions.RequestException as e:
#         print(f"API request error: {e}")

# # Control traffic flow
# # Control traffic flow
# def control_traffic():
#     for i in range(len(IMAGE_PATHS)):
#         print(f"\n==> Processing Signal {i + 1}")

#         # Step 1: Detect vehicles and calculate green time
#         green_time = calculate_green_time(i)
#         print(f"Signal {i + 1}: Green for {green_time} seconds")

#         # Step 2: Prepare signal data and send to API immediately
#         signal_data = [
#             {
#                 "signal_id": f"6773b57ae9eed75638f07f8{3 + i}",
#                 "red_duration": DEFAULT_RED,
#                 "yellow_duration": DEFAULT_YELLOW,
#                 "green_duration": green_time,
#             }
#         ]
#         send_to_api(signal_data)

#         # Step 3: Use threading to handle green duration countdown
#         green_thread = threading.Thread(target=green_duration_countdown, args=(green_time,))
#         green_thread.start()

#         # Step 4: Process the next signal while the current one is still green
#         green_thread.join()

# # Green duration countdown
# def green_duration_countdown(green_time):
#     time.sleep(green_time - 2)  # Green phase
#     print(f"Signal: Yellow for {DEFAULT_YELLOW} seconds")
#     time.sleep(DEFAULT_YELLOW)  # Yellow phase


# # Start the program
# if __name__ == "__main__":
#     control_traffic()
