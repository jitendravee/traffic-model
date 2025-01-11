

import math
import time
import cv2
import numpy as np
import requests
import threading

# =========================
# Constants & Configuration
# =========================
DEFAULT_YELLOW = 5
DEFAULT_GREEN = 15
DEFAULT_MINIMUM = 10
DEFAULT_MAXIMUM = 60
DEFAULT_RED = 130

VEHICLE_TIMES = {
    "cars": 2,
    "bikes": 1,
    "buses": 3,
    "trucks": 4,
    "rickshaws": 2,
}

VIDEO_FEEDS = [
    "/Volumes/sandisk_1/python/traffic-ai/test_images/1.jpg",
    "/Volumes/sandisk_1/python/traffic-ai/test_images/2.jpg",
    "/Volumes/sandisk_1/python/traffic-ai/test_images/3.jpg",
]

API_URL = "http://localhost:8080/v1/signal/6773b57ae9eed75638f07f86/update-count"

# =========================
# YOLO Model Setup
# =========================
net = cv2.dnn.readNet(
    "/Volumes/sandisk_1/python/traffic-ai/bin/yolov3.weights",
    "/Volumes/sandisk_1/python/traffic-ai/cfg/yolov3.cfg"
)
layer_names = net.getLayerNames()
print(net.getUnconnectedOutLayers())

out_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]


with open("/Volumes/sandisk_1/python/traffic-ai/cfg/coco.names", "r") as f:
    CLASSES = f.read().strip().split("\n")

# =========================
# Traffic Signal Class
# =========================
class TrafficSignal:
    def __init__(self, red=DEFAULT_RED, yellow=DEFAULT_YELLOW, green=DEFAULT_GREEN):
        self.red = red
        self.yellow = yellow
        self.green = green

# =========================
# Vehicle Detection
# =========================
def detect_vehicles(image_path):
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"Error: Unable to load image from {image_path}")
        return {key: 0 for key in VEHICLE_TIMES.keys()}

    height, width, _ = frame.shape
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)

    # Forward pass
    outputs = net.forward(out_layers)

    vehicle_counts = {key: 0 for key in VEHICLE_TIMES.keys()}
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                label = str(CLASSES[class_id])
                if label in VEHICLE_TIMES.keys():
                    vehicle_counts[label + "s"] += 1

    return vehicle_counts


# =========================
# Green Time Calculation
# =========================
def calculate_green_time(image_index):
    vehicle_counts = detect_vehicles(VIDEO_FEEDS[image_index])
    print(f"Vehicle counts for Signal {image_index + 1}: {vehicle_counts}")
    green_time = sum(vehicle_counts[vehicle] * VEHICLE_TIMES[vehicle] for vehicle in vehicle_counts)
    return max(DEFAULT_MINIMUM, min(math.ceil(green_time / 2), DEFAULT_MAXIMUM))

# =========================
# API Communication
# =========================
def send_to_api(signal_data):
    try:
        response = requests.patch(API_URL, json={"signals": signal_data})
        if response.status_code == 200:
            print("Successfully updated signals")
        else:
            print(f"Failed to update signals: {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"API request error: {e}")

# =========================
# Traffic Control
# =========================
def control_traffic():
    current_signal = 0
    signals = [TrafficSignal() for _ in range(len(VIDEO_FEEDS))]

    while True:
        signal_data = []

        print(f"\n==> Processing Signal {current_signal + 1}")

        # Start vehicle detection for the next signal in a separate thread
        next_signal_index = (current_signal + 1) % len(VIDEO_FEEDS)
        detection_thread = threading.Thread(target=calculate_green_time, args=(next_signal_index,))
        detection_thread.start()

        # Green light phase for the current signal
        green_time = calculate_green_time(current_signal)

        # Prepare signal data for the current signal
        signal_data.append({
            "signal_id": f"6773b57ae9eed75638f07f8{3 + current_signal}",
            "red_duration": DEFAULT_RED,
            "yellow_duration": DEFAULT_YELLOW,
            "green_duration": green_time
        })

        # Wait for the next signal's green time to be calculated in the background thread
        detection_thread.join()

        # Prepare signal data for the next signal
        next_signal_green_time = calculate_green_time(next_signal_index)  # Now we safely calculate this in the main thread
        signal_data.append({
            "signal_id": f"6773b57ae9eed75638f07f8{3 + next_signal_index}",
            "red_duration": 0,
            "yellow_duration": DEFAULT_YELLOW,
            "green_duration": next_signal_green_time
        })

        # Send the API request after both green times are calculated
        send_to_api(signal_data)

        # Green light phase for the current signal
        print(f"Signal {current_signal + 1}: Green for {green_time} seconds")
        time.sleep(green_time)  # Full green time for the current signal
        
        # Yellow light phase for the current signal
        print(f"Signal {current_signal + 1}: Yellow for {DEFAULT_YELLOW} seconds")
        time.sleep(DEFAULT_YELLOW)

        # Update to the next signal
        current_signal = (current_signal + 1) % len(VIDEO_FEEDS)


# =========================
# Main Execution
# =========================
if __name__ == "__main__":
    control_traffic()
