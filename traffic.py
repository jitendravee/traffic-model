

# import math
# import time
# import cv2
# import numpy as np
# import requests
# import threading

# # =========================
# # Constants & Configuration
# # =========================
# DEFAULT_YELLOW = 5
# DEFAULT_GREEN = 15
# DEFAULT_MINIMUM = 10
# DEFAULT_MAXIMUM = 60
# DEFAULT_RED = 130

# VEHICLE_TIMES = {
#     "cars": 2,
#     "bikes": 1,
#     "buses": 3,
#     "trucks": 4,
#     "rickshaws": 2,
# }

# VIDEO_FEEDS = [
#     "/Volumes/sandisk_1/python/traffic-ai/test_images/1.jpg",
#     "/Volumes/sandisk_1/python/traffic-ai/test_images/2.jpg",
#     "/Volumes/sandisk_1/python/traffic-ai/test_images/3.jpg",
# ]

# API_URL = "http://localhost:8080/v1/signal/6773b57ae9eed75638f07f86/update-count"

# # =========================
# # YOLO Model Setup
# # =========================
# net = cv2.dnn.readNet(
#     "/Volumes/sandisk_1/python/traffic-ai/bin/yolov3.weights",
#     "/Volumes/sandisk_1/python/traffic-ai/cfg/yolov3.cfg"
# )
# layer_names = net.getLayerNames()
# print(net.getUnconnectedOutLayers())

# out_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]


# with open("/Volumes/sandisk_1/python/traffic-ai/cfg/coco.names", "r") as f:
#     CLASSES = f.read().strip().split("\n")

# # =========================
# # Traffic Signal Class
# # =========================
# class TrafficSignal:
#     def __init__(self, red=DEFAULT_RED, yellow=DEFAULT_YELLOW, green=DEFAULT_GREEN):
#         self.red = red
#         self.yellow = yellow
#         self.green = green

# # =========================
# # Vehicle Detection
# # =========================
# def detect_vehicles(image_path):
#     frame = cv2.imread(image_path)
#     if frame is None:
#         print(f"Error: Unable to load image from {image_path}")
#         return {key: 0 for key in VEHICLE_TIMES.keys()}

#     height, width, _ = frame.shape
#     blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
#     net.setInput(blob)

#     # Forward pass
#     outputs = net.forward(out_layers)

#     vehicle_counts = {key: 0 for key in VEHICLE_TIMES.keys()}
#     for output in outputs:
#         for detection in output:
#             scores = detection[5:]
#             class_id = np.argmax(scores)
#             confidence = scores[class_id]
#             if confidence > 0.5:
#                 label = str(CLASSES[class_id])
#                 if label in VEHICLE_TIMES.keys():
#                     vehicle_counts[label + "s"] += 1

#     return vehicle_counts


# # =========================
# # Green Time Calculation
# # =========================
# def calculate_green_time(image_index):
#     vehicle_counts = detect_vehicles(VIDEO_FEEDS[image_index])
#     print(f"Vehicle counts for Signal {image_index + 1}: {vehicle_counts}")
#     green_time = sum(vehicle_counts[vehicle] * VEHICLE_TIMES[vehicle] for vehicle in vehicle_counts)
#     return max(DEFAULT_MINIMUM, min(math.ceil(green_time / 2), DEFAULT_MAXIMUM))

# # =========================
# # API Communication
# # =========================
# def send_to_api(signal_data):
#     try:
#         response = requests.patch(API_URL, json={"signals": signal_data})
#         if response.status_code == 200:
#             print("Successfully updated signals")
#         else:
#             print(f"Failed to update signals: {response.status_code}")
#     except requests.exceptions.RequestException as e:
#         print(f"API request error: {e}")

# # =========================
# # Traffic Control
# # =========================
# def control_traffic():
#     current_signal = 0
#     signals = [TrafficSignal() for _ in range(len(VIDEO_FEEDS))]

#     while True:
#         signal_data = []

#         print(f"\n==> Processing Signal {current_signal + 1}")

#         # Start vehicle detection for the next signal in a separate thread
#         next_signal_index = (current_signal + 1) % len(VIDEO_FEEDS)
#         detection_thread = threading.Thread(target=calculate_green_time, args=(next_signal_index,))
#         detection_thread.start()

#         # Green light phase for the current signal
#         green_time = calculate_green_time(current_signal)

#         # Prepare signal data for the current signal
#         signal_data.append({
#             "signal_id": f"6773b57ae9eed75638f07f8{3 + current_signal}",
#             "red_duration": DEFAULT_RED,
#             "yellow_duration": DEFAULT_YELLOW,
#             "green_duration": green_time
#         })

#         # Wait for the next signal's green time to be calculated in the background thread
#         detection_thread.join()

#         # Prepare signal data for the next signal
#         next_signal_green_time = calculate_green_time(next_signal_index)  # Now we safely calculate this in the main thread
#         signal_data.append({
#             "signal_id": f"6773b57ae9eed75638f07f8{3 + next_signal_index}",
#             "red_duration": 0,
#             "yellow_duration": DEFAULT_YELLOW,
#             "green_duration": next_signal_green_time
#         })

#         # Send the API request after both green times are calculated
#         send_to_api(signal_data)

#         # Green light phase for the current signal
#         print(f"Signal {current_signal + 1}: Green for {green_time} seconds")
#         time.sleep(green_time)  # Full green time for the current signal
        
#         # Yellow light phase for the current signal
#         print(f"Signal {current_signal + 1}: Yellow for {DEFAULT_YELLOW} seconds")
#         time.sleep(DEFAULT_YELLOW)

#         # Update to the next signal
#         current_signal = (current_signal + 1) % len(VIDEO_FEEDS)


# # =========================
# # Main Execution
# # =========================
# if __name__ == "__main__":
#     control_traffic()
import math
import time
import cv2
import numpy as np
import requests
import threading
import torch  # Import PyTorch for YOLOv5


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
    "/content/chhatrapati-shivaji-terminus-train-station-mumbai.webp",
    "/content/high-angle-view-traffic-street_1048944-6077575.webp",
    "/content/trams-buses-traffic-kolkata-india.webp",
]

API_URL = "http://localhost:8080/v1/signal/6773b57ae9eed75638f07f86/update-count"

model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

CLASSES = model.names 


class TrafficSignal:
    def __init__(self, red=DEFAULT_RED, yellow=DEFAULT_YELLOW, green=DEFAULT_GREEN):
        self.red = red
        self.yellow = yellow
        self.green = green

def detect_vehicles(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Unable to load image from {image_path}")
        return {key: 0 for key in VEHICLE_TIMES.keys()}

    # Resize the image to 640x640 (standard input size for YOLOv5)
    img = cv2.resize(img, (640, 640))

    # Perform inference using YOLOv5
    results = model(img)

    # Visualize the results (optional, for debugging)
    results.show()  # Display the image with bounding boxes

    vehicle_counts = {key: 0 for key in VEHICLE_TIMES.keys()}
    for *xyxy, conf, cls in results.xyxy[0]:  # Results in xyxy format (x1, y1, x2, y2)
        label = CLASSES[int(cls)]  # Get the class name from the index
        # Adjust for YOLO class names (singular) to your VEHICLE_TIMES keys (plural)
        if label == "car":
            vehicle_counts["cars"] += 1
        elif label == "bus":
            vehicle_counts["buses"] += 1
        elif label == "truck":
            vehicle_counts["trucks"] += 1
        elif label == "bike":
            vehicle_counts["bikes"] += 1
        # Add more vehicle types if necessary

    return vehicle_counts





def calculate_green_time(image_index):
    vehicle_counts = detect_vehicles(VIDEO_FEEDS[image_index])
    print(f"Vehicle counts for Signal {image_index + 1}: {vehicle_counts}")
    green_time = sum(vehicle_counts[vehicle] * VEHICLE_TIMES[vehicle] for vehicle in vehicle_counts)
    return max(DEFAULT_MINIMUM, min(math.ceil(green_time / 2), DEFAULT_MAXIMUM))


def send_to_api(signal_data):
    print(signal_data)
    try:
        response = requests.patch(API_URL, json={"signals": signal_data})
        if response.status_code == 200:
            print("Successfully updated signals")
        else:
            print(f"Failed to update signals: {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"API request error: {e}")


def control_traffic():
    signals_count = len(VIDEO_FEEDS)  
    current_signal = 0  
    while True:
        signal_data = []

        print(f"\n==> Processing Signal {current_signal + 1}")

        green_time_current_signal = calculate_green_time(current_signal)
        if green_time_current_signal == DEFAULT_MINIMUM:
            green_time_current_signal = DEFAULT_GREEN  # Use default green time if vehicle counts are too low

        print(f"Signal {current_signal + 1} Green time: {green_time_current_signal} seconds")

        # Step 2: Prepare the signal data for API call for the current signal
        # For the current signal, use the calculated green time, red and yellow default values
        signal_data.append({
            "signal_id": f"6773b57ae9eed75638f07f8{3 + current_signal}",
            "red_duration": 0,
            "yellow_duration": DEFAULT_YELLOW,
            "green_duration": green_time_current_signal
        })

        # Step 3: Prepare default values for all other signals
        for i in range(signals_count):
            if i != current_signal:  # Skip the current signal
                # For the next signal, its red time is calculated as current signal's green + yellow time
                if i == (current_signal + 1) % signals_count:  # Next signal
                    red_time_next_signal = green_time_current_signal + DEFAULT_YELLOW
                else:
                    red_time_next_signal = DEFAULT_RED
                
                signal_data.append({
                    "signal_id": f"6773b57ae9eed75638f07f8{3 + i}",
                    "red_duration": red_time_next_signal,
                    "yellow_duration": DEFAULT_YELLOW,
                    "green_duration": DEFAULT_GREEN  # All other signals are red during this phase
                })

        # Send API request for the current signal's state (current signal green, others red)
        print(f"Sending API request with data for Signal {current_signal + 1} (Green), and others (Red)")
        send_to_api(signal_data)

        # Step 4: Wait for the current signal's green time and yellow phase
        print(f"Signal {current_signal + 1}: Green for {green_time_current_signal} seconds")
        time.sleep(green_time_current_signal)  # Green time for current signal
        print(f"Signal {current_signal + 1}: Yellow for {DEFAULT_YELLOW} seconds")
        time.sleep(DEFAULT_YELLOW)  # Yellow time for current signal

        # Step 5: Move to the next signal (loop around)
        current_signal = (current_signal + 1) % signals_count

if __name__ == "__main__":
    control_traffic()
