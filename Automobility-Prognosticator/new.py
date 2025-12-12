import cv2
import numpy as np
import torch
import supervision as sv
from pycparser.c_ast import Default
from ultralytics import YOLO
from collections import defaultdict
import csv
from datetime import datetime, timedelta
import argparse
import sys
import signal
import os
import json


class VehicleCounter:
    def __init__(self):
        self.cumulative_counts = defaultdict(lambda: defaultdict(int))
        self.vehicle_classes = ["Bicycle", "Bus", "Car", "LCV", "Three Wheeler", "Truck", "Two Wheeler"]

    def update_counts(self, start_point, end_point, vehicle_class):
        """Update vehicle counts"""
        movement = f"{start_point}{end_point}"
        self.cumulative_counts[movement][vehicle_class] += 1

    def get_json_output(self):
        """Get current counts in JSON format"""
        formatted_data = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Cumulative Counts": {}
        }

        active_regions = set()
        for movement, counts in self.cumulative_counts.items():
            active_regions.update(list(movement))
            formatted_data["Cumulative Counts"][movement] = {
                vehicle_class: counts.get(vehicle_class, 0)
                for vehicle_class in self.vehicle_classes
            }

        return formatted_data


class VehicleTracker:
    def __init__(self, video_path, model_path,output_path):
        self.video_path = video_path
        try:
            self.model = YOLO(model_path).to('cuda')
        except:
            self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()
        self.vehicle_paths = defaultdict(list)
        self.counter = VehicleCounter()
        self.save_interval = 30 * 60  # Save every 30 seconds (assuming 60 fps)
        self.last_save_time = None
        self.output_path = output_path

        # Region selection attributes
        self.regions = []
        self.drawing = False
        self.current_region = None
        self.current_frame = None
        self.region_labels = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')  # Extended labels
        self.used_labels = set()

        # Vehicle class mapping
        self.vehicle_class_map = {
            1: 'Bicycle',
            2: 'Car',
            3: 'Two Wheeler',
            5: 'Bus',
            7: 'Truck'
        }

    @staticmethod
    def _point_in_region(point, region):
        """Check if a point is within a region"""
        x, y = point
        x1, y1, x2, y2 = region['coords']
        return x1 <= x <= x2 and y1 <= y <= y2

    def select_regions(self):
        """Interactive region selection interface"""
        cap = cv2.VideoCapture(self.video_path)
        ret, frame = cap.read()
        if not ret:
            cap.release()
            raise Exception("Could not read video file")

        cv2.namedWindow("Select Regions")
        cv2.setMouseCallback("Select Regions", self._mouse_callback)

        self.current_frame = frame.copy()
        self.regions = []
        self.used_labels = set()

        print("\nRegion Selection Instructions:")
        print("1. Left click and drag to draw a region")
        print("2. Release to complete the region")
        print("3. Press 'q' when done selecting regions")
        print("4. Press 'r' to reset all regions\n")

        while True:
            display_frame = self.current_frame.copy()

            # Draw completed regions
            for region in self.regions:
                x1, y1, x2, y2 = region['coords']
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(display_frame, region['label'], (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

            # Draw current region
            if self.drawing and self.current_region:
                x1, y1 = self.current_region['start']
                x2, y2 = self.current_region['end']
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            cv2.imshow("Select Regions", display_frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break
            elif key == ord('r'):
                self.regions = []
                self.used_labels = set()
                self.current_frame = frame.copy()

        cap.release()
        cv2.destroyAllWindows()
        return self.regions

    def _mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events for region selection"""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.current_region = {
                'start': (x, y),
                'end': (x, y)
            }

        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            if self.current_region:
                self.current_region['end'] = (x, y)

        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            if self.current_region:
                available_labels = [l for l in self.region_labels if l not in self.used_labels]
                if available_labels:
                    label = available_labels[0]
                    self.used_labels.add(label)

                    x1, y1 = self.current_region['start']
                    x2, y2 = self.current_region['end']
                    x1, x2 = min(x1, x2), max(x1, x2)
                    y1, y2 = min(y1, y2), max(y1, y2)

                    self.regions.append({
                        'label': label,
                        'coords': (x1, y1, x2, y2)
                    })

                self.current_region = None

    def process_video(self, regions):
        """Process video and track vehicles"""
        cap = cv2.VideoCapture(self.video_path)
        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            current_time = datetime.now()

            # Object detection
            results = self.model(frame)[0]
            detections = sv.Detections(
                xyxy=results.boxes.xyxy.cpu().numpy(),
                confidence=results.boxes.conf.cpu().numpy(),
                class_id=results.boxes.cls.cpu().numpy().astype(int)
            )

            # Filter out person detections
            valid_indices = [i for i, class_id in enumerate(detections.class_id) if class_id != 0]
            detections = detections[valid_indices]

            # Tracking
            tracker_results = self.tracker.update_with_detections(detections)

            # Process detections if any exist
            if len(tracker_results.xyxy) > 0:
                self._process_detections(tracker_results, regions, current_time)

            # Save results periodically
            if (self.last_save_time is None or
                    (current_time - self.last_save_time).seconds >= self.save_interval):
                self._save_results(current_time)
                self.last_save_time = current_time

            # Visualization
            self._draw_visualization(frame, tracker_results, regions)

            cv2.imshow("Tracking", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        self._save_results()

    def _process_detections(self, tracker_results, regions, current_time):
        """Process detection results and update vehicle paths"""
        for i in range(len(tracker_results.xyxy)):
            bbox = tracker_results.xyxy[i]
            track_id = tracker_results.tracker_id[i]
            class_id = tracker_results.class_id[i]

            center_x = (bbox[0] + bbox[2]) / 2
            center_y = (bbox[1] + bbox[3]) / 2

            for region in regions:
                if self._point_in_region((center_x, center_y), region):
                    vehicle_class = self.vehicle_class_map.get(class_id, 'Unknown')
                    self.vehicle_paths[track_id].append({
                        'region': region['label'],
                        'class': vehicle_class,
                        'timestamp': current_time
                    })

                    if len(self.vehicle_paths[track_id]) >= 2:
                        start_point = self.vehicle_paths[track_id][0]['region']
                        end_point = self.vehicle_paths[track_id][-1]['region']
                        if start_point != end_point:
                            self.counter.update_counts(start_point, end_point, vehicle_class)

    def _draw_visualization(self, frame, tracker_results, regions):
        """Draw visualization overlays on the frame"""
        # Draw tracked objects
        for i in range(len(tracker_results.xyxy)):
            bbox = tracker_results.xyxy[i]
            track_id = tracker_results.tracker_id[i]
            class_id = tracker_results.class_id[i]

            cv2.rectangle(frame,
                          (int(bbox[0]), int(bbox[1])),
                          (int(bbox[2]), int(bbox[3])),
                          (0, 255, 0), 2)

            vehicle_class = self.vehicle_class_map.get(class_id, 'Unknown')
            cv2.putText(frame, f"ID: {track_id} ({vehicle_class})",
                        (int(bbox[0]), int(bbox[1]) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Draw regions
        for region in regions:
            x1, y1, x2, y2 = region['coords']
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, region['label'],
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    def _save_results(self, current_time=None):
        """Save tracking results with varied timestamps"""
        if current_time is None:
            current_time = datetime.now()

        os.makedirs(self.output_path, exist_ok=True)

        # Identify completed vehicle paths
        completed_paths = {
            track_id: paths for track_id, paths in self.vehicle_paths.items()
            if len(paths) >= 2 and paths[0]['region'] != paths[-1]['region']
        }

        # Calculate class-wise counts
        class_counts = defaultdict(int)
        for paths in completed_paths.values():
            vehicle_class = paths[0]['class']
            class_counts[vehicle_class] += 1

        # Save analysis to JSON
        with open(self.output_path + '/traffic_analysis.json', 'w') as f:
            json.dump({
                "timestamp": current_time.strftime("%Y-%m-%d %H:%M:%S"),
                "vehicle_paths": {
                    str(track_id): [
                        {
                            "region": path["region"],
                            "class": path["class"],
                            "timestamp": path["timestamp"].strftime("%Y-%m-%d %H:%M:%S.%f")
                        }
                        for path in paths
                    ]
                    for track_id, paths in completed_paths.items()
                },
                "counts": self.counter.get_json_output()
            }, f, indent=4)

        # Write to traffic analysis CSV
        with open(self.output_path +'/traffic_analysis.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['timestamp', 'track_id', 'vehicle_class', 'start_region', 'end_region', 'duration'])

            for track_id, paths in completed_paths.items():
                start_path = paths[0]
                end_path = paths[-1]
                duration = (end_path['timestamp'] - start_path['timestamp']).total_seconds()
                vehicle_class = start_path['class']

                writer.writerow([
                    end_path['timestamp'].strftime("%Y-%m-%d %H:%M:%S.%f"),
                    track_id,
                    vehicle_class,
                    start_path['region'],
                    end_path['region'],
                    duration
                ])

        # Write vehicle class counts
        with open(self.output_path + '/vehicle_class_counts.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['timestamp', 'vehicle_class', 'total_count'])

            for vehicle_class, count in class_counts.items():
                writer.writerow([
                    current_time.strftime("%Y-%m-%d %H:%M:%S.%f"),
                    vehicle_class,
                    count
                ])


# def main():
#     """Main function to run the vehicle tracking system"""
#     parser = argparse.ArgumentParser(description='Vehicle Tracking System')
#     parser.add_argument('--video', type=str, required=True, help='Path to input video file')
#     parser.add_argument('--model', type=str, required=True, help='Path to YOLO model weights')
#     args = parser.parse_args()
def main(video, model, output):
    # Initialize tracker
    tracker = VehicleTracker(video, model, output)

    # Select regions
    print("Please select regions for tracking...")
    regions = tracker.select_regions()

    if not regions:
        print("No regions selected. Exiting...")
        return

    # Handle graceful shutdown
    def signal_handler(sig, frame):
        print("\nSaving final results and shutting down...")
        tracker._save_results()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    # Process video
    print("Processing video...")
    tracker.process_video(regions)

    print("Processing complete. Results saved in 'output' directory.")


if __name__ == "__main__":
    """Main function to run the vehicle tracking system"""
    parser = argparse.ArgumentParser(description='Vehicle Tracking System')
    parser.add_argument('--video', type=str, required=True, help='Path to input video file')
    parser.add_argument('--model', type=str, required=True, help='Path to YOLO model weights')
    parser.add_argument('--output', type=str, default='output', help='Path to output folder')

    args = parser.parse_args()

    print(f"Starting Vehicle Tracking with Video: {args.video}")
    print(f"Using Model: {args.model}")

    main(args.video, args.model, args.output)