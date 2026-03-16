import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
from collections import defaultdict, deque

# --- CONFIG ---
VIDEO_PATH = 'video.mp4'
OUTPUT_PATH = 'output.mp4'
YOLO_MODEL = 'yolov8n.pt'
VEHICLE_CLASSES = ['car', 'truck', 'bus', 'motorcycle']
SIZE_THRESHOLDS = {'small': 4000, 'medium': 12000}  # bounding box area
TRAJECTORY_LENGTH = 30  # frames
GRAPH_WIDTH = 400

# --- UTILS ---
def get_vehicle_size(area):
    if area < SIZE_THRESHOLDS['small']:
        return 'small'
    elif area < SIZE_THRESHOLDS['medium']:
        return 'medium'
    else:
        return 'large'

# --- MAIN ---
def main():
    # Load YOLO model
    model = YOLO(YOLO_MODEL)

    # Open video
    cap = cv2.VideoCapture(VIDEO_PATH)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width + GRAPH_WIDTH, height))

    # Tracking state
    trajectories = defaultdict(lambda: deque(maxlen=TRAJECTORY_LENGTH))
    vehicle_sizes = defaultdict(str)
    vehicle_counts = []
    size_distributions = []
    total_vehicle_ids = set()

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # YOLO detection and tracking
        results = model.track(frame, persist=True, tracker='bytetrack.yaml')
        boxes = results[0].boxes
        ids = results[0].boxes.id if boxes.id is not None else []
        cls = results[0].boxes.cls if boxes.cls is not None else []
        xyxy = results[0].boxes.xyxy.cpu().numpy() if boxes.xyxy is not None else np.empty((0, 4))

        # Per-frame analytics
        frame_vehicle_count = 0
        size_count = {'small': 0, 'medium': 0, 'large': 0}

        for i, box in enumerate(xyxy):
            class_idx = int(cls[i]) if i < len(cls) else None
            class_name = model.names[class_idx] if class_idx is not None else None
            if class_name not in VEHICLE_CLASSES:
                continue
            vid = int(ids[i]) if i < len(ids) else None
            if vid is None:
                continue
            x1, y1, x2, y2 = map(int, box)
            area = (x2 - x1) * (y2 - y1)
            size = get_vehicle_size(area)
            vehicle_sizes[vid] = size
            size_count[size] += 1
            frame_vehicle_count += 1
            total_vehicle_ids.add(vid)

            # Draw bounding box
            color = (0, 255, 0) if size == 'small' else (255, 255, 0) if size == 'medium' else (255, 0, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f'{class_name} ID:{vid} {size}'
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # Update trajectory
            center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
            trajectories[vid].append(center)

        # Draw trajectories
        for vid, points in trajectories.items():
            if len(points) > 1:
                for j in range(1, len(points)):
                    color = (0, 255, 0) if vehicle_sizes[vid] == 'small' else (255, 255, 0) if vehicle_sizes[vid] == 'medium' else (255, 0, 0)
                    cv2.line(frame, points[j - 1], points[j], color, 2)

        vehicle_counts.append(frame_vehicle_count)
        size_distributions.append(size_count.copy())

        # --- GRAPH DASHBOARD ---

        # --- Improved Dashboard ---
        dashboard = np.ones((height, GRAPH_WIDTH, 3), dtype=np.uint8) * 240  # light gray
        pad = 20
        graph_h = (height - 3 * pad) // 2

        # Traffic density line graph
        plt.figure(figsize=(4, 3))
        plt.plot(vehicle_counts, color='blue', linewidth=2)
        plt.title('Traffic Density', fontsize=18, fontweight='bold')
        plt.xlabel('Frame', fontsize=14)
        plt.ylabel('Vehicles', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.tight_layout(pad=1.0)
        plt.savefig('temp_density.png', bbox_inches='tight')
        plt.close()
        density_img = cv2.imread('temp_density.png')
        density_img = cv2.resize(density_img, (GRAPH_WIDTH - 2 * pad, graph_h))
        dashboard[pad:pad + graph_h, pad:GRAPH_WIDTH - pad, :] = density_img

        # Vehicle size distribution bar chart
        sizes = ['small', 'medium', 'large']
        counts = [size_count[s] for s in sizes]
        plt.figure(figsize=(4, 2))
        plt.bar(sizes, counts, color=['green', 'yellow', 'red'])
        plt.title('Vehicle Size Distribution', fontsize=18, fontweight='bold')
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.tight_layout(pad=1.0)
        plt.savefig('temp_size.png', bbox_inches='tight')
        plt.close()
        size_img = cv2.imread('temp_size.png')
        size_img = cv2.resize(size_img, (GRAPH_WIDTH - 2 * pad, graph_h))
        dashboard[pad * 2 + graph_h:pad * 2 + 2 * graph_h, pad:GRAPH_WIDTH - pad, :] = size_img

        # Add header
        cv2.putText(dashboard, 'Analytics', (pad, pad - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

        # Add black border
        cv2.rectangle(dashboard, (0, 0), (GRAPH_WIDTH - 1, height - 1), (0, 0, 0), 3)

        # Combine video and dashboard
        combined = np.hstack((frame, dashboard))
        out.write(combined)

        frame_idx += 1
        print(f'Processed frame {frame_idx}', end='\r')

    cap.release()
    out.release()
    print(f'\nProcessing complete. Total vehicles: {len(total_vehicle_ids)}')

if __name__ == '__main__':
    main()
