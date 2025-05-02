'''
Person Tracking and Heatmap with YOLOv8

Requirements:
    pip install ultralytics opencv-python numpy

Usage:
    python yolov8_person_heatmap.py --source video.mp4 --model yolov8n.pt --target_id 0
'''

import argparse
import cv2
import numpy as np
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description='YOLOv8 Person Tracking and Heatmap Generator')
    parser.add_argument('--source', type=str, default=0,
                        help='Path to video file or camera index (default: 0)')
    parser.add_argument('--model', type=str, default='yolov8n.pt',
                        help='Path to YOLOv8 model (default: yolov8n.pt)')
    parser.add_argument('--tracker', type=str, default='bytetrack.yaml',
                        help='Tracker config file for tracking (default: bytetrack.yaml)')
    parser.add_argument('--target_id', type=int, default=0,
                        help='Person track ID to generate heatmap for (default: 0)')
    parser.add_argument('--display', action='store_true',
                        help='Display tracking frames in real time')
    args = parser.parse_args()

    # Pre-open video to get dimensions
    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        print(f"Error: Unable to open source {args.source}")
        return
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    # Load YOLOv8 model
    model = YOLO(args.model)

    # Dictionary to store per-ID trajectories
    trajectories = {}

    # Run tracking
    for result in model.track(source=args.source,
                                tracker=args.tracker,
                                classes=[0],  # only detect persons (class 0)
                                stream=True,
                                persist=True):
        frame = result.orig_img  # BGR image

        # Iterate detections
        for box in result.boxes:
            track_id = int(box.id)
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            # Append center to trajectory
            trajectories.setdefault(track_id, []).append((cx, cy))

            # Draw bounding box and ID
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, str(track_id), (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)

        # Display tracking if requested
        if args.display:
            cv2.imshow('YOLOv8 Person Tracking', frame)
            if cv2.waitKey(1) == 27:  # ESC to quit
                break

    # After tracking loop, generate heatmap for the specified ID
    heatmap = np.zeros((height, width), dtype=np.float32)
    points = trajectories.get(args.target_id, [])
    if not points:
        print(f"No trajectory found for ID {args.target_id}")
        return

    for x, y in points:
        # accumulate visits
        if 0 <= x < width and 0 <= y < height:
            heatmap[y, x] += 1

    # Normalize heatmap to 0-255
    heatmap = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)
    heatmap = heatmap.astype(np.uint8)

    # Apply a color map (JET)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # Overlay heatmap on a blank image or transparent background
    overlay = cv2.addWeighted(heatmap_color, 0.6,
                              np.zeros_like(heatmap_color), 0.4, 0)

    # Display the final heatmap
    cv2.imshow(f'Heatmap for ID {args.target_id}', overlay)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
