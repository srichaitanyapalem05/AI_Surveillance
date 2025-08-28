import os
import torch
import cv2
from datetime import datetime
from deep_sort_realtime.deepsort_tracker import DeepSort

# macOS notification function
def notify_mac(title, message):
    os.system(f'''osascript -e 'display notification "{message}" with title "{title}"' ''')

# Video path and output directory
video_path = "data/01.mp4"
save_dir = "runs/detect/"
os.makedirs(save_dir, exist_ok=True)

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Initialize DeepSort
tracker = DeepSort(max_age=30)

# Open video
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print(f"ERROR: Cannot open video {video_path}")
    exit()

# Output video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(os.path.join(save_dir, 'output_surveillance.mp4'), fourcc, 30.0,
                      (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                       int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

# Cumulative count dictionary
cumulative_count = {}

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # YOLOv5 detection
    results = model(frame)
    detections = []

    # Convert YOLOv5 results to DeepSort format
    for *xyxy, conf, cls in results.xyxy[0]:
        class_name = model.names[int(cls)]
        if class_name == "person":
            x1, y1, x2, y2 = [int(x.item()) for x in xyxy]
            # Pass as [bbox, confidence, class_name]
            detections.append([[x1, y1, x2, y2], float(conf.item()), class_name])

    # Update tracker with proper detection format
    tracks = tracker.update_tracks(detections, frame=frame)

    # Count people in this frame
    person_count = 0
    for track in tracks:
        if not track.is_confirmed():
            continue
        if track.get_det_class() == "person":
            person_count += 1
            # Update cumulative count
            cumulative_count[track.track_id] = 1
            # Draw bbox
            x1, y1, x2, y2 = map(int, track.to_ltrb())
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, f"ID: {track.track_id}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Display cumulative count on frame
    cv2.putText(frame, f"Persons in frame: {person_count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(frame, f"Cumulative persons: {len(cumulative_count)}", (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Send macOS notification if at least one person detected
    if person_count > 0:
        notify_mac("AI Surveillance Alert",
                   f"{person_count} person(s) detected at {datetime.now().strftime('%H:%M:%S')}")

    # Show frame live
    cv2.imshow('AI Surveillance', frame)

    # Write frame to output video
    out.write(frame)

    # Quit if 'q' pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()
print(f"Surveillance completed! Video saved to {save_dir}")

