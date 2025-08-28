import torch
import cv2

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Video path
video_path = "data/01.avi"
cap = cv2.VideoCapture(video_path)

# Output video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('runs/detect/output_live.mp4', fourcc, 30.0,
                      (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run detection on frame
    results = model(frame)

    # Render results on frame
    img = results.render()[0]

    # Show frame
    cv2.imshow('YOLOv5 Detection', img)

    # Write frame to output video
    out.write(img)

    # Press 'q' to quit early
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
print("Video detection completed! Saved to runs/detect/output_live.mp4")

