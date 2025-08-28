import torch

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Path to your video
video_path = "data/01.avi"

# Run detection
results = model(video_path)

# Save results
results.save(save_dir='runs/detect/')
print("Detection completed! Check runs/detect/ for output.")

