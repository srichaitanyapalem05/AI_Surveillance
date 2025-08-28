import torch
import argparse
from pathlib import Path
import os
import glob

def run_detection(source="data/01.avi", save_dir="runs/detect/exp"):
    # Load YOLOv5 model
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

    # Run detection
    results = model(source)

    # Save results
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    results.save(save_dir=save_dir)

    print(f"✅ Detection finished. Output saved in {save_dir}")

    # Find the output video (usually same filename as input, inside save_dir)
    video_files = glob.glob(f"{save_dir}/*.avi") + glob.glob(f"{save_dir}/*.mp4")

    if video_files:
        print(f"🎬 Opening {video_files[0]} ...")
        os.system(f"open {video_files[0]}")  # macOS: opens with default video player
    else:
        print("⚠️ No video file found in output folder.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, default="data/01.avi", help="video or image source")
    parser.add_argument("--save_dir", type=str, default="runs/detect/exp", help="where to save results")
    args = parser.parse_args()

    run_detection(source=args.source, save_dir=args.save_dir)


