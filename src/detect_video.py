import torch
import cv2

def run_detection(video_path, save_dir='runs/detect/'):
    # Load YOLOv5 model
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

    # Open video
    cap = cv2.VideoCapture(video_path)

    # Video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(f'{save_dir}output.mp4', fourcc, 30.0,
                          (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                           int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Run detection on frame
        results = model(frame)

        # Render results on frame
        img = results.render()[0]

        # Show frame (optional)
        cv2.imshow('YOLOv5 Detection', img)

        # Write frame to output
        out.write(img)

        # Press 'q' to quit early
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Detection completed! Video saved to {save_dir}output.mp4")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default='data/01.avi', help='path to video file')
    parser.add_argument('--save_dir', type=str, default='runs/detect/', help='directory to save output')
    args = parser.parse_args()

    run_detection(args.source, args.save_dir)

