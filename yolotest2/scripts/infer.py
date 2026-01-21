import argparse
import csv
import time
from pathlib import Path

import cv2
from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, help="path to best.pt")
    ap.add_argument("--source", required=True, help="path to video, e.g., assets/videos/TrafficPolice.mp4")
    ap.add_argument("--imgsz", type=int, default=416)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.7)
    ap.add_argument("--device", default="", help="e.g. 0 or cpu; empty=auto")
    ap.add_argument("--name", default="veh_demo_video", help="output folder name under runs/detect")
    ap.add_argument("--save_dir", default="runs/detect", help="base output dir")
    ap.add_argument("--save_csv", action="store_true", help="save per-frame counts to CSV")
    args = ap.parse_args()

    weights = Path(args.weights)
    source = Path(args.source)

    save_root = Path(args.save_dir) / args.name
    save_root.mkdir(parents=True, exist_ok=True)

    out_video = save_root / "veh_demo_boxed.mp4"
    out_csv = save_root / "veh_demo_counts.csv"

    model = YOLO(str(weights))

    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {source}")

    fps_in = cap.get(cv2.CAP_PROP_FPS)
    if not fps_in or fps_in <= 0:
        fps_in = 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_video), fourcc, fps_in, (w, h))

    csv_file = None
    csv_writer = None
    if args.save_csv:
        csv_file = open(out_csv, "w", newline="", encoding="utf-8")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["frame", "time_sec", "total_dets", "Ambulance", "Bus", "Car", "Motorcycle", "Truck"])

    frame_idx = 0
    t_start = time.perf_counter()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(
            source=frame,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            device=args.device,
            verbose=False
        )
        r = results[0]
        annotated = r.plot()  # annotated BGR frame
        writer.write(annotated)

        if csv_writer is not None:
            counts = [0, 0, 0, 0, 0]  # A,B,C,M,T
            total = 0
            if r.boxes is not None and r.boxes.cls is not None:
                cls = r.boxes.cls.cpu().numpy().astype(int)
                total = int(cls.shape[0])
                for c in cls:
                    if 0 <= c < 5:
                        counts[c] += 1
            csv_writer.writerow([frame_idx, frame_idx / fps_in, total] + counts)

        frame_idx += 1

    t_end = time.perf_counter()
    elapsed = t_end - t_start
    fps_proc = frame_idx / elapsed if elapsed > 0 else 0.0

    cap.release()
    writer.release()
    if csv_file:
        csv_file.close()

    print("=== Video Demo Done ===")
    print(f"Input video: {source}")
    print(f"Weights: {weights}")
    print(f"Frames processed: {frame_idx}")
    print(f"Elapsed: {elapsed:.2f}s")
    print(f"Processing FPS (model+write): {fps_proc:.2f}")
    print(f"Saved video: {out_video}")
    if args.save_csv:
        print(f"Saved counts CSV: {out_csv}")


if __name__ == "__main__":
    main()
