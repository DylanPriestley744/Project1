import argparse
from pathlib import Path
from ultralytics import YOLO


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="Path to dataset.yaml")
    p.add_argument("--model", default="yolov8n.pt", help="yolov8n.pt / yolov8s.pt ...")
    p.add_argument("--imgsz", type=int, default=416, help="Input size")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--device", default="0", help="'0' for GPU0, or 'cpu'")
    p.add_argument("--name", default="veh_baseline", help="Run name")
    p.add_argument("--project", default="runs/detect", help="Output directory")
    return p.parse_args()


def main():
    args = get_args()
    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"dataset.yaml not found: {data_path}")

    model = YOLO(args.model)  # will download pretrained weights if needed

    model.train(
        data=str(data_path),
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        amp=True,       # mixed precision saves VRAM
        workers=2,      # keep small for laptops
        pretrained=True,
        verbose=True
    )

    print(f"Done. Results saved to: {args.project}/{args.name}")


if __name__ == "__main__":
    main()
