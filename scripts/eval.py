import argparse
from pathlib import Path
from ultralytics import YOLO


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="Path to dataset.yaml")
    p.add_argument("--weights", required=True, help="Path to best.pt")
    p.add_argument("--imgsz", type=int, default=416)
    p.add_argument("--device", default="0")
    p.add_argument("--split", default="val", choices=["val", "test"])
    return p.parse_args()


def main():
    args = get_args()
    data_path = Path(args.data)
    weights_path = Path(args.weights)

    if not data_path.exists():
        raise FileNotFoundError(f"dataset.yaml not found: {data_path}")
    if not weights_path.exists():
        raise FileNotFoundError(f"weights not found: {weights_path}")

    model = YOLO(str(weights_path))
    metrics = model.val(
        data=str(data_path),
        imgsz=args.imgsz,
        device=args.device,
        split=args.split
    )

    print(f"Done. Split={args.split}")
    print(metrics)


if __name__ == "__main__":
    main()
