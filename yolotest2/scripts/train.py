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

    # ✅ 改这里：不要写 runs/detect，让 Ultralytics 自己拼 detect
    p.add_argument("--project", default="runs", help="Output root directory (default: runs)")
    return p.parse_args()


def main():
    args = get_args()
    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"dataset.yaml not found: {data_path}")

    model = YOLO(args.model)

    model.train(
        data=str(data_path),
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=f"detect/{args.name}",
        exist_ok=True,  # ✅ 可选：同名覆盖更干净；不想覆盖就删掉这一行
        amp=True,
        workers=2,
        pretrained=True,
        verbose=True
    )

    # ✅ 更准确地打印实际输出目录
    save_dir = getattr(getattr(model, "trainer", None), "save_dir", None)
    if save_dir is not None:
        print(f"Done. Results saved to: {save_dir}")
    else:
        print(f"Done. Results saved to: {args.project}/detect/{args.name}")


if __name__ == "__main__":
    main()
