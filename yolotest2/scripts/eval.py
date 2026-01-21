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
    p.add_argument("--plots", action="store_true", help="Save PR/CM plots")
    return p.parse_args()


def main():
    args = get_args()
    data_path = Path(args.data)
    weights_path = Path(args.weights)

    if not data_path.exists():
        raise FileNotFoundError(f"dataset.yaml not found: {data_path}")
    if not weights_path.exists():
        raise FileNotFoundError(f"weights not found: {weights_path}")

    # ✅ 从 best.pt 推断 run_name：runs/detect/<run_name>/weights/best.pt
    run_name = weights_path.parent.parent.name  # weights 的上一级目录名

    project = "runs"
    name = f"detect/{run_name}/eval_{args.split}"  # -> runs/detect/<run_name>/eval_val|eval_test

    model = YOLO(str(weights_path))
    metrics = model.val(
        data=str(data_path),
        imgsz=args.imgsz,
        device=args.device,
        split=args.split,
        project=project,
        name=name,
        exist_ok=True,
        plots=args.plots,
        verbose=True
    )

    # ✅ 稳定拿到标量指标
    rd = getattr(metrics, "results_dict", None) or {}
    p = rd.get("metrics/precision(B)", None)
    r = rd.get("metrics/recall(B)", None)
    map50 = rd.get("metrics/mAP50(B)", None)
    map5095 = rd.get("metrics/mAP50-95(B)", None)

    print(f"\nDone. Split={args.split}")
    if None not in (p, r, map50, map5095):
        print(f"P={p:.4f}  R={r:.4f}  mAP50={map50:.4f}  mAP50-95={map5095:.4f}")
    else:
        print("Metrics dict missing keys; please use the 'all' line above.")
    print(f"Saved to: {project}/{name}")


if __name__ == "__main__":
    main()
