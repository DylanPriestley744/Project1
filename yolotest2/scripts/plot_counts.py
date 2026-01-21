import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="veh_demo_counts.csv path")
    ap.add_argument("--outdir", default="assets/plots", help="output folder")
    ap.add_argument("--per_second", action="store_true", help="aggregate by second")
    args = ap.parse_args()

    csv_path = Path(args.csv)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    for c in ["total_dets","Ambulance","Bus","Car","Motorcycle","Truck"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)

    # 1) total_dets vs time
    plt.figure()
    plt.plot(df["time_sec"], df["total_dets"])
    plt.xlabel("time (s)")
    plt.ylabel("total detections per frame")
    plt.title("Total detections vs time")
    plt.tight_layout()
    plt.savefig(outdir / "total_dets_vs_time.png", dpi=200)
    plt.close()

    # 2) per-class vs time
    plt.figure(figsize=(10,5))
    for c in ["Ambulance","Bus","Car","Motorcycle","Truck"]:
        plt.plot(df["time_sec"], df[c], label=c)
    plt.xlabel("time (s)")
    plt.ylabel("detections per frame")
    plt.title("Per-class detections vs time")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "per_class_dets_vs_time.png", dpi=200)
    plt.close()

    # 3) per-second aggregated (recommended)
    if args.per_second:
        df2 = df.copy()
        df2["sec"] = df2["time_sec"].round().astype(int)
        g = df2.groupby("sec", as_index=False)["total_dets"].mean()

        plt.figure()
        plt.plot(g["sec"], g["total_dets"])
        plt.xlabel("time (s)")
        plt.ylabel("avg detections per second")
        plt.title("Total detections per second (avg over frames)")
        plt.tight_layout()
        plt.savefig(outdir / "total_dets_per_second.png", dpi=200)
        plt.close()

    print("[Saved]", (outdir / "total_dets_vs_time.png"))
    print("[Saved]", (outdir / "per_class_dets_vs_time.png"))
    if args.per_second:
        print("[Saved]", (outdir / "total_dets_per_second.png"))

if __name__ == "__main__":
    main()
