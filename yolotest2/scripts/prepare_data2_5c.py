import argparse, shutil
from pathlib import Path

OLD_NAMES = ['ambulance', 'army vehicle', 'auto rickshaw', 'bicycle', 'bus', 'car', 'garbagevan',
             'human hauler', 'minibus', 'minivan', 'motorbike', 'pickup', 'policecar', 'rickshaw',
             'scooter', 'suv', 'taxi', 'three wheelers -CNG-', 'truck', 'van', 'wheelbarrow']

NEW_NAMES = ["Ambulance","Bus","Car","Motorcycle","Truck"]  # 0..4

MAP = {
    'ambulance': 0,
    'bus': 1, 'minibus': 1,
    'car': 2, 'suv': 2, 'taxi': 2, 'policecar': 2, 'van': 2, 'minivan': 2, 'pickup': 2,
    'auto rickshaw': 2, 'rickshaw': 2, 'three wheelers -CNG-': 2, 'human hauler': 2,
    'motorbike': 3, 'scooter': 3,
    'truck': 4, 'garbagevan': 4, 'army vehicle': 4,
    'bicycle': -1, 'wheelbarrow': -1,
}

def convert_split(src_img_dir: Path, src_lbl_dir: Path, dst_img_dir: Path, dst_lbl_dir: Path):
    dst_img_dir.mkdir(parents=True, exist_ok=True)
    dst_lbl_dir.mkdir(parents=True, exist_ok=True)

    img_paths = sorted([p for p in src_img_dir.glob("*") if p.suffix.lower() in [".jpg",".jpeg",".png"]])
    kept_images = 0
    obj_cnt = [0]*len(NEW_NAMES)

    for img in img_paths:
        lbl = src_lbl_dir / (img.stem + ".txt")
        dst_img = dst_img_dir / img.name
        dst_lbl = dst_lbl_dir / (img.stem + ".txt")

        shutil.copy2(img, dst_img)

        new_lines = []
        if lbl.exists():
            txt = lbl.read_text(encoding="utf-8", errors="ignore").strip()
            if txt:
                for line in txt.splitlines():
                    parts = line.strip().split()
                    if len(parts) != 5:
                        continue
                    old_id = int(parts[0])
                    if 0 <= old_id < len(OLD_NAMES):
                        old_name = OLD_NAMES[old_id]
                        new_id = MAP.get(old_name, -1)
                        if new_id == -1:
                            continue
                        parts[0] = str(new_id)
                        new_lines.append(" ".join(parts))
                        obj_cnt[new_id] += 1

        dst_lbl.write_text("\n".join(new_lines), encoding="utf-8")
        kept_images += 1

    return kept_images, obj_cnt

def write_yaml(dst_root: Path, yaml_path: Path):
    txt = f"""path: {dst_root.as_posix()}
train: train/images
val: valid/images
test: valid/images
names: {NEW_NAMES}
nc: 5
"""
    yaml_path.write_text(txt, encoding="utf-8")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help=".../trafic_data")
    ap.add_argument("--dst", default="datasets/RoadVehicleData2_5c")
    ap.add_argument("--yaml_out", default="veh5_data2.yaml")
    args = ap.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)
    yaml_out = Path(args.yaml_out)

    tr_n, tr_cnt = convert_split(src/"train/images", src/"train/labels", dst/"train/images", dst/"train/labels")
    va_n, va_cnt = convert_split(src/"valid/images", src/"valid/labels", dst/"valid/images", dst/"valid/labels")

    write_yaml(dst, yaml_out)

    print("Done.")
    print(f"Train images: {tr_n}, Valid images: {va_n}")
    print("Objects per class (Train+Valid):")
    for i, n in enumerate(NEW_NAMES):
        print(f"  {i}:{n:<12}  {tr_cnt[i]+va_cnt[i]}")
    print(f"YAML saved to: {yaml_out}")

if __name__ == "__main__":
    main()
