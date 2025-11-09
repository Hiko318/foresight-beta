import argparse
import os
from pathlib import Path
import shutil

import cv2
from tqdm import tqdm


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def parse_visdrone_ann(line: str):
    parts = [p.strip() for p in line.split(',')]
    if len(parts) < 6:
        return None
    try:
        x = float(parts[0]); y = float(parts[1]); w = float(parts[2]); h = float(parts[3])
        # parts[4] = score (ignored), parts[5] = category
        cat = int(parts[5])
        return x, y, w, h, cat
    except Exception:
        return None


def keep_category(cat: int) -> bool:
    # Support both 0/1 mapping and the official 1/4/5 mapping
    return cat in {0, 1, 4, 5}


def convert_split(vis_root: Path, split: str, out_images: Path, out_labels: Path):
    img_dir = vis_root / "raw" / "images" / split
    ann_dir = vis_root / "raw" / "annotations" / split
    ensure_dir(out_images)
    ensure_dir(out_labels)

    for ann_file in tqdm(list(ann_dir.glob("*.txt")), desc=f"VisDrone {split}"):
        stem = ann_file.stem
        # images may be .jpg or .png
        img_path = None
        for ext in (".jpg", ".png", ".jpeg", ".JPG"):
            candidate = img_dir / f"{stem}{ext}"
            if candidate.exists():
                img_path = candidate
                break
        if img_path is None:
            continue

        dst_img = out_images / img_path.name
        ensure_dir(dst_img.parent)
        if not dst_img.exists():
            shutil.copy2(img_path, dst_img)

        img = cv2.imread(str(img_path))
        if img is None:
            # create empty label
            (out_labels / f"{stem}.txt").write_text("")
            continue
        h, w = img.shape[:2]

        lines = []
        for line in ann_file.read_text().splitlines():
            parsed = parse_visdrone_ann(line)
            if not parsed:
                continue
            x, y, bw, bh, cat = parsed
            if not keep_category(cat):
                continue
            x_c = (x + bw / 2.0) / w
            y_c = (y + bh / 2.0) / h
            nw = bw / w
            nh = bh / h
            lines.append(f"0 {x_c:.6f} {y_c:.6f} {nw:.6f} {nh:.6f}")

        (out_labels / f"{stem}.txt").write_text("\n".join(lines))


def main():
    ap = argparse.ArgumentParser(description="Convert VisDrone annotations to YOLO people-only labels")
    ap.add_argument("--visdrone-root", default=str(Path("datasets/visdrone")), help="Root path for VisDrone dataset")
    args = ap.parse_args()

    vis_root = Path(args.visdrone_root)
    out_images_train = vis_root / "images" / "train"
    out_labels_train = vis_root / "labels" / "train"
    out_images_val = vis_root / "images" / "val"
    out_labels_val = vis_root / "labels" / "val"

    convert_split(vis_root, "train", out_images_train, out_labels_train)
    convert_split(vis_root, "val", out_images_val, out_labels_val)


if __name__ == "__main__":
    main()