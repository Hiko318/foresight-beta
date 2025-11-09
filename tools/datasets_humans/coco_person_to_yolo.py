import argparse
import json
import os
import shutil
from pathlib import Path

from tqdm import tqdm
import cv2
from pycocotools.coco import COCO


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def coco_category_id_for_person(coco: COCO) -> int:
    cats = coco.loadCats(coco.getCatIds())
    for c in cats:
        if c.get("name") == "person":
            return c.get("id")
    # Fallback to common COCO 2017 id
    return 1


def to_yolo(x, y, w, h, img_w, img_h):
    x_c = (x + w / 2.0) / img_w
    y_c = (y + h / 2.0) / img_h
    return x_c, y_c, w / img_w, h / img_h


def convert_split(coco_root: Path, split: str, out_images: Path, out_labels: Path, limit: int | None):
    ann_path = coco_root / "raw" / "annotations" / f"instances_{split}.json"
    img_root = coco_root / "raw" / split
    ensure_dir(out_images)
    ensure_dir(out_labels)

    coco = COCO(str(ann_path))
    person_id = coco_category_id_for_person(coco)

    img_ids = coco.getImgIds()
    if limit:
        img_ids = img_ids[:limit]

    for img_id in tqdm(img_ids, desc=f"COCO {split}"):
        img_info = coco.loadImgs([img_id])[0]
        file_name = img_info["file_name"]
        src_img = img_root / file_name
        if not src_img.exists():
            continue
        # copy image
        dst_img = out_images / file_name
        ensure_dir(dst_img.parent)
        if not dst_img.exists():
            shutil.copy2(src_img, dst_img)

        # labels
        ann_ids = coco.getAnnIds(imgIds=[img_id], catIds=[person_id], iscrowd=None)
        anns = coco.loadAnns(ann_ids)
        if not anns:
            # still create empty label to be explicit
            (out_labels / (Path(file_name).stem + ".txt")).write_text("")
            continue

        # image size
        img = cv2.imread(str(src_img))
        if img is None:
            continue
        h, w = img.shape[:2]

        lines = []
        for a in anns:
            if a.get("iscrowd", 0) == 1:
                continue
            x, y, bw, bh = a["bbox"]
            x_c, y_c, nw, nh = to_yolo(x, y, bw, bh, w, h)
            lines.append(f"0 {x_c:.6f} {y_c:.6f} {nw:.6f} {nh:.6f}")

        (out_labels / (Path(file_name).stem + ".txt")).write_text("\n".join(lines))


def main():
    ap = argparse.ArgumentParser(description="Convert COCO person annotations to YOLO labels")
    ap.add_argument("--coco-root", default=str(Path("datasets/coco2017")), help="Root path for COCO dataset")
    ap.add_argument("--limit", type=int, default=None, help="Optional limit to number of images per split")
    args = ap.parse_args()

    coco_root = Path(args.coco_root)
    out_images_train = coco_root / "images" / "train2017"
    out_labels_train = coco_root / "labels" / "train"
    out_images_val = coco_root / "images" / "val2017"
    out_labels_val = coco_root / "labels" / "val"

    convert_split(coco_root, "train2017", out_images_train, out_labels_train, args.limit)
    convert_split(coco_root, "val2017", out_images_val, out_labels_val, args.limit)


if __name__ == "__main__":
    main()