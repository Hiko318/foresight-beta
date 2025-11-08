"""
Prepare a unified YOLOv8 dataset for human detection by merging:
- COCO 2017 (person class only)
- VisDrone2019-DET (person/pedestrian/people classes)
- KAIST Multispectral Pedestrian (person)

This script converts each source dataset's annotations to YOLO format
and copies images/labels into a single target structure:

  datasets/foresight-human/
    images/train/, images/val/
    labels/train/, labels/val/

Usage (PowerShell):
  py -3 foresight-beta/scripts/prepare_human_datasets.py `
     --coco-root C:\data\coco2017 `
     --visdrone-root C:\data\VisDrone2019-DET `
     --kaist-root C:\data\KAIST `
     --out-root C:\data\datasets

Notes:
- COCO: expects the COCO 2017 structure with annotations JSON.
- VisDrone: expects the DET subset with `annotations` txt files.
- KAIST: annotations vary by source. Provide `annotations` txt
  files with lines: <filename> <x> <y> <w> <h>. If your
  distribution differs, adapt parse_kaist_annotations().

Dependencies: pip install ultralytics pycocotools opencv-python tqdm
"""

import os
import json
import shutil
from pathlib import Path
from typing import List, Tuple

import cv2
from tqdm import tqdm


def ensure_dirs(root: Path):
    (root / 'images' / 'train').mkdir(parents=True, exist_ok=True)
    (root / 'images' / 'val').mkdir(parents=True, exist_ok=True)
    (root / 'labels' / 'train').mkdir(parents=True, exist_ok=True)
    (root / 'labels' / 'val').mkdir(parents=True, exist_ok=True)


def coco_to_yolo_bbox(x, y, w, h, img_w, img_h):
    cx = (x + w / 2) / img_w
    cy = (y + h / 2) / img_h
    nw = w / img_w
    nh = h / img_h
    return cx, cy, nw, nh


def convert_coco_person(coco_root: Path, split: str, out_root: Path, class_id: int = 0):
    """Convert COCO <split> (train/val) person annotations to YOLO format."""
    ann_path = coco_root / 'annotations' / f'instances_{split}2017.json'
    images_dir = coco_root / f'{split}2017'
    if not ann_path.exists():
        print(f"[COCO] Missing: {ann_path}")
        return 0
    with open(ann_path, 'r') as f:
        data = json.load(f)
    id_to_image = {img['id']: img for img in data['images']}
    person_cat_ids = {cat['id'] for cat in data['categories'] if cat['name'] == 'person'}
    anns_by_image = {}
    for ann in data['annotations']:
        if ann.get('iscrowd', 0) == 1:
            continue
        if ann['category_id'] in person_cat_ids:
            anns_by_image.setdefault(ann['image_id'], []).append(ann)
    count = 0
    for img_id, anns in tqdm(anns_by_image.items(), desc=f'COCO {split} person'):
        img_info = id_to_image[img_id]
        img_file = images_dir / img_info['file_name']
        if not img_file.exists():
            continue
        img = cv2.imread(str(img_file))
        if img is None:
            continue
        img_h, img_w = img.shape[:2]
        # Copy image
        out_img = out_root / 'images' / split / img_file.name
        shutil.copy2(img_file, out_img)
        # Labels
        out_lbl = out_root / 'labels' / split / (img_file.stem + '.txt')
        lines = []
        for ann in anns:
            x, y, w, h = ann['bbox']
            cx, cy, nw, nh = coco_to_yolo_bbox(x, y, w, h, img_w, img_h)
            lines.append(f"{class_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n")
        out_lbl.write_text(''.join(lines))
        count += 1
    return count


def parse_visdrone_annotation_line(line: str) -> Tuple[int, int, int, int, int]:
    # Format: x,y,w,h,score,category,truncation,occlusion
    parts = [p.strip() for p in line.split(',')]
    if len(parts) < 6:
        return None
    x, y, w, h = map(int, parts[:4])
    category = int(parts[5])
    return x, y, w, h, category


VISDRONE_PERSON_CATS = {1, 4, 5}  # pedestrian=1, person=4, people=5


def convert_visdrone_person(vis_root: Path, split: str, out_root: Path, class_id: int = 0):
    imgs_dir = vis_root / f'images/{split}'
    anns_dir = vis_root / f'annotations/{split}'
    if not imgs_dir.exists() or not anns_dir.exists():
        print(f"[VisDrone] Missing dirs: {imgs_dir} or {anns_dir}")
        return 0
    count = 0
    for ann_file in tqdm(list(anns_dir.glob('*.txt')), desc=f'VisDrone {split} person'):
        # corresponding image name
        img_name = ann_file.stem + '.jpg'
        img_file = imgs_dir / img_name
        if not img_file.exists():
            # some sets use .png
            alt = imgs_dir / (ann_file.stem + '.png')
            if alt.exists():
                img_file = alt
            else:
                continue
        img = cv2.imread(str(img_file))
        if img is None:
            continue
        img_h, img_w = img.shape[:2]
        # Copy image
        out_img = out_root / 'images' / split / img_file.name
        shutil.copy2(img_file, out_img)
        # Labels
        out_lbl = out_root / 'labels' / split / (img_file.stem + '.txt')
        lines = []
        for line in ann_file.read_text().splitlines():
            parsed = parse_visdrone_annotation_line(line)
            if not parsed:
                continue
            x, y, w, h, cat = parsed
            if cat not in VISDRONE_PERSON_CATS:
                continue
            cx, cy, nw, nh = coco_to_yolo_bbox(x, y, w, h, img_w, img_h)
            lines.append(f"{class_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n")
        out_lbl.write_text(''.join(lines))
        count += 1
    return count


def parse_kaist_annotations(kaist_ann_dir: Path) -> List[Tuple[Path, List[Tuple[int,int,int,int]]]]:
    """Expect text files with lines: <filename> <x> <y> <w> <h> (person only).
    Return list of (image_path, boxes). Adapt if your KAIST format differs.
    """
    results = []
    for txt in kaist_ann_dir.rglob('*.txt'):
        boxes_by_img = {}
        for line in txt.read_text().splitlines():
            parts = line.strip().split()
            if len(parts) != 5:
                # allow: <filename> <x> <y> <w> <h> <cls>
                if len(parts) >= 6:
                    parts = parts[:5]
                else:
                    continue
            fname, x, y, w, h = parts[0], *map(int, parts[1:])
            boxes_by_img.setdefault(fname, []).append((x, y, w, h))
        for fname, boxes in boxes_by_img.items():
            results.append((txt.parent / fname, boxes))
    return results


def convert_kaist_person(kaist_root: Path, out_root: Path, split: str, class_id: int = 0):
    imgs_dir = kaist_root / f'images/{split}'
    ann_dir = kaist_root / f'annotations/{split}'
    if not imgs_dir.exists() or not ann_dir.exists():
        print(f"[KAIST] Missing dirs: {imgs_dir} or {ann_dir}")
        return 0
    items = parse_kaist_annotations(ann_dir)
    count = 0
    for img_path, boxes in tqdm(items, desc=f'KAIST {split} person'):
        # resolve image path
        img_file = imgs_dir / img_path.name if not img_path.is_absolute() else img_path
        if not img_file.exists():
            continue
        img = cv2.imread(str(img_file), cv2.IMREAD_UNCHANGED)
        if img is None:
            continue
        # If thermal/grayscale, expand to 3 channels to keep training consistent
        if len(img.shape) == 2 or (img.shape[2] == 1 if len(img.shape) == 3 else False):
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        img_h, img_w = img.shape[:2]
        out_img = out_root / 'images' / split / img_file.name
        cv2.imwrite(str(out_img), img)
        # Labels
        out_lbl = out_root / 'labels' / split / (img_file.stem + '.txt')
        lines = []
        for x, y, w, h in boxes:
            cx, cy, nw, nh = coco_to_yolo_bbox(x, y, w, h, img_w, img_h)
            lines.append(f"{class_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n")
        out_lbl.write_text(''.join(lines))
        count += 1
    return count


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--coco-root', type=Path, required=False)
    p.add_argument('--visdrone-root', type=Path, required=False)
    p.add_argument('--kaist-root', type=Path, required=False)
    p.add_argument('--out-root', type=Path, required=True, help='Root to create datasets/foresight-human')
    args = p.parse_args()

    out_root = args.out_root / 'datasets' / 'foresight-human'
    ensure_dirs(out_root)

    total = 0
    # COCO
    if args.coco_root:
        total += convert_coco_person(args.coco_root, 'train', out_root)
        total += convert_coco_person(args.coco_root, 'val', out_root)
    # VisDrone
    if args.visdrone_root:
        total += convert_visdrone_person(args.visdrone_root, 'train', out_root)
        total += convert_visdrone_person(args.visdrone_root, 'val', out_root)
    # KAIST
    if args.kaist_root:
        total += convert_kaist_person(args.kaist_root, out_root, 'train')
        total += convert_kaist_person(args.kaist_root, out_root, 'val')

    print(f"[DONE] Prepared {total} labeled images at: {out_root}")


if __name__ == '__main__':
    main()