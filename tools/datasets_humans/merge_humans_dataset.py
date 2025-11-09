import argparse
import csv
from pathlib import Path
import shutil
from tqdm import tqdm


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def collect_pairs(images_dir: Path, labels_dir: Path):
    pairs = []
    for img in images_dir.glob("*.*"):
        stem = img.stem
        lab = labels_dir / f"{stem}.txt"
        if lab.exists():
            pairs.append((img, lab))
    return pairs


def copy_pairs(pairs, out_img_dir: Path, out_lab_dir: Path, max_count: int | None):
    ensure_dir(out_img_dir); ensure_dir(out_lab_dir)
    copied = 0
    for img, lab in pairs:
        if max_count and copied >= max_count:
            break
        dst_img = out_img_dir / img.name
        dst_lab = out_lab_dir / lab.name
        if not dst_img.exists():
            shutil.copy2(img, dst_img)
        if not dst_lab.exists():
            shutil.copy2(lab, dst_lab)
        copied += 1
    return copied


def write_yaml(root: Path):
    yaml_text = (
        "path: datasets/humans_mix\n"
        "train: images/train\n"
        "val: images/val\n"
        "names:\n  0: person\n"
    )
    (root / "humans_mix.yaml").write_text(yaml_text)


def main():
    ap = argparse.ArgumentParser(description="Merge COCO/VisDrone/KAIST YOLO person datasets into a single humans_mix")
    ap.add_argument("--coco-root", default=str(Path("datasets/coco2017")))
    ap.add_argument("--visdrone-root", default=str(Path("datasets/visdrone")))
    ap.add_argument("--kaist-root", default=str(Path("datasets/kaist")))
    ap.add_argument("--out-root", default=str(Path("datasets/humans_mix")))
    ap.add_argument("--max_per_source_train", type=int, default=80000)
    ap.add_argument("--max_per_source_val", type=int, default=10000)
    args = ap.parse_args()

    out_root = Path(args.out_root)
    out_train_img = out_root / "images" / "train"
    out_train_lab = out_root / "labels" / "train"
    out_val_img = out_root / "images" / "val"
    out_val_lab = out_root / "labels" / "val"
    ensure_dir(out_train_img); ensure_dir(out_train_lab)
    ensure_dir(out_val_img); ensure_dir(out_val_lab)

    sources = [
        (Path(args.coco_root) / "images" / "train2017", Path(args.coco_root) / "labels" / "train", "coco/train"),
        (Path(args.visdrone_root) / "images" / "train", Path(args.visdrone_root) / "labels" / "train", "visdrone/train"),
        (Path(args.kaist_root) / "images" / "train", Path(args.kaist_root) / "labels" / "train", "kaist/train"),
    ]
    sources_val = [
        (Path(args.coco_root) / "images" / "val2017", Path(args.coco_root) / "labels" / "val", "coco/val"),
        (Path(args.visdrone_root) / "images" / "val", Path(args.visdrone_root) / "labels" / "val", "visdrone/val"),
        (Path(args.kaist_root) / "images" / "val", Path(args.kaist_root) / "labels" / "val", "kaist/val"),
    ]

    summary_rows = []

    # Train
    for img_dir, lab_dir, name in sources:
        pairs = collect_pairs(img_dir, lab_dir)
        copied = copy_pairs(pairs, out_train_img, out_train_lab, args.max_per_source_train)
        summary_rows.append([name, "train", len(pairs), copied])

    # Val
    for img_dir, lab_dir, name in sources_val:
        pairs = collect_pairs(img_dir, lab_dir)
        copied = copy_pairs(pairs, out_val_img, out_val_lab, args.max_per_source_val)
        summary_rows.append([name, "val", len(pairs), copied])

    write_yaml(out_root)

    # CSV summary
    with open(out_root / "summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["source", "split", "available_pairs", "copied"])
        for r in summary_rows:
            w.writerow(r)

    print(f"Merged dataset written to: {out_root}")


if __name__ == "__main__":
    main()