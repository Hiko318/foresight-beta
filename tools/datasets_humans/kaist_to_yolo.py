import argparse
from pathlib import Path
import shutil
import cv2
from tqdm import tqdm


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def parse_kaist_line(line: str):
    # Expected format: filename x y w h [optional flags]
    parts = line.strip().split()
    if len(parts) < 5:
        return None
    fname = parts[0]
    try:
        x = float(parts[1]); y = float(parts[2]); w = float(parts[3]); h = float(parts[4])
    except Exception:
        return None
    # Detect ignores/occlusions if extra tokens present
    flags = set(p.lower() for p in parts[5:])
    if any(f in {"ignore", "ignored", "crowd", "occluded", "occlusion"} for f in flags):
        return None
    return fname, x, y, w, h


def find_image(base: Path, fname: str, modality: str):
    # Modality directories may be named 'rgb', 'visible', 'lwir', 'thermal'
    candidates_dirs = []
    if modality == "rgb":
        candidates_dirs = ["rgb", "visible", "RGB", "VIS"]
    elif modality == "thermal":
        candidates_dirs = ["lwir", "thermal", "LWIR", "IR"]
    else:
        candidates_dirs = ["rgb", "visible", "lwir", "thermal"]

    for d in candidates_dirs:
        for ext in (".jpg", ".png", ".jpeg"):
            p = base / d / fname
            if p.suffix.lower() != ext:
                p = base / d / (Path(fname).stem + ext)
            if p.exists():
                return p
    # Fallback in base
    p = base / fname
    if p.exists():
        return p
    return None


def convert_split(kaist_root: Path, split: str, out_images: Path, out_labels: Path, modality: str, dual: bool):
    ann_dir = kaist_root / "raw" / "annotations" / split
    img_base = kaist_root / "raw" / "images" / split
    ensure_dir(out_images); ensure_dir(out_labels)

    for ann_file in tqdm(list(ann_dir.glob("*.txt")), desc=f"KAIST {split} ({modality}{'+dual' if dual else ''})"):
        lines = []
        img_path_rgb = None
        for line in ann_file.read_text().splitlines():
            parsed = parse_kaist_line(line)
            if not parsed:
                continue
            fname, x, y, w, h = parsed
            # choose modality image
            if modality == "rgb":
                img_path = find_image(img_base, fname, "rgb")
            elif modality == "thermal":
                img_path = find_image(img_base, fname, "thermal")
            else:
                img_path = find_image(img_base, fname, None)

            if img_path is None:
                continue
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            H, W = img.shape[:2]
            x_c = (x + w / 2.0) / W
            y_c = (y + h / 2.0) / H
            nw = w / W
            nh = h / H
            lines.append((img_path, f"0 {x_c:.6f} {y_c:.6f} {nw:.6f} {nh:.6f}"))

        # write labels and copy images
        for img_path, yline in lines:
            dst_img = out_images / img_path.name
            if not dst_img.exists():
                shutil.copy2(img_path, dst_img)
            (out_labels / f"{Path(img_path).stem}.txt").write_text(yline + "\n")

            if dual and modality == "rgb":
                # attempt to find thermal counterpart
                t_img = find_image(img_base, img_path.name, "thermal")
                if t_img and t_img.exists():
                    tdst = out_images / (Path(t_img).stem + "_thermal" + t_img.suffix)
                    if not tdst.exists():
                        shutil.copy2(t_img, tdst)
                    (out_labels / f"{Path(tdst).stem}.txt").write_text(yline + "\n")


def main():
    ap = argparse.ArgumentParser(description="Convert KAIST annotations to YOLO person labels")
    ap.add_argument("--kaist-root", default=str(Path("datasets/kaist")), help="Root path for KAIST dataset")
    ap.add_argument("--thermal", action="store_true", help="Use thermal images instead of RGB")
    ap.add_argument("--dual", action="store_true", help="Duplicate entries with thermal counterparts")
    args = ap.parse_args()

    kaist_root = Path(args.kaist_root)
    modality = "thermal" if args.thermal else "rgb"
    out_images_train = kaist_root / "images" / "train"
    out_labels_train = kaist_root / "labels" / "train"
    out_images_val = kaist_root / "images" / "val"
    out_labels_val = kaist_root / "labels" / "val"

    convert_split(kaist_root, "train", out_images_train, out_labels_train, modality, args.dual)
    convert_split(kaist_root, "val", out_images_val, out_labels_val, modality, args.dual)


if __name__ == "__main__":
    main()