import argparse
import subprocess
from pathlib import Path


def run(cmd, cwd=None):
    print("$", " ".join(cmd))
    return subprocess.run(cmd, cwd=cwd, check=True)


def ensure_dirs():
    for p in [
        Path("datasets/kaist/raw"),
        Path("datasets/visdrone/raw"),
        Path("datasets/coco2017/raw"),
        Path("datasets/humans_mix"),
    ]:
        p.mkdir(parents=True, exist_ok=True)


def maybe_download_ultralytics_visdrone():
    # Try Ultralytics built-in downloader via CLI
    try:
        run(["python", "-m", "ultralytics", "cfg"])
        # As of Ultralytics recent versions, datasets may be auto-downloaded via the YAML
        # We will just log instructions for manual download to raw if auto fails.
    except Exception:
        print("[WARN] Ultralytics cfg check failed. Please download VisDrone manually to datasets/visdrone/raw/")


def convert_all(use_rgb: bool, thermal: bool, dual: bool, limit_coco: int | None):
    # COCO
    run(["python", "tools/datasets_humans/coco_person_to_yolo.py", "--coco-root", "datasets/coco2017", *( ["--limit", str(limit_coco)] if limit_coco else [] )])
    # VisDrone
    run(["python", "tools/datasets_humans/visdrone_to_yolo_people.py", "--visdrone-root", "datasets/visdrone"])
    # KAIST
    kaist_cmd = ["python", "tools/datasets_humans/kaist_to_yolo.py", "--kaist-root", "datasets/kaist"]
    if thermal:
        kaist_cmd.append("--thermal")
    if dual:
        kaist_cmd.append("--dual")
    run(kaist_cmd)


def merge_all():
    run(["python", "tools/datasets_humans/merge_humans_dataset.py", "--coco-root", "datasets/coco2017", "--visdrone-root", "datasets/visdrone", "--kaist-root", "datasets/kaist", "--out-root", "datasets/humans_mix"])


def main():
    ap = argparse.ArgumentParser(description="Prepare merged human dataset from COCO/VisDrone/KAIST")
    ap.add_argument("--use_rgb", action="store_true", help="Use RGB for KAIST (default)")
    ap.add_argument("--thermal", action="store_true", help="Use thermal instead of RGB for KAIST")
    ap.add_argument("--dual", action="store_true", help="Duplicate KAIST entries with thermal counterparts")
    ap.add_argument("--limit_coco", type=int, default=None, help="Limit COCO images per split")
    ap.add_argument("--balance", action="store_true", help="Reserved flag for future balancing")
    args = ap.parse_args()

    ensure_dirs()
    maybe_download_ultralytics_visdrone()
    convert_all(use_rgb=args.use_rgb or (not args.thermal), thermal=args.thermal, dual=args.dual, limit_coco=args.limit_coco)
    merge_all()

    print("\nDone. Merged dataset at datasets/humans_mix. YAML at datasets/humans_mix/humans_mix.yaml")


if __name__ == "__main__":
    main()