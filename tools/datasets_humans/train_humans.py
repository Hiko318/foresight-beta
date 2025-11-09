import argparse
from pathlib import Path
from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser(description="Train YOLO on merged humans_mix dataset")
    ap.add_argument("--weights", default="yolo11m.pt", help="Pretrained weights: yolo11m.pt or yolov8m.pt")
    ap.add_argument("--data", default="datasets/humans_mix/humans_mix.yaml")
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", default="auto")
    ap.add_argument("--device", default=0)
    ap.add_argument("--lr0", type=float, default=0.005)
    ap.add_argument("--weight_decay", type=float, default=0.0005)
    ap.add_argument("--mosaic", type=float, default=1.0)
    ap.add_argument("--hsv_h", type=float, default=0.015)
    ap.add_argument("--hsv_s", type=float, default=0.7)
    ap.add_argument("--hsv_v", type=float, default=0.4)
    ap.add_argument("--flipud", type=float, default=0.0)
    ap.add_argument("--fliplr", type=float, default=0.5)
    ap.add_argument("--mixup", type=float, default=0.05)
    ap.add_argument("--cache", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--val", action="store_true", help="Run validation after training")
    ap.add_argument("--export_onnx", action="store_true", help="Export ONNX after training")
    ap.add_argument("--export_trt", action="store_true", help="Export TensorRT engine (requires TensorRT)")
    args = ap.parse_args()

    m = YOLO(args.weights)
    batch_val = -1 if str(args.batch).lower() == "auto" else int(args.batch)

    m.train(
        data=args.data,
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=batch_val,
        device=args.device,
        lr0=args.lr0,
        weight_decay=args.weight_decay,
        mosaic=args.mosaic,
        hsv_h=args.hsv_h,
        hsv_s=args.hsv_s,
        hsv_v=args.hsv_v,
        flipud=args.flipud,
        fliplr=args.fliplr,
        mixup=args.mixup,
        cache=args.cache,
        workers=args.workers,
        project="runs/detect",
        name="humans_mix"
    )

    if args.val:
        m.val(data=args.data, imgsz=args.imgsz, device=args.device)

    if args.export_onnx:
        m.export(format="onnx", opset=13)
    if args.export_trt:
        m.export(format="engine", half=True, dynamic=True)


if __name__ == "__main__":
    main()