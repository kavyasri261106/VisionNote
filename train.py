"""
train.py
Fine-tunes YOLOv8n on your currency note dataset.

Expected dataset layout (standard Ultralytics/YOLO format):

    dataset/
        images/
            train/   *.jpg
            val/     *.jpg
        labels/
            train/   *.txt   (YOLO format: class x_center y_center width height, normalized)
            val/     *.txt
        data.yaml

If you labeled your images in Roboflow, export in "YOLOv8" format and it
will give you this exact structure plus a ready-made data.yaml — just drop
it into dataset/ and you're set.

Usage:
    python train.py
    python train.py --epochs 80 --imgsz 640 --batch 16
"""

import argparse
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description="Train VisionNote currency detector")
    parser.add_argument("--data", default="dataset/data.yaml", help="Path to data.yaml")
    parser.add_argument("--model", default="yolov8n.pt", help="Base model to fine-tune from")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--name", default="visionnote_run", help="Run name under runs/detect/")
    args = parser.parse_args()

    model = YOLO(args.model)

    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        name=args.name,
        patience=15,       # early stop if val metrics plateau
        augment=True,      # built-in mosaic/flip/hsv augmentation
    )

    # Validate and print final metrics (precision, recall, mAP)
    metrics = model.val()
    print("\n=== Final Validation Metrics ===")
    print(metrics)

    # Best weights are auto-saved to runs/detect/<name>/weights/best.pt
    print(f"\nBest weights saved to: runs/detect/{args.name}/weights/best.pt")
    print("Copy that file to model/trained_model.pt to use it in app.py")


if __name__ == "__main__":
    main()
