import argparse
import os
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/dataset.yaml")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--model", default="yolo26n-seg.pt")
    parser.add_argument("--project", default="models")
    parser.add_argument("--batch", type=int, default=8)
    args = parser.parse_args()

    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        project=os.path.abspath(args.project),
        name="bagvision",
        batch=args.batch,
    )


if __name__ == "__main__":
    main()
