# yolov8n-cls'i trashnet uzerinde ince ayar ile egitiyor
# en kucuk modeli sectim cunku raspberry pi'de calisacak
# kullanim: python egit.py 30
import sys

from ultralytics import YOLO

EPOCH = int(sys.argv[1]) if len(sys.argv) > 1 else 30


def main():
    model = YOLO("yolov8n-cls.pt")  # imagenet agirliklari
    model.train(
        data="veri",
        epochs=EPOCH,
        imgsz=224,
        batch=32,
        patience=10,  # dogrulama 10 epoch iyilesmezse duruyor
        seed=0,
        deterministic=True,
        workers=2,
        project="calismalar",
        name="yolov8n_cls",
        exist_ok=True,
        plots=True,
    )


if __name__ == "__main__":
    main()
