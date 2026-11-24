"""YOLOv8n-cls modelini TrashNet üzerinde ince ayar (fine-tuning) ile eğitir.

ImageNet üzerinde önceden eğitilmiş yolov8n-cls ağırlıkları ile başlanıyor.
Küçük model seçildi çünkü hedef cihaz Raspberry Pi.

Kullanım:
    python egit.py            # varsayılan 30 epoch
    python egit.py 50         # epoch sayısı
"""
import sys

from ultralytics import YOLO

EPOCH = int(sys.argv[1]) if len(sys.argv) > 1 else 30


def main():
    model = YOLO("yolov8n-cls.pt")
    model.train(
        data="veri",
        epochs=EPOCH,
        imgsz=224,
        batch=32,
        patience=10,       # doğrulama başarısı 10 epoch artmazsa dur
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
