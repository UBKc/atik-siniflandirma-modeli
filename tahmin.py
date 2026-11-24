"""Tek bir fotoğraf için atık türü tahmini.

Kullanım:
    python tahmin.py foto.jpg
    python tahmin.py foto.jpg model/atik_yolov8n_cls.pt
"""
import sys

from ultralytics import YOLO

from kutu_eslestirme import KUTU


def main():
    if len(sys.argv) < 2:
        print("Kullanım: python tahmin.py foto.jpg [model.pt]")
        return

    foto = sys.argv[1]
    model_yolu = sys.argv[2] if len(sys.argv) > 2 else "model/atik_yolov8n_cls.pt"
    model = YOLO(model_yolu)

    sonuc = model(foto, imgsz=224, verbose=False)[0]
    olasiliklar = sonuc.probs.data.tolist()
    sirali = sorted(range(len(olasiliklar)), key=lambda i: olasiliklar[i], reverse=True)

    for i in sirali[:3]:
        print(f"{model.names[i]:<8} %{olasiliklar[i] * 100:5.1f}")

    en_iyi = model.names[sirali[0]]
    print(f"\nKutu bölmesi: {KUTU[en_iyi]}")


if __name__ == "__main__":
    main()
