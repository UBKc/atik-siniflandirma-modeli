"""Eğitilen modeli test kümesinde değerlendirir.

Test kümesi eğitimde ve model seçiminde hiç kullanılmadı. Çıktılar:
- doğruluk, sınıf bazında precision / recall / F1
- karışıklık matrisi (sonuclar/karisiklik_matrisi.png)
- akıllı geri dönüşüm kutusundaki 5 bölmeye göre doğruluk
- işlemcide tek görüntü için ortalama tahmin süresi

Kullanım:
    python degerlendir.py model/atik_yolov8n_cls.pt
"""
import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from ultralytics import YOLO

from kutu_eslestirme import KUTU

TEST = Path("veri/test")
SONUC = Path("sonuclar")


def main():
    model_yolu = sys.argv[1] if len(sys.argv) > 1 else "model/atik_yolov8n_cls.pt"
    model = YOLO(model_yolu)
    siniflar = [model.names[i] for i in range(len(model.names))]
    SONUC.mkdir(exist_ok=True)

    gercek, tahmin, sureler = [], [], []
    dosyalar = sorted(TEST.glob("*/*.jpg"))

    # İlk birkaç tahmin model yüklenmesi yüzünden yavaş, süreye katılmıyor
    for dosya in dosyalar[:5]:
        model(str(dosya), imgsz=224, verbose=False, device="cpu")

    for dosya in dosyalar:
        t0 = time.perf_counter()
        sonuc = model(str(dosya), imgsz=224, verbose=False, device="cpu")[0]
        sureler.append((time.perf_counter() - t0) * 1000)
        gercek.append(dosya.parent.name)
        tahmin.append(siniflar[int(sonuc.probs.top1)])

    dogruluk = accuracy_score(gercek, tahmin)
    print(f"Test görüntüsü: {len(dosyalar)}")
    print(f"Doğruluk (6 sınıf): %{dogruluk * 100:.2f}\n")
    rapor = classification_report(gercek, tahmin, labels=siniflar, digits=3, output_dict=True)
    print(classification_report(gercek, tahmin, labels=siniflar, digits=3))

    # Kutudaki 5 bölmeye göre: karton ile kağıt aynı bölmeye gidiyor
    kutu_gercek = [KUTU[s] for s in gercek]
    kutu_tahmin = [KUTU[s] for s in tahmin]
    kutu_dogruluk = accuracy_score(kutu_gercek, kutu_tahmin)
    print(f"Doğruluk (kutudaki 5 bölme): %{kutu_dogruluk * 100:.2f}")

    ort_sure = float(np.mean(sureler))
    print(f"Ortalama tahmin süresi (CPU, tek görüntü): {ort_sure:.1f} ms")

    # Karışıklık matrisi
    cm = confusion_matrix(gercek, tahmin, labels=siniflar)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(siniflar)), siniflar, rotation=45)
    ax.set_yticks(range(len(siniflar)), siniflar)
    ax.set_xlabel("Tahmin")
    ax.set_ylabel("Gerçek")
    for i in range(len(siniflar)):
        for j in range(len(siniflar)):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_title(f"Test kümesi - doğruluk %{dogruluk * 100:.1f}")
    fig.tight_layout()
    fig.savefig(SONUC / "karisiklik_matrisi.png", dpi=120)

    ozet = {
        "test_goruntu_sayisi": len(dosyalar),
        "dogruluk_6_sinif": round(dogruluk, 4),
        "dogruluk_5_bolme": round(kutu_dogruluk, 4),
        "ortalama_sure_ms": round(ort_sure, 1),
        "siniflar": {s: {k: round(v, 3) for k, v in rapor[s].items()} for s in siniflar},
        "karisiklik_matrisi": cm.tolist(),
    }
    (SONUC / "test_sonuclari.json").write_text(json.dumps(ozet, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
