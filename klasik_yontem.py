"""Karşılaştırma için derin öğrenme kullanmayan klasik yöntem: HOG + renk histogramı + SVM.

Her görüntüden iki tür özellik çıkarılıyor:
- HOG (yönlü gradyan histogramı): şekil ve kenar bilgisi
- HSV renk histogramı: cam/plastik gibi renkle ayrışabilen sınıflar için

SVM'in C ve gamma değerleri doğrulama kümesine göre seçiliyor, test kümesi
sadece en sonda bir kez kullanılıyor.

Kullanım:
    python klasik_yontem.py
"""
import json
import time
from pathlib import Path

import cv2
import numpy as np
from skimage.feature import hog
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from kutu_eslestirme import KUTU

VERI = Path("veri")
BOYUT = (128, 96)  # orijinal 512x384, oran korunarak küçültülüyor


def ozellik_cikar(dosya):
    img = cv2.imread(str(dosya))
    img = cv2.resize(img, BOYUT)

    gri = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hog_ozellik = hog(gri, orientations=9, pixels_per_cell=(16, 16), cells_per_block=(2, 2))

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    renk = cv2.calcHist([hsv], [0, 1, 2], None, [8, 4, 4], [0, 180, 0, 256, 0, 256]).flatten()
    renk = renk / renk.sum()

    return np.concatenate([hog_ozellik, renk])


def kume_yukle(bolum):
    X, y = [], []
    for dosya in sorted((VERI / bolum).glob("*/*.jpg")):
        X.append(ozellik_cikar(dosya))
        y.append(dosya.parent.name)
    return np.array(X), np.array(y)


def main():
    t0 = time.time()
    X_train, y_train = kume_yukle("train")
    X_val, y_val = kume_yukle("val")
    X_test, y_test = kume_yukle("test")
    print(f"Özellik boyutu: {X_train.shape[1]}, çıkarma süresi: {time.time() - t0:.1f} sn")

    # Doğrulama kümesine göre basit ızgara araması
    en_iyi = None
    for C in [1, 10, 100]:
        for gamma in ["scale", 0.001, 0.0001]:
            model = make_pipeline(StandardScaler(), SVC(C=C, gamma=gamma))
            model.fit(X_train, y_train)
            val_dogruluk = accuracy_score(y_val, model.predict(X_val))
            print(f"C={C:<4} gamma={gamma!s:<7} doğrulama %{val_dogruluk * 100:.2f}")
            if en_iyi is None or val_dogruluk > en_iyi[0]:
                en_iyi = (val_dogruluk, C, gamma)

    _, C, gamma = en_iyi
    model = make_pipeline(StandardScaler(), SVC(C=C, gamma=gamma))
    model.fit(X_train, y_train)
    tahmin = model.predict(X_test)

    dogruluk = accuracy_score(y_test, tahmin)
    kutu_dogruluk = accuracy_score([KUTU[s] for s in y_test], [KUTU[s] for s in tahmin])
    print(f"\nSeçilen: C={C}, gamma={gamma}")
    print(f"Test doğruluğu (6 sınıf): %{dogruluk * 100:.2f}")
    print(f"Test doğruluğu (kutudaki 5 bölme): %{kutu_dogruluk * 100:.2f}")

    Path("sonuclar").mkdir(exist_ok=True)
    Path("sonuclar/klasik_yontem.json").write_text(json.dumps({
        "C": C, "gamma": gamma,
        "dogruluk_6_sinif": round(dogruluk, 4),
        "dogruluk_5_bolme": round(kutu_dogruluk, 4),
    }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
