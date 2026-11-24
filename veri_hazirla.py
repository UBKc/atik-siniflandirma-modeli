"""TrashNet veri setini eğitim / doğrulama / test olarak ayırır.

Her sınıf kendi içinde %70 / %15 / %15 bölünüyor (stratified), böylece az örneği
olan "çöp" sınıfı da üç kümede aynı oranda bulunuyor. Seed sabit, bölme her
çalıştırmada aynı çıkıyor.

Kullanım:
    python veri_hazirla.py ham/dataset-resized
"""
import random
import shutil
import sys
from pathlib import Path

# TrashNet klasör adlarının Türkçe karşılıkları
SINIF_ADLARI = {
    "cardboard": "karton",
    "glass": "cam",
    "metal": "metal",
    "paper": "kagit",
    "plastic": "plastik",
    "trash": "cop",
}

ORANLAR = {"train": 0.70, "val": 0.15, "test": 0.15}
SEED = 42


def main():
    kaynak = Path(sys.argv[1] if len(sys.argv) > 1 else "ham/dataset-resized")
    hedef = Path("veri")

    if hedef.exists():
        shutil.rmtree(hedef)

    rastgele = random.Random(SEED)
    print(f"{'sınıf':<10}{'train':>7}{'val':>6}{'test':>6}")

    for ing_ad, tr_ad in SINIF_ADLARI.items():
        dosyalar = sorted((kaynak / ing_ad).glob("*.jpg"))
        rastgele.shuffle(dosyalar)

        n = len(dosyalar)
        n_train = round(n * ORANLAR["train"])
        n_val = round(n * ORANLAR["val"])
        bolumler = {
            "train": dosyalar[:n_train],
            "val": dosyalar[n_train:n_train + n_val],
            "test": dosyalar[n_train + n_val:],
        }

        for bolum, liste in bolumler.items():
            klasor = hedef / bolum / tr_ad
            klasor.mkdir(parents=True, exist_ok=True)
            for dosya in liste:
                shutil.copy(dosya, klasor / dosya.name)

        print(f"{tr_ad:<10}{len(bolumler['train']):>7}{len(bolumler['val']):>6}{len(bolumler['test']):>6}")


if __name__ == "__main__":
    main()
