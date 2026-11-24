"""Eğitim sırasında kaydedilen results.csv dosyasından kayıp ve doğruluk grafiği çizer.

Kullanım:
    python egitim_grafigi.py runs/classify/calismalar/yolov8n_cls/results.csv
"""
import csv
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    yol = sys.argv[1] if len(sys.argv) > 1 else "sonuclar/egitim_sonuclari.csv"
    satirlar = list(csv.DictReader(open(yol, encoding="utf-8")))
    epoch = [int(s["epoch"]) for s in satirlar]

    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    ax[0].plot(epoch, [float(s["train/loss"]) for s in satirlar], label="eğitim")
    ax[0].plot(epoch, [float(s["val/loss"]) for s in satirlar], label="doğrulama")
    ax[0].set_title("Kayıp (loss)")
    ax[0].set_xlabel("Epoch")
    ax[0].legend()

    ax[1].plot(epoch, [float(s["metrics/accuracy_top1"]) * 100 for s in satirlar])
    ax[1].set_title("Doğrulama doğruluğu (%)")
    ax[1].set_xlabel("Epoch")

    fig.tight_layout()
    fig.savefig("sonuclar/egitim_grafigi.png", dpi=110)


if __name__ == "__main__":
    main()
