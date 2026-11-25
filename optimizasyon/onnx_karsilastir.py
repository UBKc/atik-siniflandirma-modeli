# raspberry pi icin modeli kucultme denemesi: pytorch, onnx (fp32) ve onnx int8 (8 bit) karsilastirmasi
# int8 iki sekilde: dinamik (sadece agirliklar) ve statik (egitim goruntulerinden 100 tanesiyle kalibrasyon)
# hepsi ayni test kumesinde: dogruluk, model boyutu, tek goruntu suresi (1 cekirdek ve butun cekirdekler)
# kullanim (repo klasorunden): python optimizasyon/onnx_karsilastir.py
# not: on islemeyi kendim PIL ile yapiyorum, ultralytics ile arada 1 goruntu fark cikiyor
import json
import random
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_dynamic, quantize_static
from PIL import Image
from ultralytics import YOLO

MODEL = Path("model/atik_yolov8n_cls.pt")
CIKTI = Path("optimizasyon/modeller")
TEST = Path("veri/test")
KALIBRASYON = Path("veri/train")
BOYUT = 224


def on_isle(yol):
    # ultralytics'in siniflandirma on islemesiyle ayni: kisa kenar 224, ortadan 224x224 kirp, 0-1 arasi
    img = Image.open(yol).convert("RGB")
    w, h = img.size
    oran = BOYUT / min(w, h)
    img = img.resize((round(w * oran), round(h * oran)), Image.BILINEAR)
    w, h = img.size
    sol, ust = (w - BOYUT) // 2, (h - BOYUT) // 2
    img = img.crop((sol, ust, sol + BOYUT, ust + BOYUT))
    x = np.asarray(img, dtype=np.float32) / 255.0
    return x.transpose(2, 0, 1)[None]


class Kalibrasyon(CalibrationDataReader):
    def __init__(self, dosyalar, giris_adi):
        self.veri = iter([{giris_adi: on_isle(f)} for f in dosyalar])

    def get_next(self):
        return next(self.veri, None)


def onnx_degerlendir(yol, dosyalar, siniflar, cekirdek):
    secenek = ort.SessionOptions()
    secenek.intra_op_num_threads = cekirdek
    oturum = ort.InferenceSession(str(yol), secenek, providers=["CPUExecutionProvider"])
    giris = oturum.get_inputs()[0].name

    for f in dosyalar[:5]:  # isinma
        oturum.run(None, {giris: on_isle(f)})

    dogru, sureler = 0, []
    for f in dosyalar:
        x = on_isle(f)
        t0 = time.perf_counter()
        cikti = oturum.run(None, {giris: x})[0]
        sureler.append((time.perf_counter() - t0) * 1000)
        dogru += siniflar[int(cikti.argmax())] == f.parent.name
    return dogru / len(dosyalar), float(np.median(sureler))


def main():
    CIKTI.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(MODEL))
    siniflar = [model.names[i] for i in range(len(model.names))]
    dosyalar = sorted(TEST.glob("*/*.jpg"))

    # pytorch modeli, referans icin
    dogru = sum(siniflar[int(model(str(f), imgsz=BOYUT, verbose=False, device="cpu")[0].probs.top1)] == f.parent.name
                for f in dosyalar)
    pt_dogruluk = dogru / len(dosyalar)

    fp32 = Path(model.export(format="onnx", imgsz=BOYUT, simplify=True, opset=17))
    fp32 = fp32.rename(CIKTI / "atik_fp32.onnx")
    giris_adi = ort.InferenceSession(str(fp32), providers=["CPUExecutionProvider"]).get_inputs()[0].name

    dinamik = CIKTI / "atik_int8_dinamik.onnx"
    quantize_dynamic(str(fp32), str(dinamik), weight_type=QuantType.QInt8)

    random.seed(0)
    kal_dosyalar = random.sample(sorted(KALIBRASYON.glob("*/*.jpg")), 100)
    statik = CIKTI / "atik_int8_statik.onnx"
    quantize_static(str(fp32), str(statik), Kalibrasyon(kal_dosyalar, giris_adi),
                    quant_format=QuantFormat.QDQ, activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8,
                    per_channel=True)

    sonuclar = {"pytorch (.pt)": {"dogruluk": round(pt_dogruluk, 4), "boyut_mb": round(MODEL.stat().st_size / 1e6, 2)}}
    for ad, yol in [("onnx fp32", fp32), ("onnx int8 dinamik", dinamik), ("onnx int8 statik", statik)]:
        d1, s1 = onnx_degerlendir(yol, dosyalar, siniflar, 1)
        _, s_tum = onnx_degerlendir(yol, dosyalar, siniflar, 0)
        sonuclar[ad] = {"dogruluk": round(d1, 4), "boyut_mb": round(yol.stat().st_size / 1e6, 2),
                        "sure_ms_1_cekirdek": round(s1, 2), "sure_ms_tum_cekirdekler": round(s_tum, 2)}

    for ad, s in sonuclar.items():
        print(f"{ad:<20} dogruluk %{s['dogruluk'] * 100:.2f}  boyut {s['boyut_mb']:.2f} MB  "
              + (f"sure {s['sure_ms_1_cekirdek']:.1f} ms (1 cekirdek), {s['sure_ms_tum_cekirdekler']:.1f} ms (hepsi)"
                 if "sure_ms_1_cekirdek" in s else ""))

    with open("optimizasyon/sonuclar.json", "w", encoding="utf-8") as f:
        json.dump(sonuclar, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
