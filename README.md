# 🤟 İşaret Dili Tanıma ve Çeviri Sistemi

Gerçek zamanlı, on-device, lightweight bir işaret dili tanıma + akıcı cümle üretimi + çoklu dil çevirisi sistemi.

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## ✨ Özellikler

- 🇹🇷 **Türk İşaret Dili (AUTSL)** — 226 sınıf üzerinde **%90 doğruluk**
- 🇬🇧 **Amerikan İşaret Dili (ASL Citizen)** — 20 sınıf üzerinde **%68 doğruluk**
- 💬 **On-device LLM** — Qwen 2.5 3B ile akıcı Türkçe/İngilizce cümle üretimi (few-shot prompting)
- 🌐 **Çoklu dil çevirisi** — Helsinki-NLP OPUS-MT ile TR↔EN
- ⚡ **Gerçek zamanlı** — ~40 ms/kare inference, MediaPipe Holistic ile keypoint çıkarımı
- 🪶 **Lightweight** — Model checkpoint ~10 MB, parametre ~920k
- 🔒 **Bulut bağımsız** — Tüm model inference local çalışır

## 🏗️ Mimari

```
Kamera → MediaPipe Holistic → 1692-d keypoint
       → 30-frame sliding window
       → per-sequence centering + z-score normalization
       → Transformer + Bi-LSTM hybrid
       → Word recognition
       → Gate filtreleri (el var mı? hareket var mı? güven yeterli mi?)
       → 2.5s sessizlik tespiti
       → Qwen 2.5 3B (few-shot prompt)
       → Akıcı cümle
       → (opsiyonel) Helsinki-NLP NMT çevirisi
       → Ekran çıktısı
```

## 📊 Performans Karşılaştırması

| Dataset | Sınıf Sayısı | Sınıf Başına Örnek | Val Accuracy |
|---------|---|---|---|
| **AUTSL** (Türkçe) | 226 | ~170 | **%90** |
| **ASL Citizen** (İngilizce) | 20 | ~30 | **%68** |
| WLASL (yedek) | 30 | ~10 | %39 |
| PHOENIX-2014-T (terkedilmiş) | 100 | varies | %28.8 |

## 🚀 Hızlı Başlangıç

Detaylı kurulum için [SETUP.md](SETUP.md) dosyasına bak.

```bash
# 1. Bağımlılıkları kur
pip install -r requirements.txt
brew install ollama && brew services start ollama
ollama pull qwen2.5:3b

# 2. MediaPipe modelini indir
curl -L -o holistic_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/holistic_landmarker/holistic_landmarker/float16/latest/holistic_landmarker.task

# 3. Datasetleri indir (büyük dosyalar)
# AUTSL: https://www.kaggle.com/datasets/ngphmng/autsl-dataset
# ASL Citizen: https://www.microsoft.com/en-us/download/details.aspx?id=105253

# 4. Veri prep + eğitim
python main.py prep --dataset autsl --max-per 60
python main.py prep --dataset asl_citizen --top-k 20 --max-per 80
python main.py train --dataset autsl --epochs 100
python main.py train --dataset asl_citizen --epochs 100 --hidden 96

# 5. Türkçe etiketleri haritala
python relabel_model.py

# 6. Canlı tanıma
python main.py run
```

## 📁 Proje Yapısı

```
capstone_final/
├── data/                       ← Datasetler (gitignore'da)
│   ├── autsl/
│   ├── asl_citizen/
│   └── keypoints/              ← Prep çıktıları
├── models/                     ← Eğitilmiş modeller (.pt + .json)
│   ├── model_autsl.pt
│   ├── model_asl_citizen.pt
│   └── labels_*.json
├── dataset_prep.py             ← Veri ön-işleme
├── train.py                    ← Model eğitimi
├── recognition.py              ← Gerçek zamanlı tanıma
├── sentence_constructor.py     ← Qwen + few-shot prompt
├── relabel_model.py            ← Etiket Türkçeleştirme
├── main.py                     ← CLI giriş noktası
├── autsl_class_map.csv         ← 226 → Türkçe kelime haritası
├── holistic_landmarker.task    ← MediaPipe model
├── requirements.txt
├── SETUP.md                    ← Detaylı kurulum
└── README.md
```

## 🧪 Teknoloji Stack'i

- **Python 3.12** + **PyTorch** (Apple MPS / CUDA / CPU)
- **MediaPipe Tasks** (Holistic Landmarker, float16)
- **OpenCV 4** + **Pillow** (Unicode overlay)
- **scikit-learn** (stratified split)
- **Ollama** + **Qwen 2.5 3B** (cümle inşası)
- **Helsinki-NLP OPUS-MT** (NMT)

## 📚 Datasetler

| Dataset | Kaynak | Boyut | Lisans |
|---------|--------|-------|--------|
| AUTSL | [Kaggle](https://www.kaggle.com/datasets/ngphmng/autsl-dataset) | ~50 GB | CC BY-NC 4.0 |
| ASL Citizen | [Microsoft](https://www.microsoft.com/en-us/download/details.aspx?id=105253) | ~87 GB | Research-only |
| WLASL (yedek) | [Kaggle](https://www.kaggle.com/datasets/risangbaskoro/wlasl-processed) | ~6 GB | Research-only |

> **Not:** Datasetler GitHub repo'ya dahil değil (büyük boyut nedeniyle). Yukarıdaki linklerden indirin.

## 🎯 Karşılaşılan Sorunlar ve Çözümler

1. **PHOENIX çoklu etiket sorunu** → İzole kelime dataset'lerine geçiş (AUTSL + WLASL → ASL Citizen)
2. **Sınıf dengesizliği** → 2 ayrı model (Türkçe + İngilizce) + dil seçim menüsü
3. **Tutor noise (aynı sunucu yüzü)** → Per-sequence centering
4. **MediaPipe segmentation crash** → Sabit 480×360 resize
5. **DeepSeek Türkçe gramer hataları** → Qwen 2.5 3B + few-shot prompting
6. **WLASL veri kıtlığı** → ASL Citizen entegrasyonu (+train+val+test birleştirme)
7. **False positive bombardımanı** → 3 katmanlı gate filtre (el, hareket, güven)
8. **Mac Continuity Camera** → Otomatik kamera tespiti
9. **LLM kamera donduruyordu** → Threading worker

## 👥 Takım

| İsim | Bölüm |
|------|-------|
| Aleyna Can | Yapay Zeka Mühendisliği |
| Elanur Yoloğlu | Yapay Zeka Mühendisliği |
| Berika Dikici | Yapay Zeka Mühendisliği |
| Ali Kaan Özdemir | Bilgisayar Mühendisliği |
| Naki Erim Özer | Bilgisayar Mühendisliği |

**Danışmanlar:**
- Asst. Prof. Arezoo Sadeghzadeh
- Asst. Prof. Fatih Kahraman
- Asst. Prof. Tarkan Aydın

## 📄 Lisans

MIT License — Detaylar için [LICENSE](LICENSE) dosyasına bakın.

## 📖 Atıf

Bu projeyi akademik çalışmalarınızda kullanırsanız, lütfen şu şekilde atıf yapın:

```bibtex
@misc{capstone2026signlang,
  title={İşaret Dili Tanıma ve Çeviri Sistemi - Capstone 2026},
  author={Can, Aleyna and Dikici, Berika and Yoloğlu, Elanur and Özdemir, Ali Kaan and Özer, Naki Erim},
  year={2026},
  publisher={GitHub},
  howpublished={\url{https://github.com/USERNAME/REPO}}
}
```

## 🙏 Teşekkürler

- **AUTSL** — Sincan & Keles, IEEE Access 2020
- **ASL Citizen** — Microsoft Research, 2023
- **WLASL** — Dongxu Li et al., WACV 2020
- **MediaPipe** — Google
- **Qwen 2.5** — Alibaba Cloud
- **Helsinki-NLP** — University of Helsinki
