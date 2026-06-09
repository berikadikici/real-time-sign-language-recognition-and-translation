# Kurulum ve Çalıştırma Rehberi

Bu rehber, `capstone_final` projesini sıfırdan ayağa kaldırmak için adım adım yol haritasıdır.

**Sistem mimarisi özet:**
- 🇹🇷 **Türkçe** → AUTSL modeli (226 sınıf, ~%90 doğruluk)
- 🇬🇧 **İngilizce** → ASL Citizen modeli (20 sınıf, ~%68 doğruluk)
- 💬 **Cümle üretimi** → Ollama + Qwen 2.5 3B (few-shot prompt)
- 🌐 **Çeviri** → Helsinki-NLP OPUS-MT (TR↔EN)
- 📷 **Kamera** → MediaPipe Holistic + OpenCV

---

## 1. Python paketleri

```bash
cd ~/PycharmProjects/capstone_final
pip install -r requirements.txt
```

`requirements.txt` içindekiler: `numpy, opencv-python, mediapipe, torch, torchaudio, scikit-learn, transformers, sentencepiece, pillow`.

## 2. MediaPipe modelini indir (gerekirse)

```bash
curl -L -o holistic_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/holistic_landmarker/holistic_landmarker/float16/latest/holistic_landmarker.task
```

(Zaten varsa atla.)

## 3. Ollama + Qwen 2.5 kur

LLM cümle üretimi için. Türkçe gramerde başarılı olan en küçük model **Qwen 2.5 3B** (~2 GB).

```bash
# Ollama kur
brew install ollama

# Servisi başlat (arka planda kalıcı)
brew services start ollama

# Qwen 2.5 3B indir
ollama pull qwen2.5:3b
```

Test:

```bash
ollama run qwen2.5:3b "Sen kitap okumak → Sen kitap okuyorsun. Biz yemek yemek istemek → Biz yemek yemek istiyoruz. Ben spor yapmak sevmek →"
```

Beklenen çıktı: `Ben spor yapmayı seviyorum.`

> **Not:** Önceki sürümde DeepSeek-R1-Distill-Qwen 1.5B kullanılıyordu. Türkçe morfolojisinde başarısız olduğu için Qwen 2.5 3B'ye geçtik.

## 4. Datasetleri indir

### 4.1 AUTSL — Türk İşaret Dili (~50 GB)

1. https://www.kaggle.com/datasets/ngphmng/autsl-dataset
2. **Download** → ZIP indir
3. `data/autsl/` altına aç:

```bash
mkdir -p data/autsl
unzip ~/Downloads/autsl-dataset.zip -d data/autsl/
```

Yapı:
```
data/autsl/
├── train/
│   ├── signer0_sample1_color.mp4
│   └── ...
├── train.csv          (filename, label_id formatı)
├── val.csv
└── test.csv
```

### 4.2 ASL Citizen — Amerikan İşaret Dili (~87 GB)

WLASL'in yerine geçti (daha kaliteli ve kapsamlı). Microsoft Research'tan ücretsiz, kayıt gerekmiyor.

1. https://www.microsoft.com/en-us/download/details.aspx?id=105253 → Download
2. Veya direkt link:
```bash
curl -L -o ASL_Citizen.zip \
  "https://download.microsoft.com/download/b/8/8/b88c0bae-e6c1-43e1-8726-98cf5af36ca4/ASL_Citizen.zip"
```
> 💡 Tarayıcıdan indirmek genelde curl'den daha hızlı (paralel bağlantı).

3. `data/asl_citizen/` altına aç:
```bash
mkdir -p data/asl_citizen
unzip ASL_Citizen.zip -d data/asl_citizen/
```

Yapı:
```
data/asl_citizen/
├── splits/
│   ├── train.csv     (Participant ID, Video file, Gloss, ASL-LEX Code)
│   ├── val.csv
│   └── test.csv
└── videos/
    ├── <id>-GLOSS.mp4
    └── ...
```

### 4.3 (Opsiyonel) WLASL — Yedek İngilizce model

ASL Citizen'a geçtikten sonra WLASL yedek olarak duruyor. İstersen indir:
- https://www.kaggle.com/datasets/risangbaskoro/wlasl-processed
- `data/wlasl/` altına aç

## 5. Veri durumunu kontrol et

```bash
python main.py status
```

Beklenen çıktı:
```
✓ AUTSL kaynak       → .../data/autsl/train
✓ ASL_Citizen kaynak → .../data/asl_citizen/videos
✓ WLASL kaynak       → .../data/wlasl/videos    (varsa)
```

## 6. Keypoint çıkarımı (data prep)

```bash
# AUTSL (~1-2 saat, 28k video)
python main.py prep --dataset autsl --max-per 60

# ASL Citizen (top 20 sınıf, train+val+test birleşik, ~30 dk)
python main.py prep --dataset asl_citizen --top-k 20 --max-per 80
```

> **Önemli:** ASL Citizen'da `--top-k 20 --max-per 80` kombinasyonu ile en iyi sonuç alındı (%68.2). Bu kombinasyon train+val+test split'lerini otomatik birleştirir.

> Test için `--max-per 20` ile hızlı deneme yapabilirsin.

## 7. Modelleri eğit

```bash
# Türkçe modeli (~30 dk, M4 Pro)
python main.py train --dataset autsl --epochs 100

# İngilizce modeli (~10 dk)
python main.py train --dataset asl_citizen --epochs 100 --hidden 96 --batch 32 --lr 5e-4
```

Beklenen accuracy:
- **AUTSL: ~%90** (226 sınıf üzerinde)
- **ASL Citizen: ~%68** (20 sınıf üzerinde)

> İngilizce modelde `--hidden 96` overfitting'i azaltır.

## 8. AUTSL etiketlerini Türkçeleştir (KRİTİK ADIM!)

AUTSL Kaggle versiyonunda etiketler sayısal (CLASS_000, CLASS_001...). LLM'in anlamlı cümle kurabilmesi için Türkçe karşılıklara çevirmemiz lazım. 226 kelimelik haritayı kullan:

```bash
python relabel_model.py
```

Bu komut:
- `models/model_autsl.pt`'yi açar
- Etiketleri `autsl_class_map.csv`'deki Türkçe karşılıklara çevirir
- Yedek alır (`model_autsl.pt.bak`)
- Model ağırlıklarına **dokunmaz** (retrain gerekmez)

Çıktı:
```
0 → MERHABA
1 → NASILSIN
2 → IYIYIM
...
10 → BEN
11 → SEN
```

> Resmi AUTSL SignList'i ileride bulursan, `autsl_class_map.csv`'yi düzenleyip script'i tekrar çalıştır. Anlık güncellenir.

## 9. Canlı tanıma — interaktif menü

```bash
python main.py run
```

Şu menü çıkar:

```
══════════════════════════════════════════════════
  🤟  İşaret Dili Tanıma Sistemi
══════════════════════════════════════════════════
  Dil seçin / Select language:
  1) 🇹🇷 Türkçe İşaret Dili (AUTSL) ✓
  2) 🇬🇧 American Sign Language (ASL Citizen) ✓
  q) Çıkış
Seçim [1/2/q]:
```

`1` → AUTSL modeli + Türkçe cümle
`2` → ASL Citizen modeli + İngilizce cümle

**Veya direkt flag ile:**

```bash
python main.py run --lang tr     # Türkçe
python main.py run --lang en     # İngilizce
```

**Örnek tanıma akışı:**

```
🇹🇷 BEN (87%)
🇹🇷 SPOR (76%)
🇹🇷 YAPMAK (68%)
🇹🇷 SEVMEK (82%)
─────────────────────────────────
Kelimeler: BEN → SPOR → YAPMAK → SEVMEK
Cümle    : Ben spor yapmayı seviyorum.   [0.4s]
─────────────────────────────────
```

ESC veya `q` ile çıkış.

## 10. Kamera otomasyonu

Sistem **default olarak otomatik kamera bulur** (`--camera -1`). Mac/Windows farklı davranır:
- **Windows**: Local kamera index 0
- **Mac (Continuity Camera bağlı)**: Local kamera index 1, iPhone index 0

İndex'i değiştirmek için:

```bash
python main.py run --camera 1      # Spesifik index
python main.py cameras             # Mevcut kameraları listele
```

## 11. Sentence constructor'ı bağımsız test et

```bash
python main.py test-sentence --lang tr ben spor yapmak sevmek
```

Çıktı:
```
Girdi: ['ben', 'spor', 'yapmak', 'sevmek']
Çıktı: Ben spor yapmayı seviyorum.
```

İngilizce için:
```bash
python main.py test-sentence --lang en i sport do like
```

## 12. NMT — Türkçe ↔ İngilizce çeviri (opsiyonel)

Helsinki-NLP modelleri ilk kullanımda otomatik iner (~600 MB her biri). Modeller:
- **Türkçe → İngilizce**: `Helsinki-NLP/opus-mt-tc-big-tr-en`
- **İngilizce → Türkçe**: `Helsinki-NLP/opus-mt-en-tr`

İnternet bağlantısı olduğunda ilk run'da otomatik indirilir.

---

## İpuçları ve Sorun Giderme

| Sorun | Çözüm |
|-------|-------|
| Qwen yavaş çalışıyor | İlk çağrı ~5-10 sn (model RAM'a yükleniyor), sonrası 0.3-1 sn. |
| Kamera açılmıyor | Sistem Ayarları → Gizlilik → Kamera → Terminal'e izin ver. |
| Continuity Camera istemiyorum | iPhone'u kilitle veya AirPlay & Devam → Süreklilik Kamerası → Kapat |
| Confidence düşük | İşareti yavaş yap, kameraya dik dur, ışık ayarla. |
| Aynı kelime tekrar tekrar | `--threshold 0.6` ile güven eşiğini yükselt. |
| Cümle çok hızlı tamamlanıyor | `--silence 3.5` ile sessizlik süresini artır. |
| Ollama bağlantı hatası | `brew services start ollama`, `ollama list` ile model kontrolü. |
| Etiketler hâlâ CLASS_X | `python relabel_model.py` çalıştır (Adım 8). |
| ASL Citizen çok az dizi | `python main.py prep --dataset asl_citizen --top-k 20 --max-per 80` (3 CSV birleştirilir). |
| MediaPipe segmentation crash | Karelerin sabit boyuta resize edildiğini kontrol et (kodda hazır). |

---

## Final Klasör Yapısı

```
capstone_final/
├── data/
│   ├── autsl/                 ← Kaggle (AUTSL)
│   ├── asl_citizen/           ← Microsoft (ASL Citizen)
│   ├── wlasl/                 ← (opsiyonel) Kaggle (WLASL yedek)
│   └── keypoints/             ← prep'in ürettiği (.npy sekanslar)
│       ├── autsl/
│       ├── asl_citizen/
│       └── wlasl/
├── models/
│   ├── model_autsl.pt         ← Türkçe model (~%90)
│   ├── model_autsl.pt.bak     ← relabel öncesi yedek
│   ├── labels_autsl.json
│   ├── labels_autsl.json.bak
│   ├── model_asl_citizen.pt   ← İngilizce model (~%68)
│   ├── labels_asl_citizen.json
│   ├── model_wlasl.pt         ← (yedek)
│   └── labels_wlasl.json      ← (yedek)
├── holistic_landmarker.task   ← MediaPipe model
├── autsl_class_map.csv        ← 226 ID → Türkçe kelime eşlemesi
├── dataset_prep.py            ← AUTSL/WLASL/ASL Citizen preprocess
├── train.py                   ← Model eğitim
├── recognition.py             ← Canlı tanıma + gate filtreleri
├── sentence_constructor.py    ← Qwen wrapper + few-shot prompt
├── relabel_model.py           ← Etiket Türkçeleştirme
├── main.py                    ← CLI giriş noktası
├── requirements.txt
└── SETUP.md                   ← Bu dosya
```

## Pipeline Akışı (Özet)

```
Kamera → MediaPipe Holistic → 1692-d keypoint
       → 30-frame sliding window
       → per-sequence centering + z-score
       → Transformer + Bi-LSTM
       → Kelime listesi
       → Gate filtreleri (el var mı? hareket var mı? güven yeterli mi?)
       → 2.5s sessizlik tespiti
       → Qwen 2.5 3B (few-shot prompt)
       → Akıcı cümle (TR veya EN)
       → (opsiyonel) Helsinki-NLP NMT çevirisi
       → Ekran çıktısı
```

## Mevcut Performans

| Komponent | Metrik |
|-----------|--------|
| AUTSL recognition | %90 val acc (226 sınıf) |
| ASL Citizen recognition | %68 val acc (20 sınıf) |
| Model boyutu | ~10 MB (her biri) |
| Parameter sayısı | ~920k (lightweight) |
| MediaPipe latency | ~30 ms/kare |
| Model inference | ~5 ms/sekans |
| Qwen cümle üretimi | 0.3-0.6 s (warmup sonrası) |
| Toplam kullanıcı latency | ~3 saniye (2.5s bekleme dahil) |
