"""
dataset_prep.py
===============
AUTSL (Türk İşaret Dili) + WLASL (Amerikan İşaret Dili) keypoint çıkarımı.

Her video → tek etiket → temiz veri.

Kullanım:
    python main.py prep --dataset autsl
    python main.py prep --dataset wlasl --top-k 100
    python main.py prep --dataset both
    python main.py status
"""
from __future__ import annotations

import os, csv, json, argparse, time
import numpy as np
import cv2

# ── Sabitler ──────────────────────────────────────────────────────────────────
SEQ_LEN = 30
FEATURE_DIM = 1692       # pose 132 + face 1434 + lh 63 + rh 63
FACE_DIM = 1434
MIN_SAMPLES = 1
TARGET_SIZE = (480, 360) # MediaPipe için tüm kareleri bu boyuta resize et

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(BASE_DIR, "data")
KP_DIR    = os.path.join(DATA_ROOT, "keypoints")
AUTSL_DIR = os.path.join(DATA_ROOT, "autsl")
WLASL_DIR = os.path.join(DATA_ROOT, "wlasl")
ASLCITIZEN_DIR = os.path.join(DATA_ROOT, "asl_citizen")

_MP_TS = 0  # mediapipe video mode timestamp


# ── MediaPipe holistic ────────────────────────────────────────────────────────
def _make_holistic():
    from mediapipe.tasks.python import vision as mp_vision
    from mediapipe.tasks.python.core.base_options import BaseOptions

    task = os.path.join(BASE_DIR, "holistic_landmarker.task")
    if not os.path.exists(task):
        raise FileNotFoundError(
            f"holistic_landmarker.task bulunamadı.\n"
            f"İndirmek için:\n"
            f"  curl -L -o {task} "
            f"https://storage.googleapis.com/mediapipe-models/holistic_landmarker/holistic_landmarker/float16/latest/holistic_landmarker.task"
        )
    opts = mp_vision.HolisticLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=task),
        running_mode=mp_vision.RunningMode.VIDEO,
        min_face_detection_confidence=0.3,
        min_face_landmarks_confidence=0.3,
        min_pose_detection_confidence=0.3,
        min_pose_landmarks_confidence=0.3,
        min_hand_landmarks_confidence=0.3,
    )
    return mp_vision.HolisticLandmarker.create_from_options(opts)


def _extract(result) -> np.ndarray:
    pose = (np.array(
        [[lm.x, lm.y, lm.z, getattr(lm, "visibility", 0.0)] for lm in result.pose_landmarks],
        np.float32).flatten() if result.pose_landmarks else np.zeros(132, np.float32))
    face = (np.array([[lm.x, lm.y, lm.z] for lm in result.face_landmarks],
        np.float32).flatten() if result.face_landmarks else np.zeros(FACE_DIM, np.float32))
    lh = (np.array([[lm.x, lm.y, lm.z] for lm in result.left_hand_landmarks],
        np.float32).flatten() if result.left_hand_landmarks else np.zeros(63, np.float32))
    rh = (np.array([[lm.x, lm.y, lm.z] for lm in result.right_hand_landmarks],
        np.float32).flatten() if result.right_hand_landmarks else np.zeros(63, np.float32))
    return np.concatenate([pose, face, lh, rh])


def bgr_frames_to_seq(frames: list, holistic) -> np.ndarray | None:
    """BGR kare listesinden (SEQ_LEN, FEATURE_DIM) sekansı üretir."""
    import mediapipe as mp
    global _MP_TS
    kps = []
    for frm in frames[:SEQ_LEN]:
        _MP_TS += 33
        # Sabit boyuta resize — MediaPipe segmentation smoothing'in çakılmaması için
        # farklı çözünürlüklü videolar arasında geçişte sorun çıkıyor
        frm = cv2.resize(frm, TARGET_SIZE)
        rgb = cv2.cvtColor(frm, cv2.COLOR_BGR2RGB)
        img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        try:
            res = holistic.detect_for_video(img, _MP_TS)
        except (ValueError, RuntimeError):
            continue
        vec = _extract(res)
        if vec.shape[0] < FEATURE_DIM:
            vec = np.concatenate([vec, np.zeros(FEATURE_DIM - vec.shape[0], np.float32)])
        elif vec.shape[0] > FEATURE_DIM:
            vec = vec[:FEATURE_DIM]
        kps.append(vec)
    if not kps:
        return None
    arr = np.array(kps, np.float32)
    if len(arr) < SEQ_LEN:
        arr = np.concatenate([arr, np.zeros((SEQ_LEN - len(arr), FEATURE_DIM), np.float32)])
    return arr[:SEQ_LEN]


def video_to_seq(path: str, holistic) -> np.ndarray | None:
    """Video dosyasından düzgün dağıtılmış SEQ_LEN kare alıp sekansa dönüştür."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return None
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return None
    # Uniform sample
    if total >= SEQ_LEN:
        target_idx = np.linspace(0, total - 1, SEQ_LEN).astype(int)
    else:
        target_idx = np.arange(total)
    target_set = set(target_idx.tolist())
    frames = []
    fi = 0
    while True:
        ret, frm = cap.read()
        if not ret:
            break
        if fi in target_set:
            frames.append(frm)
        fi += 1
    cap.release()
    if not frames:
        return None
    return bgr_frames_to_seq(frames, holistic)


def _save(arr: np.ndarray, out_dir: str, seq_id: int):
    folder = os.path.join(out_dir, str(seq_id))
    os.makedirs(folder, exist_ok=True)
    np.save(os.path.join(folder, "sequence.npy"), arr)


# ── AUTSL ─────────────────────────────────────────────────────────────────────
class AUTSLProcessor:
    """
    Kaggle ngphmng/autsl-dataset için processor.

    Beklenen klasör yapısı (esnek, otomatik bulur):
        data/autsl/
            train/                  <- .mp4 videoları (her biri tek işaret)
            train_labels.csv        <- sample_id,label_id
            SignList_ClassId_TR_EN.csv  <- ClassId,TR,EN
    """
    def __init__(self):
        self.root = AUTSL_DIR
        self.out  = os.path.join(KP_DIR, "autsl")

    # — yol bulma —
    def _find(self, names: list[str]) -> str | None:
        for n in names:
            p = os.path.join(self.root, n)
            if os.path.exists(p):
                return p
            for sub in ("train", "Train", "data"):
                p2 = os.path.join(self.root, sub, n)
                if os.path.exists(p2):
                    return p2
        return None

    def labels_csv(self):
        return self._find([
            "train_labels.csv", "train.csv", "labels.csv",
            "Train_labels.csv", "Train.csv",
        ])

    def class_csv(self):
        return self._find([
            "SignList_ClassId_TR_EN.csv",
            "SignList.csv",
            "ClassId.csv",
            "sign_list.csv",
            "classes.csv",
            "class_map.csv",
        ])

    def video_dir(self):
        for sub in ("train", "Train", "videos", "data"):
            p = os.path.join(self.root, sub)
            if os.path.isdir(p):
                # En az bir mp4 olmalı
                for f in os.listdir(p)[:5]:
                    if f.lower().endswith(".mp4"):
                        return p
                # Alt klasör de olabilir
                for s2 in os.listdir(p):
                    pp = os.path.join(p, s2)
                    if os.path.isdir(pp):
                        for f in os.listdir(pp)[:5]:
                            if f.lower().endswith(".mp4"):
                                return p  # ana train dir'i döndür
        return None

    def available(self):
        return bool(self.labels_csv() and self.video_dir())

    def load_class_map(self) -> dict[int, str]:
        """label_id → TR sözcük."""
        cp = self.class_csv()
        result = {}
        if not cp:
            return result
        with open(cp, encoding="utf-8") as f:
            sample = f.read(2048); f.seek(0)
            delim = "\t" if "\t" in sample else ","
            reader = csv.DictReader(f, delimiter=delim)
            for row in reader:
                # Esnek kolon adları
                cid = (row.get("ClassId") or row.get("classid") or
                       row.get("class_id") or row.get("id"))
                tr  = (row.get("TR") or row.get("Turkish") or row.get("tr") or
                       row.get("label") or row.get("name"))
                if cid is None or tr is None:
                    continue
                try:
                    result[int(cid)] = str(tr).strip().upper()
                except ValueError:
                    continue
        return result

    def process(self, max_per: int = 60):
        if not self.available():
            print(f"[AUTSL] Veri bulunamadı → {self.root}")
            print(f"  labels: {self.labels_csv()}")
            print(f"  videos: {self.video_dir()}")
            return {}

        class_map = self.load_class_map()
        if not class_map:
            print("[AUTSL] UYARI: Sınıf haritası yok. Etiketler CLASS_<id> olarak kaydedilecek.")
            print("         (Türkçe kelimeler için sonra class_map.csv ekleyebilirsin)")

        labels_path = self.labels_csv()
        vdir = self.video_dir()
        print(f"[AUTSL] CSV: {labels_path}")
        print(f"[AUTSL] Videolar: {vdir}")

        # CSV'yi oku — AUTSL formatı: "filename.mp4,label_id" (header yok)
        rows = []
        with open(labels_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                # Header satırı varsa atla
                parts = line.split(",")
                if len(parts) < 2:
                    continue
                fname = parts[0].strip()
                try:
                    lid = int(parts[1].strip())
                except ValueError:
                    continue  # header satırı muhtemelen
                rows.append((fname, lid))

        if not rows:
            print("[AUTSL] CSV'den hiç satır okunamadı")
            return {}

        print(f"[AUTSL] {len(rows)} örnek bulundu")

        # Video dosyalarını cache et (hızlı arama için)
        all_videos = {}
        for root, _, files in os.walk(vdir):
            for f in files:
                if f.lower().endswith(".mp4"):
                    all_videos[f] = os.path.join(root, f)
        print(f"[AUTSL] {len(all_videos)} mp4 dosyası bulundu")

        # Toplam sınıf sayısını CSV'den hesapla (erken çıkış için)
        all_class_ids = {lid for _, lid in rows}
        target_total = len(all_class_ids) * max_per

        saved: dict[str, int] = {}
        holistic = _make_holistic()
        try:
            for i, (fname, lid) in enumerate(rows):
                label = class_map.get(lid, f"CLASS_{lid:03d}")
                if saved.get(label, 0) >= max_per:
                    continue
                # Tüm sınıflar dolduysa erken çık
                if sum(saved.values()) >= target_total and len(saved) >= len(all_class_ids):
                    print(f"  [AUTSL] Tüm sınıflar doldu ({len(saved)} sınıf × {max_per}), erken çıkış.")
                    break
                vpath = all_videos.get(fname)
                if not vpath and not fname.endswith(".mp4"):
                    vpath = all_videos.get(f"{fname}.mp4")
                if not vpath:
                    continue
                arr = video_to_seq(vpath, holistic)
                if arr is None:
                    continue
                _save(arr, os.path.join(self.out, label), saved.get(label, 0))
                saved[label] = saved.get(label, 0) + 1
                if (i + 1) % 500 == 0:
                    print(f"  [AUTSL] {i+1}/{len(rows)} işlendi, "
                          f"{sum(saved.values())} dizi, {len(saved)} sınıf")
        finally:
            holistic.close()

        print(f"[AUTSL] ✓ {sum(saved.values())} dizi, {len(saved)} sınıf")
        return saved


# ── WLASL ─────────────────────────────────────────────────────────────────────
class WLASLProcessor:
    """
    Kaggle risangbaskoro/wlasl-processed için processor.

    Beklenen klasör yapısı:
        data/wlasl/
            videos/                 <- *.mp4 (video_id.mp4)
            WLASL_v0.3.json         <- metadata
    """
    def __init__(self):
        self.root = WLASL_DIR
        self.out  = os.path.join(KP_DIR, "wlasl")

    def find_json(self) -> str | None:
        for n in ("WLASL_v0.3.json", "wlasl.json", "nslt_2000.json"):
            p = os.path.join(self.root, n)
            if os.path.exists(p):
                return p
        # Alt klasör
        for sub in os.listdir(self.root) if os.path.isdir(self.root) else []:
            sp = os.path.join(self.root, sub)
            if os.path.isdir(sp):
                for n in ("WLASL_v0.3.json", "wlasl.json"):
                    p = os.path.join(sp, n)
                    if os.path.exists(p):
                        return p
        return None

    def find_video_dir(self) -> str | None:
        for sub in ("videos", "video", "raw_videos"):
            p = os.path.join(self.root, sub)
            if os.path.isdir(p):
                return p
        # Tüm root'ta mp4 var mı?
        for f in os.listdir(self.root) if os.path.isdir(self.root) else []:
            if f.lower().endswith(".mp4"):
                return self.root
        return None

    def available(self):
        return bool(self.find_json() and self.find_video_dir())

    def process(self, top_k: int = 100, max_per: int = 60):
        if not self.available():
            print(f"[WLASL] Veri bulunamadı → {self.root}")
            print(f"  JSON: {self.find_json()}")
            print(f"  videos: {self.find_video_dir()}")
            return {}

        jp = self.find_json()
        vdir = self.find_video_dir()
        print(f"[WLASL] JSON: {jp}")
        print(f"[WLASL] Videolar: {vdir}")

        with open(jp, encoding="utf-8") as f:
            data = json.load(f)

        # En çok örneği olan top_k gloss
        glosses = sorted(data, key=lambda e: -len(e.get("instances", [])))
        if top_k > 0:
            glosses = glosses[:top_k]

        saved: dict[str, int] = {}
        holistic = _make_holistic()
        try:
            for entry in glosses:
                gloss = entry.get("gloss", "").strip().upper()
                if not gloss:
                    continue
                for inst in entry.get("instances", []):
                    if saved.get(gloss, 0) >= max_per:
                        break
                    vid = inst.get("video_id")
                    if not vid:
                        continue
                    # video bul
                    candidates = [
                        f"{vid}.mp4",
                        f"{vid:0>5}.mp4" if str(vid).isdigit() else None,
                    ]
                    vpath = None
                    for c in filter(None, candidates):
                        p = os.path.join(vdir, c)
                        if os.path.exists(p):
                            vpath = p; break
                    if not vpath:
                        # Geniş arama
                        for f in os.listdir(vdir):
                            if f.startswith(str(vid)) and f.lower().endswith(".mp4"):
                                vpath = os.path.join(vdir, f); break
                    if not vpath:
                        continue
                    arr = video_to_seq(vpath, holistic)
                    if arr is None:
                        continue
                    _save(arr, os.path.join(self.out, gloss), saved.get(gloss, 0))
                    saved[gloss] = saved.get(gloss, 0) + 1
                if len(saved) % 10 == 0:
                    print(f"  [WLASL] {len(saved)} sınıf, {sum(saved.values())} dizi")
        finally:
            holistic.close()

        print(f"[WLASL] ✓ {sum(saved.values())} dizi, {len(saved)} sınıf")
        return saved


# ── ASL Citizen ───────────────────────────────────────────────────────────────
class ASLCitizenProcessor:
    """
    Google ASL Citizen (2023) processor — 83k klip, ~2700 sınıf.

    Beklenen klasör yapısı (esnek):
        data/asl_citizen/
            splits/                    veya  ./
                train.csv              veya  *.csv
                val.csv
                test.csv
            videos/                    veya  ./videos/
                *.mp4

    CSV format (esnek):
        Participant_ID, ASL_Video_ID, Gloss, Split
        veya: gloss, video_path
        veya: filename, label
    """
    def __init__(self):
        self.root = ASLCITIZEN_DIR
        self.out = os.path.join(KP_DIR, "asl_citizen")

    def _find_csv(self) -> str | None:
        """train.csv'i bul."""
        for sub in ("splits", "split", "annotations", ""):
            base = os.path.join(self.root, sub) if sub else self.root
            if not os.path.isdir(base):
                continue
            for name in ("train.csv", "train_split.csv", "train_labels.csv"):
                p = os.path.join(base, name)
                if os.path.exists(p):
                    return p
        return None

    def _find_all_csvs(self) -> list[str]:
        """train + val + test CSV'lerinin hepsini bul (daha çok veri için)."""
        found = []
        for sub in ("splits", "split", "annotations", ""):
            base = os.path.join(self.root, sub) if sub else self.root
            if not os.path.isdir(base):
                continue
            for name in ("train.csv", "val.csv", "test.csv",
                         "validation.csv", "dev.csv"):
                p = os.path.join(base, name)
                if os.path.exists(p):
                    found.append(p)
        return found

    def _find_video_dir(self) -> str | None:
        for sub in ("videos", "video", "raw_videos"):
            p = os.path.join(self.root, sub)
            if os.path.isdir(p):
                return p
        return None

    def available(self):
        return bool(self._find_csv() and self._find_video_dir())

    def process(self, top_k: int = 0, max_per: int = 60,
                use_all_splits: bool = True):
        if not self.available():
            print(f"[ASL_Citizen] Veri bulunamadı → {self.root}")
            print(f"  CSV   : {self._find_csv()}")
            print(f"  videos: {self._find_video_dir()}")
            return {}

        # Tüm split CSV'lerini birleştir (daha çok veri için)
        if use_all_splits:
            csv_paths = self._find_all_csvs()
            print(f"[ASL_Citizen] {len(csv_paths)} CSV bulundu: {[os.path.basename(p) for p in csv_paths]}")
        else:
            csv_paths = [self._find_csv()]

        vdir = self._find_video_dir()
        print(f"[ASL_Citizen] Videolar: {vdir}")

        # CSV'leri oku (header'lı, esnek kolon adları)
        rows: list[tuple[str, str]] = []  # (filename, gloss)
        for csv_path in csv_paths:
            with open(csv_path, encoding="utf-8") as f:
                reader = csv.DictReader(f)
                print(f"[ASL_Citizen] {os.path.basename(csv_path)} kolonları: {reader.fieldnames}")
                for row in reader:
                    # Esnek kolon adları (ASL Citizen: "Video file", "Gloss")
                    vid = (row.get("Video file") or row.get("video_file")
                           or row.get("ASL_Video_ID") or row.get("asl_video_id")
                           or row.get("Video_ID") or row.get("video_id")
                           or row.get("filename") or row.get("video_path") or "")
                    gloss = (row.get("Gloss") or row.get("gloss")
                             or row.get("label") or row.get("word") or "")
                    if not vid or not gloss:
                        continue
                    vid = vid.strip()
                    gloss = gloss.strip().upper().replace(" ", "_")
                    if not vid.endswith(".mp4"):
                        vid = f"{vid}.mp4"
                    rows.append((vid, gloss))

        if not rows:
            print("[ASL_Citizen] CSV'den hiç satır okunamadı")
            return {}

        # Sınıf frekansını hesapla, top_k'da filtrele
        from collections import Counter
        freq = Counter(g for _, g in rows)
        if top_k > 0:
            top_classes = set(g for g, _ in freq.most_common(top_k))
            rows = [(v, g) for v, g in rows if g in top_classes]

        print(f"[ASL_Citizen] {len(rows)} klip, {len(set(g for _, g in rows))} sınıf")

        # Video cache
        all_videos = {}
        for root, _, files in os.walk(vdir):
            for f in files:
                if f.lower().endswith(".mp4"):
                    all_videos[f] = os.path.join(root, f)
        print(f"[ASL_Citizen] {len(all_videos)} mp4 bulundu")

        saved: dict[str, int] = {}
        holistic = _make_holistic()
        try:
            for i, (vid, gloss) in enumerate(rows):
                if saved.get(gloss, 0) >= max_per:
                    continue
                vpath = all_videos.get(vid) or all_videos.get(os.path.basename(vid))
                if not vpath:
                    continue
                arr = video_to_seq(vpath, holistic)
                if arr is None:
                    continue
                _save(arr, os.path.join(self.out, gloss), saved.get(gloss, 0))
                saved[gloss] = saved.get(gloss, 0) + 1
                if (i + 1) % 500 == 0:
                    print(f"  [ASL_Citizen] {i+1}/{len(rows)} işlendi, "
                          f"{sum(saved.values())} dizi, {len(saved)} sınıf")
        finally:
            holistic.close()

        print(f"[ASL_Citizen] ✓ {sum(saved.values())} dizi, {len(saved)} sınıf")
        return saved


# ── Yardımcılar ───────────────────────────────────────────────────────────────
def build_label_list(kp_dir: str = KP_DIR, dataset: str | None = None,
                     min_samples: int = MIN_SAMPLES) -> list[str]:
    """Verilen dataset (veya tümü) için etiket listesi."""
    labels = []
    if not os.path.isdir(kp_dir):
        return labels
    datasets = [dataset] if dataset else os.listdir(kp_dir)
    for ds in datasets:
        dd = os.path.join(kp_dir, ds)
        if not os.path.isdir(dd):
            continue
        for g in os.listdir(dd):
            gd = os.path.join(dd, g)
            if not os.path.isdir(gd):
                continue
            n = sum(1 for s in os.listdir(gd)
                    if os.path.exists(os.path.join(gd, s, "sequence.npy")))
            if n >= min_samples:
                labels.append(g)
    return sorted(set(labels))


def status():
    print("\n" + "═" * 46)
    print("  Veri Seti Durum Raporu")
    print("═" * 46)

    autsl = AUTSLProcessor()
    if autsl.available():
        print(f"✓  AUTSL kaynak  → {autsl.video_dir()}")
    else:
        print(f"✗  AUTSL kaynak  → {AUTSL_DIR}")
        print(f"   labels: {autsl.labels_csv()}")
        print(f"   videos: {autsl.video_dir()}")

    wlasl = WLASLProcessor()
    if wlasl.available():
        print(f"✓  WLASL kaynak  → {wlasl.find_video_dir()}")
    else:
        print(f"✗  WLASL kaynak  → {WLASL_DIR}")

    asl_cit = ASLCitizenProcessor()
    if asl_cit.available():
        print(f"✓  ASL_Citizen kaynak  → {asl_cit._find_video_dir()}")
    else:
        print(f"✗  ASL_Citizen kaynak  → {ASLCITIZEN_DIR}")
        print(f"   CSV   : {asl_cit._find_csv()}")
        print(f"   videos: {asl_cit._find_video_dir()}")

    autsl_lbl = build_label_list(dataset="autsl")
    wlasl_lbl = build_label_list(dataset="wlasl")
    asl_cit_lbl = build_label_list(dataset="asl_citizen")
    print(f"\n✓  İşlenmiş AUTSL keypoint → {len(autsl_lbl)} sınıf")
    if autsl_lbl:
        print(f"   Örnek: {autsl_lbl[:6]}")
    print(f"✓  İşlenmiş WLASL keypoint → {len(wlasl_lbl)} sınıf")
    if wlasl_lbl:
        print(f"   Örnek: {wlasl_lbl[:6]}")
    print(f"✓  İşlenmiş ASL_Citizen keypoint → {len(asl_cit_lbl)} sınıf")
    if asl_cit_lbl:
        print(f"   Örnek: {asl_cit_lbl[:6]}")
    print("═" * 46 + "\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["autsl", "wlasl", "both", "status"], default="status")
    p.add_argument("--top-k", type=int, default=0)
    p.add_argument("--max-per", type=int, default=60)
    args = p.parse_args()

    os.makedirs(KP_DIR, exist_ok=True)

    if args.dataset == "status":
        status(); return
    if args.dataset in ("autsl", "both"):
        AUTSLProcessor().process(args.max_per)
    if args.dataset in ("wlasl", "both"):
        WLASLProcessor().process(args.top_k or 100, args.max_per)
    status()


if __name__ == "__main__":
    main()
