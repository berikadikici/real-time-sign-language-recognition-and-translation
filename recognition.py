"""
recognition.py
==============
Gerçek zamanlı, cümle bazlı işaret dili tanıma motoru.

Pipeline:
  Kamera → MediaPipe Holistic → 30-kare buffer → Model → Kelime
  CTC + momentum filtresi → temiz kelime listesi → SentenceConstructor

Kullanım:
    rec = SignRecognizer.from_checkpoint("models/model_wlasl.pt")
    rec.run_realtime(camera_index=-1)   # otomatik kamera
"""
from __future__ import annotations
import os, time, urllib.request, threading
import numpy as np
import cv2

import mediapipe as mp
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.core.base_options import BaseOptions

# Unicode metin desteği (Türkçe karakterler için)
try:
    from PIL import Image, ImageDraw, ImageFont
    _PIL_OK = True
except ImportError:
    _PIL_OK = False


BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
_TASK_URL  = ("https://storage.googleapis.com/mediapipe-models/"
              "holistic_landmarker/holistic_landmarker/float16/latest/"
              "holistic_landmarker.task")
_TASK_FILE = os.path.join(BASE_DIR, "holistic_landmarker.task")

SEQ_LEN     = 30
FEATURE_DIM = 1692
FACE_DIM    = 1434
TARGET_SIZE = (480, 360)  # MediaPipe için sabit resize

# Feature index aralıkları
POSE_END = 132
FACE_END = POSE_END + FACE_DIM     # 1566
LH_END   = FACE_END + 63           # 1629
RH_END   = LH_END + 63             # 1692


def _has_hands(kp: np.ndarray, thresh: float = 0.05) -> bool:
    """Sekansta el landmarks var mı? Sıfır vektör değilse el var."""
    lh = kp[FACE_END:LH_END]
    rh = kp[LH_END:RH_END]
    return (np.abs(lh).sum() > thresh) or (np.abs(rh).sum() > thresh)


def _has_motion(buf: list[np.ndarray], min_std: float = 0.004) -> bool:
    """Buffer'da yeterli hareket var mı? El koordinatlarının zaman varyansına bak."""
    if len(buf) < 10:
        return False
    arr = np.array(buf)  # (T, F)
    # Sadece el koordinatları için varyans
    hands = arr[:, FACE_END:RH_END]
    return hands.std(axis=0).mean() > min_std


# ── MediaPipe ─────────────────────────────────────────────────────────────────
def _ensure_task():
    if os.path.exists(_TASK_FILE):
        return
    print("holistic_landmarker.task indiriliyor (~12 MB)...")
    req = urllib.request.Request(_TASK_URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as r:
        data = r.read()
    with open(_TASK_FILE, "wb") as f:
        f.write(data)
    print("İndirme tamamlandı.")


def _make_holistic(mode=mp_vision.RunningMode.VIDEO):
    _ensure_task()
    opts = mp_vision.HolisticLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=_TASK_FILE),
        running_mode=mode,
        min_face_detection_confidence=0.5,
        min_face_landmarks_confidence=0.5,
        min_pose_detection_confidence=0.5,
        min_pose_landmarks_confidence=0.5,
        min_hand_landmarks_confidence=0.5,
    )
    return mp_vision.HolisticLandmarker.create_from_options(opts)


def extract_keypoints(result) -> np.ndarray:
    pose = (np.array([[lm.x, lm.y, lm.z, getattr(lm, "visibility", 0.0)]
                      for lm in result.pose_landmarks], np.float32).flatten()
            if result.pose_landmarks else np.zeros(132, np.float32))
    face = (np.array([[lm.x, lm.y, lm.z] for lm in result.face_landmarks],
                     np.float32).flatten()
            if result.face_landmarks else np.zeros(FACE_DIM, np.float32))
    lh = (np.array([[lm.x, lm.y, lm.z] for lm in result.left_hand_landmarks],
                   np.float32).flatten()
          if result.left_hand_landmarks else np.zeros(63, np.float32))
    rh = (np.array([[lm.x, lm.y, lm.z] for lm in result.right_hand_landmarks],
                   np.float32).flatten()
          if result.right_hand_landmarks else np.zeros(63, np.float32))
    vec = np.concatenate([pose, face, lh, rh])
    if vec.shape[0] < FEATURE_DIM:
        vec = np.concatenate([vec, np.zeros(FEATURE_DIM - vec.shape[0], np.float32)])
    elif vec.shape[0] > FEATURE_DIM:
        vec = vec[:FEATURE_DIM]
    return vec


# ── Görselleştirme ────────────────────────────────────────────────────────────
def draw_landmarks(frame: np.ndarray, result) -> None:
    h, w = frame.shape[:2]

    def pts(lms):
        return [(int(lm.x * w), int(lm.y * h)) for lm in lms] if lms else []

    def lines(pts_list, conns, color):
        for a, b in conns:
            if a < len(pts_list) and b < len(pts_list):
                cv2.line(frame, pts_list[a], pts_list[b], color, 1)

    def dots(pts_list, color, r=2):
        for p in pts_list:
            cv2.circle(frame, p, r, color, -1)

    POSE_C = [(11,12),(11,13),(13,15),(12,14),(14,16),
              (11,23),(12,24),(23,24),(23,25),(24,26),(25,27),(26,28)]
    HAND_C = [(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),
              (0,9),(9,10),(10,11),(11,12),(0,13),(13,14),(14,15),(15,16),
              (0,17),(17,18),(18,19),(19,20),(5,9),(9,13),(13,17)]

    face_p = pts(result.face_landmarks)
    pose_p = pts(result.pose_landmarks)
    lh_p   = pts(result.left_hand_landmarks)
    rh_p   = pts(result.right_hand_landmarks)

    dots(face_p, (180,180,255), r=1)
    lines(pose_p, POSE_C, (80,200,80));  dots(pose_p, (0,255,100))
    lines(lh_p,   HAND_C, (255,150,50)); dots(lh_p,   (255,200,100))
    lines(rh_p,   HAND_C, (50,150,255)); dots(rh_p,   (100,200,255))


# ── Kamera ────────────────────────────────────────────────────────────────────
def list_cameras(n: int = 4) -> list[int]:
    out = []
    for i in range(n):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ok, _ = cap.read()
            if ok:
                out.append(i)
            cap.release()
    return out


def _find_camera() -> int:
    cams = list_cameras()
    if not cams:
        _cam_help()
        raise RuntimeError("Hiç kamera bulunamadı.")
    if len(cams) > 1:
        print(f"Birden fazla kamera bulundu: {cams} → index={cams[0]} kullanılıyor.")
        print("Değiştirmek için: --camera <index>")
    return cams[0]


def _cam_help():
    print("\n[KAMERA HATASI] macOS için:")
    print("  Sistem Ayarları → Gizlilik ve Güvenlik → Kamera → Terminal'e izin ver")
    print("  iPhone Continuity'yi devre dışı bırakmak için iPhone'u kilitle veya")
    print("  Ayarlar → Genel → AirPlay ve Devam → Süreklilik Kamerası → Kapat")


def _open_camera(index: int):
    """macOS AVFoundation backend'i deneyip, başarısızsa default'a düş."""
    cap = cv2.VideoCapture(index, cv2.CAP_AVFOUNDATION)
    if cap.isOpened():
        ok, _ = cap.read()
        if ok:
            return cap
        cap.release()
    cap = cv2.VideoCapture(index)
    return cap


# ── Unicode metin yardımcısı ──────────────────────────────────────────────────
_FONT_PATHS = [
    "/System/Library/Fonts/Helvetica.ttc",     # macOS
    "/System/Library/Fonts/HelveticaNeue.ttc",
    "/Library/Fonts/Arial.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",  # Linux
    "C:/Windows/Fonts/arial.ttf",                       # Windows
]
_font_cache: dict[int, "ImageFont.FreeTypeFont"] = {}


def _get_font(size: int):
    if not _PIL_OK:
        return None
    if size in _font_cache:
        return _font_cache[size]
    for p in _FONT_PATHS:
        if os.path.exists(p):
            try:
                f = ImageFont.truetype(p, size)
                _font_cache[size] = f
                return f
            except Exception:
                continue
    return ImageFont.load_default()


def draw_text_unicode(frame: np.ndarray, text: str, xy: tuple[int, int],
                      font_size: int = 22, color: tuple[int, int, int] = (255, 230, 100)):
    """OpenCV frame'e Unicode (Türkçe) metin yaz. color BGR."""
    if not _PIL_OK or not text:
        cv2.putText(frame, text, xy, cv2.FONT_HERSHEY_SIMPLEX,
                    font_size / 30, color, 2)
        return frame
    pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)
    font = _get_font(font_size)
    # BGR → RGB
    draw.text(xy, text, fill=(color[2], color[1], color[0]), font=font)
    arr = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    np.copyto(frame, arr)
    return frame


def _wrap_text(text: str, max_chars: int) -> list[str]:
    """Uzun metni satırlara böl."""
    words = text.split()
    lines, cur = [], ""
    for w in words:
        if len(cur) + len(w) + 1 <= max_chars:
            cur = (cur + " " + w).strip()
        else:
            if cur:
                lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


# ── Standardization (training ile aynı) ───────────────────────────────────────
def per_seq_center(seq: np.ndarray) -> np.ndarray:
    s = seq.copy()
    nz = np.abs(s).sum(axis=1) > 1e-6
    if nz.sum() > 0:
        s[nz] = s[nz] - s[nz].mean(axis=0, keepdims=True)
    return s


# ── Ana sınıf ─────────────────────────────────────────────────────────────────
class SignRecognizer:
    """
    Sliding window + momentum filter + CTC-style deduplication ile
    cümle bazlı sürekli işaret tanıma.
    """

    # Ayarlar (override edilebilir)
    CONF_THRESH = 0.35   # bu eşiğin altındaki tahminler "blank" sayılır
    STEP        = 5      # kaç karede bir tahmin yapılacak
    MOMENTUM    = 2      # aynı tahmin kaç kere üst üste gelirse emit edilir
    PAUSE_SEC   = 2.5    # bu kadar saniye sessizlikten sonra cümle biter
    MIN_HAND_RATIO = 0.6 # son frame'lerin %X'inde el görünmeli
    MIN_MOTION  = 0.004  # el hareketi varyansı bu eşikten büyük olmalı

    def __init__(self, model, labels, input_size, hidden_size, n_classes,
                 sequence_length, n_layers, feat_mean, feat_std, device):
        self.model = model.eval()
        self.labels = labels
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_classes = n_classes
        self.sequence_length = sequence_length
        self.n_layers = n_layers
        self.feat_mean = feat_mean
        self.feat_std = feat_std
        self.device = device

        # Runtime state
        self._buf: list[np.ndarray] = []
        self._step_cnt = 0
        self._pending: str | None = None
        self._streak = 0
        self._last_emit: str | None = None
        self._last_sign_t = 0.0
        self._sentence: list[str] = []

        # LLM çıktısı için
        self._constructed_text: str = ""
        self._constructed_t: float = 0.0
        self._constructing: bool = False
        self._lock = threading.Lock()

    def set_constructed_sentence(self, text: str, status: str = "done"):
        """LLM'den dönen cümleyi UI'da göstermek için kaydet.
        status: 'thinking' (yapıyor), 'done' (bitti), 'error' (hata)."""
        with self._lock:
            self._constructed_text = text
            self._constructed_t = time.time()
            self._constructing = (status == "thinking")

    @classmethod
    def from_checkpoint(cls, path: str) -> "SignRecognizer":
        import torch
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        from train import build_model
        device = (torch.device("mps") if torch.backends.mps.is_available()
                  else torch.device("cuda") if torch.cuda.is_available()
                  else torch.device("cpu"))
        model = build_model(
            ckpt["input_size"], ckpt["hidden_size"], ckpt["n_classes"],
            ckpt.get("n_layers", 3),
        )
        model.load_state_dict(ckpt["state_dict"])
        model.to(device)
        print(f"Model yüklendi: {path} ({device}) — {len(ckpt['labels'])} sınıf")
        return cls(
            model=model,
            labels=ckpt["labels"],
            input_size=ckpt["input_size"],
            hidden_size=ckpt["hidden_size"],
            n_classes=ckpt["n_classes"],
            sequence_length=ckpt.get("sequence_length", 30),
            n_layers=ckpt.get("n_layers", 3),
            feat_mean=ckpt.get("feat_mean"),
            feat_std=ckpt.get("feat_std"),
            device=device,
        )

    # ── Inference ─────────────────────────────────────────────────────────
    def _infer(self, buf: list[np.ndarray]) -> tuple[str, float]:
        import torch
        arr = np.stack(buf).astype(np.float32)
        arr = per_seq_center(arr)
        if self.feat_mean is not None and self.feat_std is not None:
            zm = np.abs(arr).sum(axis=-1, keepdims=True) <= 1e-6
            arr = (arr - self.feat_mean) / self.feat_std
            arr[np.broadcast_to(zm, arr.shape)] = 0.0
        with torch.no_grad():
            x = torch.from_numpy(arr).unsqueeze(0).to(self.device)
            logits = self.model(x)[0]
            probs = torch.softmax(logits, dim=0).cpu().numpy()
        idx = int(np.argmax(probs))
        return self.labels[idx], float(probs[idx])

    def process_frame(self, kp: np.ndarray) -> tuple[str | None, float]:
        """Bir kareyi işle, gerekirse yeni kelime emit et."""
        self._buf.append(kp)
        if len(self._buf) > self.sequence_length:
            self._buf.pop(0)

        self._step_cnt += 1
        if self._step_cnt < self.STEP:
            return None, 0.0
        self._step_cnt = 0

        if len(self._buf) < self.sequence_length:
            return None, 0.0

        # ── GATE 1: El görünüyor mu? ──
        recent = self._buf[-15:]
        hand_frames = sum(1 for k in recent if _has_hands(k))
        if hand_frames < len(recent) * self.MIN_HAND_RATIO:
            self._pending = None
            self._streak = 0
            return None, 0.0

        # ── GATE 2: Yeterli hareket var mı? ──
        if not _has_motion(self._buf, self.MIN_MOTION):
            self._pending = None
            self._streak = 0
            return None, 0.0

        label, conf = self._infer(self._buf)

        # ── GATE 3: Güven eşiği ──
        if conf < self.CONF_THRESH:
            self._pending = None
            self._streak = 0
            return None, conf

        # Momentum filtresi
        if label == self._pending:
            self._streak += 1
        else:
            self._pending = label
            self._streak = 1

        if self._streak >= self.MOMENTUM:
            # CTC: son emit ile aynıysa atla, sadece zaman güncelle
            if label != self._last_emit:
                self._last_emit = label
                self._last_sign_t = time.time()
                self._sentence.append(label)
                if len(self._sentence) > 15:
                    self._sentence = self._sentence[-15:]
                return label, conf
            self._last_sign_t = time.time()

        return None, conf

    def sentence_complete(self) -> bool:
        if not self._sentence:
            return False
        return (time.time() - self._last_sign_t) >= self.PAUSE_SEC

    def flush_sentence(self) -> list[str]:
        s = list(self._sentence)
        self._sentence.clear()
        self._last_emit = None
        self._pending = None
        self._streak = 0
        return s

    # ── UI ────────────────────────────────────────────────────────────────
    def _draw_ui(self, frame: np.ndarray, last_conf: float):
        h, w = frame.shape[:2]

        # Üst şerit — algılanan kelimeler
        cv2.rectangle(frame, (0, 0), (w, 60), (15, 15, 30), -1)
        cv2.putText(frame, "Algilanan Kelimeler:", (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1)
        sent_str = " · ".join(self._sentence[-10:]) if self._sentence else "—"
        cv2.putText(frame, sent_str, (10, 48),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 255, 180), 2)

        # ── ALT PANEL: LLM Çıktısı ──
        with self._lock:
            text = self._constructed_text
            is_thinking = self._constructing

        panel_h = 100
        # Yarı şeffaf koyu arkaplan
        overlay = frame[h - panel_h:h, 0:w].copy()
        cv2.rectangle(overlay, (0, 0), (w, panel_h), (10, 10, 25), -1)
        cv2.addWeighted(overlay, 0.78, frame[h - panel_h:h, 0:w], 0.22, 0,
                        frame[h - panel_h:h, 0:w])

        # Üst kenar çizgisi
        cv2.line(frame, (0, h - panel_h), (w, h - panel_h), (60, 150, 220), 2)

        # Etiket
        label = "LLM Cumlesi" if not is_thinking else "LLM Dusunuyor..."
        label_color = (100, 200, 255) if not is_thinking else (100, 200, 100)
        cv2.putText(frame, label, (10, h - panel_h + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, label_color, 1)

        # Cümle metni (Unicode destekli)
        if text:
            max_chars = max(20, w // 14)  # ekran genişliğine göre
            lines = _wrap_text(text, max_chars)
            y_off = h - panel_h + 50
            for i, line in enumerate(lines[:2]):  # max 2 satır
                draw_text_unicode(
                    frame, line, (10, y_off + i * 28),
                    font_size=22,
                    color=(255, 240, 120),  # sıcak sarı
                )
        else:
            cv2.putText(frame, "(isaret yapip 2.5sn bekleyin)",
                        (10, h - panel_h + 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)

        # ── Buffer doluluk çubuğu (alt, panel'in üstünde) ──
        ratio = len(self._buf) / self.sequence_length
        bar_y = h - panel_h - 6
        cv2.rectangle(frame, (0, bar_y), (int(w * ratio), bar_y + 4),
                      (0, 200, 140), -1)

        # Son tahmin + güven (panel üstü)
        if self._last_emit:
            color = (0, 255, 100) if last_conf >= self.CONF_THRESH else (100, 100, 200)
            cv2.putText(frame, f"{self._last_emit}  {last_conf:.0%}",
                        (w - 250, bar_y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # Cümle sonu geri sayım çubuğu
        if self._sentence and self._last_sign_t:
            elapsed = time.time() - self._last_sign_t
            remaining = max(0, self.PAUSE_SEC - elapsed)
            pct = 1 - remaining / self.PAUSE_SEC
            cv2.rectangle(frame, (0, bar_y - 4), (int(w * pct), bar_y),
                          (50, 150, 255), -1)

    # ── Gerçek zamanlı döngü ──────────────────────────────────────────────
    def run_realtime(
        self,
        camera_index: int = -1,
        sentence_callback=None,
        word_callback=None,
        show_window: bool = True,
    ):
        """
        Args:
            camera_index: -1 → otomatik, 0/1/2 → spesifik index
            sentence_callback(words): cümle tamamlanınca
            word_callback(word, conf): her tanınan kelimede
            show_window: kamera penceresini göster
        """
        if camera_index == -1:
            camera_index = _find_camera()

        cap = _open_camera(camera_index)
        if not cap.isOpened():
            _cam_help()
            raise RuntimeError(f"Kamera açılamadı (index={camera_index})")

        # Akış parametreleri ayarla (FPS, çözünürlük)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)

        holistic = _make_holistic()
        ts_ms = 0
        print(f"Kamera açıldı (index={camera_index}). ESC veya 'q' ile çıkın.\n")

        try:
            while cap.isOpened():
                ok, frame = cap.read()
                if not ok:
                    break
                ts_ms += 33

                # Sabit boyuta resize (training ile tutarlı)
                proc = cv2.resize(frame, TARGET_SIZE)
                rgb = cv2.cvtColor(proc, cv2.COLOR_BGR2RGB)
                mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

                try:
                    result = holistic.detect_for_video(mp_img, ts_ms)
                except (ValueError, RuntimeError):
                    continue

                kp = extract_keypoints(result)
                label, conf = self.process_frame(kp)

                if label:
                    print(f"  [{time.strftime('%H:%M:%S')}] {label} ({conf:.0%})")
                    if word_callback:
                        word_callback(label, conf)

                if self.sentence_complete():
                    sent = self.flush_sentence()
                    if sent:
                        print(f"\n  ── Cümle: {' '.join(sent)} ──")
                        if sentence_callback:
                            sentence_callback(sent)

                if show_window:
                    draw_landmarks(frame, result)
                    self._draw_ui(frame, conf)
                    cv2.imshow("Sign Recognition  (q=cikis)", frame)
                    k = cv2.waitKey(1) & 0xFF
                    if k in (27, ord('q')):
                        break

        finally:
            cap.release()
            holistic.close()
            if show_window:
                cv2.destroyAllWindows()
            # Kalan cümleyi gönder
            if self._sentence:
                sent = self.flush_sentence()
                print(f"\n  ── Son cümle: {' '.join(sent)} ──")
                if sentence_callback:
                    sentence_callback(sent)
