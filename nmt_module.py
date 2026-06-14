"""
nmt_module.py
=============
Neural Machine Translation modülü.

İşaret dili kelimelerini → akıcı cümleye → hedef dile çevirir.

Pipeline:
    [İşaret kelimeleri] → [Ham cümle] → NMTTranslator → [Çeviri]

Desteklenen çeviri yönleri:
    tr → en   (Türkçe → İngilizce)
    en → fr   (İngilizce → Fransızca)

Kullanılan modeller (HuggingFace, tamamen offline çalışır):
    Helsinki-NLP/opus-mt-tc-big-tr-en
    Helsinki-NLP/opus-mt-en-fr

Kurulum:
    pip install transformers sentencepiece sacremoses

Kullanım:
    from nmt_module import NMTTranslator

    # Türkçe → İngilizce
    t = NMTTranslator(source_lang="tr", target_lang="en")
    t.translate("Ben eve gidiyorum.")
    # → "I am going home."

    # İngilizce → Fransızca
    t = NMTTranslator(source_lang="en", target_lang="fr")
    t.translate("I am going home.")
    # → "Je vais à la maison."
"""
from __future__ import annotations
import re
from typing import Optional

# ── Gloss → Cümle few-shot örnekler (sentence_constructor.py için referans) ──

FEW_SHOT_TR = [
    ("sen kitap okumak",         "Sen kitap okuyorsun."),
    ("biz yemek yemek istemek",  "Biz yemek yemek istiyoruz."),
    ("ben ev gitmek",            "Ben eve gidiyorum."),
    ("o müzik dinlemek sevmek",  "O müzik dinlemeyi seviyor."),
    ("biz okul gitmek lazım",    "Bizim okula gitmemiz lazım."),
    ("sen yardım istemek",       "Sen yardım istiyorsun."),
    ("ben su içmek",             "Ben su içiyorum."),
    ("o hasta olmak",            "O hasta."),
]

FEW_SHOT_EN = [
    ("you book read",            "You are reading a book."),
    ("we food eat want",         "We want to eat food."),
    ("i home go",                "I am going home."),
    ("he music listen like",     "He likes listening to music."),
    ("i water drink",            "I am drinking water."),
    ("she sick be",              "She is sick."),
]

FEW_SHOT_FR = [
    ("tu livre lire",                  "Tu lis un livre."),
    ("nous nourriture manger vouloir", "Nous voulons manger."),
    ("je maison aller",                "Je vais à la maison."),
    ("il musique écouter aimer",       "Il aime écouter de la musique."),
    ("nous école aller falloir",       "Nous devons aller à l'école."),
    ("tu aide demander",               "Tu demandes de l'aide."),
    ("je eau boire",                   "Je bois de l'eau."),
    ("elle malade être",               "Elle est malade."),
    ("je toi voir vouloir",            "Je veux te voir."),
    ("nous heureux être",              "Nous sommes heureux."),
]

# ── NMT Çevirmen ──────────────────────────────────────────────────────────────

# HuggingFace model eşleşmeleri
_MODEL_MAP = {
    ("tr", "en"): "Helsinki-NLP/opus-mt-tc-big-tr-en",
    ("en", "fr"): "Helsinki-NLP/opus-mt-en-fr",
}

_LANG_NAMES = {
    "tr": "Türkçe",
    "en": "İngilizce",
    "fr": "Fransızca",
}


class NMTTranslator:
    """
    Helsinki-NLP MarianMT tabanlı hafif çevirmen.
    Model ilk kullanımda indirilir (~300 MB), sonra cache'den yüklenir.
    """
    def __init__(self, source_lang: str = "tr", target_lang: str = "en"):
        self.source_lang = source_lang
        self.target_lang = target_lang
        self._tokenizer = None
        self._model = None
        self._loaded = False

        if source_lang == target_lang:
            raise ValueError("Kaynak ve hedef dil aynı olamaz.")
        if (source_lang, target_lang) not in _MODEL_MAP:
            supported = ", ".join(f"{s}→{t}" for s, t in _MODEL_MAP)
            raise ValueError(
                f"Desteklenmeyen dil çifti: {source_lang}→{target_lang}\n"
                f"Desteklenenler: {supported}"
            )

    def _load(self):
        """Modeli lazy-load et (ilk çeviri çağrısında)."""
        if self._loaded:
            return
        try:
            from transformers import MarianMTModel, MarianTokenizer
        except ImportError:
            raise ImportError(
                "transformers paketi gerekli.\n"
                "Kurmak için: pip install transformers sentencepiece sacremoses"
            )

        model_name = _MODEL_MAP[(self.source_lang, self.target_lang)]
        src_name = _LANG_NAMES.get(self.source_lang, self.source_lang)
        tgt_name = _LANG_NAMES.get(self.target_lang, self.target_lang)
        print(f"[NMT] {src_name} → {tgt_name} modeli yükleniyor: {model_name}")
        print("[NMT] İlk çalıştırmada model indirilir (~300 MB), sonraki çalıştırmalarda cache'den gelir.")

        self._tokenizer = MarianTokenizer.from_pretrained(model_name)
        self._model = MarianMTModel.from_pretrained(model_name)
        self._model.eval()
        self._loaded = True
        print(f"[NMT] ✓ Model hazır.")

    def translate(self, text: str) -> str:
        """Metni hedef dile çevir."""
        if not text.strip():
            return ""
        self._load()

        import torch
        inputs = self._tokenizer([text], return_tensors="pt",
                                 padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            translated = self._model.generate(**inputs, max_length=512,
                                              num_beams=4, early_stopping=True)
        result = self._tokenizer.decode(translated[0], skip_special_tokens=True)
        return result

    def translate_batch(self, texts: list[str]) -> list[str]:
        """Birden fazla cümleyi toplu çevir (daha hızlı)."""
        if not texts:
            return []
        self._load()

        import torch
        inputs = self._tokenizer(texts, return_tensors="pt",
                                 padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            translated = self._model.generate(**inputs, max_length=512,
                                              num_beams=4, early_stopping=True)
        return [self._tokenizer.decode(t, skip_special_tokens=True) for t in translated]


# ── Desteklenen diller ────────────────────────────────────────────────────────

def supported_languages() -> list[dict]:
    """Desteklenen dil çiftlerini listele."""
    return [
        {"source": s, "target": t,
         "source_name": _LANG_NAMES.get(s, s),
         "target_name": _LANG_NAMES.get(t, t),
         "model": m}
        for (s, t), m in _MODEL_MAP.items()
    ]


# ── CLI testi ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="NMT Modülü Testi")
    p.add_argument("--source", default="tr")
    p.add_argument("--target", default="en")
    p.add_argument("--list-langs", action="store_true")
    p.add_argument("words", nargs="*")
    a = p.parse_args()

    if a.list_langs:
        for lang in supported_languages():
            print(f"  {lang['source']} → {lang['target']}  "
                  f"({lang['source_name']} → {lang['target_name']})")
        raise SystemExit(0)

    words = a.words or ["ben", "ev", "gitmek"]
    translator = NMTTranslator(source_lang=a.source, target_lang=a.target)
    sentence = " ".join(words).capitalize() + "."
    print(f"Girdi  : {sentence}")
    print(f"Çeviri : {translator.translate(sentence)}")
