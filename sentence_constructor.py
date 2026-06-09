"""
sentence_constructor.py
=======================
Tanınan kelime listesini akıcı cümleye dönüştürür.

Backend: Ollama + Qwen 2.5 3B (~1.9 GB, on-device, ücretsiz).
         Few-shot prompting ile Türkçe gramer kalitesi yüksek.

Kurulum:
    brew install ollama
    brew services start ollama
    ollama pull qwen2.5:3b

Alternatifler:
    qwen2.5:7b      → daha kaliteli (~4.7 GB)
    deepseek-r1:1.5b → en küçük ama Türkçe zayıf
"""
from __future__ import annotations

import subprocess, re


# Few-shot örnekler — model bunları taklit ederek doğru gramer üretir
FEW_SHOT = {
    "tr": [
        ("sen, kitap, okumak",            "Sen kitap okuyorsun."),
        ("biz, yemek, yemek, istemek",    "Biz yemek yemek istiyoruz."),
        ("ben, ev, gitmek",               "Ben eve gidiyorum."),
        ("o, müzik, dinlemek, sevmek",    "O müzik dinlemeyi seviyor."),
        ("biz, okul, gitmek, lazım",      "Bizim okula gitmemiz lazım."),
        ("sen, ben, görmek, istemek",     "Sen beni görmek istiyorsun."),
    ],
    "en": [
        ("you, book, read",               "You are reading a book."),
        ("we, food, eat, want",           "We want to eat food."),
        ("i, home, go",                   "I am going home."),
        ("he, music, listen, like",       "He likes listening to music."),
        ("we, school, go, need",          "We need to go to school."),
        ("you, me, see, want",            "You want to see me."),
    ],
}


def _strip_think(text: str) -> str:
    """DeepSeek-R1 gibi reasoning modellerinin <think>...</think> bloklarını temizler."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


class SentenceConstructor:
    def __init__(
        self,
        model: str = "qwen2.5:3b",
        language: str = "tr",
        timeout: int = 30,
    ):
        self.model = model
        self.language = language if language in FEW_SHOT else "tr"
        self.timeout = timeout
        self._verify_ollama()

    # ── kurulum kontrolü ──────────────────────────────────────────────────
    def _verify_ollama(self):
        try:
            subprocess.run(
                ["ollama", "--version"],
                check=True, capture_output=True, timeout=5,
            )
        except (FileNotFoundError, subprocess.CalledProcessError):
            raise RuntimeError(
                "Ollama kurulu değil.\n"
                "  Kurmak için: brew install ollama\n"
                "  Ardından:   ollama serve &\n"
                "             ollama pull " + self.model
            )

        # Modeli kontrol et
        try:
            r = subprocess.run(
                ["ollama", "list"],
                capture_output=True, text=True, timeout=10,
            )
            mname = self.model.split(":")[0]
            if mname not in r.stdout:
                print(f"  ⚠ Model henüz inmemiş. İndirmek için: ollama pull {self.model}")
        except subprocess.TimeoutExpired:
            print("  ⚠ Ollama yanıt vermiyor — `ollama serve &` çalıştığından emin ol.")

    # ── prompt ────────────────────────────────────────────────────────────
    def _build_prompt(self, words: list[str]) -> str:
        """Few-shot prompting ile yüksek kalite Türkçe/İngilizce üretir."""
        words_str = ", ".join(words)
        examples = FEW_SHOT.get(self.language, FEW_SHOT["tr"])

        if self.language == "tr":
            intro = (
                "Sana Türk İşaret Dili kelime listesi vereceğim. "
                "Bunlardan akıcı, gramer kurallarına uygun TEK bir Türkçe cümle kur. "
                "Sadece cümleyi yaz, açıklama yapma.\n\n"
                "Örnekler:\n"
            )
            example_lines = "\n".join(
                f"Kelimeler: {inp}\nCümle: {out}\n" for inp, out in examples
            )
            return f"{intro}{example_lines}\nKelimeler: {words_str}\nCümle:"
        else:
            intro = (
                "I will give you a list of American Sign Language words. "
                "Convert them into ONE fluent, grammatically correct English sentence. "
                "Output only the sentence, no explanation.\n\n"
                "Examples:\n"
            )
            example_lines = "\n".join(
                f"Words: {inp}\nSentence: {out}\n" for inp, out in examples
            )
            return f"{intro}{example_lines}\nWords: {words_str}\nSentence:"

    # ── inference ─────────────────────────────────────────────────────────
    def construct(self, words: list[str]) -> str:
        words = [w.strip().lower() for w in words if w and w.strip()]
        if not words:
            return ""
        prompt = self._build_prompt(words)

        # JSON API ile çağır (Ollama REST yerine subprocess + format=json kullanmıyoruz,
        # çünkü deepseek-r1 reasoning modeli — daha güvenilir: stdin'den prompt ver)
        try:
            r = subprocess.run(
                ["ollama", "run", self.model],
                input=prompt,
                capture_output=True, text=True,
                timeout=self.timeout,
            )
        except subprocess.TimeoutExpired:
            return " ".join(words).capitalize() + "."

        out = _strip_think(r.stdout or "")
        lines = [l.strip() for l in out.splitlines() if l.strip()]
        if not lines:
            return " ".join(words).capitalize() + "."

        # Bütün satırları kontrol et, gerçek cümleyi bul:
        # En uzun, harf içeren, normal noktalama ile biten satır
        candidates = []
        for ln in lines:
            # Prefix temizle
            for prefix in ("Cümle:", "Sentence:", "Answer:", "Cevap:", "Output:"):
                if ln.lower().startswith(prefix.lower()):
                    ln = ln[len(prefix):].strip()
            # Bozuk/tırnak temizle
            ln = ln.strip('"\' .,)]}*-=>')
            ln = re.sub(r'\s+', ' ', ln)
            # En az 3 kelime ve alfanumerik karakter çoğunluğu
            if (len(ln.split()) >= 2 and
                sum(c.isalpha() for c in ln) > len(ln) * 0.5):
                candidates.append(ln)

        if not candidates:
            return " ".join(words).capitalize() + "."

        # En uzun adayı seç (genelde gerçek cümle en uzun)
        best = max(candidates, key=len)
        # Sonuna nokta ekle eğer yoksa
        if not best.endswith(('.', '!', '?')):
            best += '.'
        # Baş harfi büyüt
        if best:
            best = best[0].upper() + best[1:]
        return best

    # ── kısa test ─────────────────────────────────────────────────────────
    def __call__(self, words: list[str]) -> str:
        return self.construct(words)


# ── CLI testi ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Sentence constructor testi")
    p.add_argument("--lang", default="tr", choices=["tr", "en"])
    p.add_argument("--model", default="qwen2.5:3b")
    p.add_argument("words", nargs="+", help="Kelime listesi")
    a = p.parse_args()

    sc = SentenceConstructor(model=a.model, language=a.lang)
    print(f"Girdi:   {a.words}")
    print(f"Çıktı:   {sc.construct(a.words)}")
