"""
main.py
=======
İşaret Dili Tanıma + Akıcı Cümle Üretimi (DeepSeek) — CLI girişi.

Adımlar:
    1. Veri prep:
        python main.py prep --dataset autsl
        python main.py prep --dataset wlasl --top-k 100

    2. Eğitim:
        python main.py train --dataset autsl
        python main.py train --dataset wlasl

    3. Canlı tanıma:
        python main.py run --lang tr --camera 1
        python main.py run --lang en --camera 1

    4. Yardımcılar:
        python main.py status
        python main.py cameras
        python main.py test-sentence ben spor yapmak sevmek
"""
from __future__ import annotations
import sys, os, argparse

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")


# ── Komutlar ──────────────────────────────────────────────────────────────────
def cmd_prep(args):
    from dataset_prep import (AUTSLProcessor, WLASLProcessor,
                              ASLCitizenProcessor, status as ds_status)
    if args.dataset in ("autsl", "both"):
        AUTSLProcessor().process(args.max_per)
    if args.dataset in ("wlasl", "both"):
        WLASLProcessor().process(args.top_k, args.max_per)
    if args.dataset == "asl_citizen":
        ASLCitizenProcessor().process(args.top_k, args.max_per)
    ds_status()


def cmd_train(args):
    from train import train
    train(
        dataset = args.dataset,
        epochs  = args.epochs,
        hidden  = args.hidden,
        lr      = args.lr,
        batch   = args.batch,
    )


def cmd_status(_args):
    from dataset_prep import status
    status()
    print("Modeller:")
    for d in ("autsl", "asl_citizen", "wlasl"):
        mp = os.path.join(MODELS_DIR, f"model_{d}.pt")
        if os.path.exists(mp):
            import torch
            ckpt = torch.load(mp, map_location="cpu", weights_only=False)
            in_use = ""
            if d == "autsl":
                in_use = "  🇹🇷 (tr için kullanılıyor)"
            elif d == "asl_citizen":
                in_use = "  🇬🇧 (en için kullanılıyor)"
            elif d == "wlasl":
                in_use = "  💤 (yedek, kullanılmıyor)"
            print(f"  ✓ {d}: val_acc={ckpt.get('val_acc',0):.1%}, "
                  f"{len(ckpt['labels'])} sınıf{in_use}")
        else:
            print(f"  ✗ {d}: model yok → python main.py train --dataset {d}")


def cmd_cameras(_args):
    from recognition import list_cameras
    cams = list_cameras()
    print(f"Bulunan kameralar: {cams}")
    if not cams:
        print("Sistem Ayarları → Gizlilik → Kamera → Terminal'e izin ver")


def _pick_language_interactive() -> str:
    """Kullanıcıya dil seçim menüsü göster."""
    print("\n" + "═" * 50)
    print("  🤟  İşaret Dili Tanıma Sistemi")
    print("═" * 50)
    print()
    # Mevcut modelleri kontrol et
    tr_ok = os.path.exists(os.path.join(MODELS_DIR, "model_autsl.pt"))
    en_ok = os.path.exists(os.path.join(MODELS_DIR, "model_asl_citizen.pt"))

    print("  Dil seçin / Select language:")
    print()
    print(f"  1) 🇹🇷 Türkçe İşaret Dili (AUTSL) {'✓' if tr_ok else '✗ model yok'}")
    print(f"  2) 🇬🇧 American Sign Language (ASL Citizen) {'✓' if en_ok else '✗ model yok'}")
    print(f"  q) Çıkış")
    print()

    while True:
        choice = input("Seçim [1/2/q]: ").strip().lower()
        if choice in ("1", "tr", "türkçe", "turkce"):
            if not tr_ok:
                print("⚠ Türkçe modeli bulunamadı. Önce: python main.py train --dataset autsl")
                continue
            return "tr"
        if choice in ("2", "en", "english", "ingilizce"):
            if not en_ok:
                print("⚠ İngilizce modeli bulunamadı. Önce: python main.py train --dataset asl_citizen")
                continue
            return "en"
        if choice in ("q", "quit", "exit", "çıkış"):
            sys.exit(0)
        print("Geçersiz seçim. 1, 2, veya q yazın.")


def cmd_run(args):
    """Canlı tanıma + cümle inşası."""
    import threading

    # Dil belirlenmediyse menüyle sor
    if args.lang is None:
        args.lang = _pick_language_interactive()

    dataset = "autsl" if args.lang == "tr" else "asl_citizen"
    model_path = os.path.join(MODELS_DIR, f"model_{dataset}.pt")
    if not os.path.exists(model_path):
        print(f"Model bulunamadı: {model_path}")
        print(f"Önce eğit: python main.py train --dataset {dataset}")
        sys.exit(1)

    from recognition import SignRecognizer
    rec = SignRecognizer.from_checkpoint(model_path)

    # Sentence constructor opsiyonel
    sc = None
    if not args.no_llm:
        try:
            from sentence_constructor import SentenceConstructor
            sc = SentenceConstructor(model=args.llm, language=args.lang)
            print(f"Cümle inşası: {args.llm}")
        except Exception as e:
            print(f"⚠ Cümle inşası devre dışı ({e})")

    # NMT çevirmen opsiyonel — dil bazlı çift seç
    nmt = None
    if not args.no_translate:
        try:
            from nmt_module import NMTTranslator
            tgt = "en" if args.lang == "tr" else "tr"
            nmt = NMTTranslator(source_lang=args.lang, target_lang=tgt)
            print(f"Çeviri: {args.lang} → {tgt}")
        except Exception as e:
            print(f"⚠ NMT devre dışı ({e})")

    def on_word(word, conf):
        flag = "🇹🇷" if args.lang == "tr" else "🇬🇧"
        print(f"  {flag} {word} ({conf:.0%})")

    def on_sentence(words):
        """Cümle tamamlandığında çağrılır. Qwen'i thread'de çalıştırır."""
        print(f"\n{'─'*55}")
        print(f"  Kelimeler: {' → '.join(words)}")

        if not sc:
            # LLM yoksa sadece kelimeleri göster
            text = " ".join(words).capitalize() + "."
            rec.set_constructed_sentence(text, status="done")
            print(f"  Cümle    : {text}")
            print(f"{'─'*55}\n")
            return

        # UI'da "düşünüyor..." göster, ekranı dondurmadan thread'de çağır
        rec.set_constructed_sentence("(düşünüyor...)", status="thinking")

        def worker():
            try:
                t0 = time.time()
                sentence = sc.construct(words)

                # NMT ile çevir (varsa)
                translation = ""
                if nmt:
                    try:
                        translation = nmt.translate(sentence)
                    except Exception as te:
                        translation = f"(çeviri hatası: {te})"

                elapsed = time.time() - t0
                # UI'da hem cümle hem çeviri göster
                display = sentence
                if translation:
                    display = f"{sentence} | {translation}"
                rec.set_constructed_sentence(display, status="done")

                print(f"  Cümle    : {sentence}  [{elapsed:.1f}s]")
                if translation:
                    print(f"  Çeviri   : {translation}")
            except Exception as e:
                rec.set_constructed_sentence(f"⚠ Hata: {e}", status="error")
                print(f"  ⚠ Cümle inşası hatası: {e}")
            print(f"{'─'*55}\n")

        threading.Thread(target=worker, daemon=True).start()

    import time
    print(f"\nKamera: {args.camera if args.camera >= 0 else 'otomatik'}")
    print(f"Dil   : {args.lang} ({dataset})")
    print(f"Eşik  : {args.threshold:.0%}")
    print(f"LLM   : {args.llm if sc else 'devre dışı'}")
    print("Hazır — işaret yapmaya başlayın.\n")

    # Recognizer ayarlarını güncelle
    rec.CONF_THRESH = args.threshold
    rec.PAUSE_SEC = args.silence

    rec.run_realtime(
        camera_index=args.camera,
        sentence_callback=on_sentence,
        word_callback=on_word,
        show_window=not args.no_window,
    )


def cmd_test_sentence(args):
    """Sentence constructor'ı kelime listesiyle test et."""
    from sentence_constructor import SentenceConstructor
    sc = SentenceConstructor(model=args.llm, language=args.lang)
    print(f"Girdi: {args.words}")
    result = sc.construct(args.words)
    print(f"Çıktı: {result}")


def cmd_translate(args):
    """Kelimeleri cümle yapıp NMT ile hedef dile çevir."""
    from sentence_constructor import SentenceConstructor
    from nmt_module import NMTTranslator

    sc = SentenceConstructor(model=args.llm, language=args.lang)
    sentence = sc.construct(args.words)
    print(f"Kelimeler : {args.words}")
    print(f"Cümle     : {sentence}")

    translator = NMTTranslator(source_lang=args.lang, target_lang=args.target)
    translation = translator.translate(sentence)
    print(f"Çeviri    : {translation}")


def cmd_nmt_langs(_args):
    """Desteklenen NMT dil çiftlerini listele."""
    from nmt_module import supported_languages
    print("Desteklenen NMT dil çiftleri:")
    for lang in supported_languages():
        print(f"  {lang['source']} → {lang['target']}  "
              f"({lang['source_name']} → {lang['target_name']})")


# ── CLI ───────────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser(
        description="İşaret Dili Tanıma + Cümle İnşası",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    # prep
    pr = sub.add_parser("prep", help="Veri seti hazırla")
    pr.add_argument("--dataset",
                    choices=["autsl", "wlasl", "asl_citizen", "both"], required=True)
    pr.add_argument("--top-k", type=int, default=100,
                    help="WLASL için kullanılacak en sık k gloss (0=hepsi)")
    pr.add_argument("--max-per", type=int, default=60,
                    help="Sınıf başına maksimum örnek")

    # train
    tr = sub.add_parser("train", help="Model eğit")
    tr.add_argument("--dataset",
                    choices=["autsl", "wlasl", "asl_citizen"], required=True)
    tr.add_argument("--epochs", type=int, default=100)
    tr.add_argument("--hidden", type=int, default=128)
    tr.add_argument("--lr", type=float, default=5e-4)
    tr.add_argument("--batch", type=int, default=32)

    # run
    rn = sub.add_parser("run", help="Canlı tanıma + cümle inşası")
    rn.add_argument("--lang", choices=["tr", "en"], default=None,
                    help="tr → AUTSL modeli, en → WLASL modeli (boş = menüyle sor)")
    rn.add_argument("--camera", type=int, default=-1,
                    help="-1 = otomatik bul, 0/1/2 = spesifik index")
    rn.add_argument("--threshold", type=float, default=0.35,
                    help="Confidence eşiği (0.35 = %35)")
    rn.add_argument("--interval", type=int, default=5,
                    help="Her N karede bir tahmin yap (legacy, recognition.py STEP kullanır)")
    rn.add_argument("--silence", type=float, default=2.5,
                    help="Bu kadar saniye sessizlikten sonra cümle biter")
    rn.add_argument("--llm", default="qwen2.5:3b",
                    help="Ollama model adı")
    rn.add_argument("--no-llm", action="store_true",
                    help="LLM cümle inşasını devre dışı bırak")
    rn.add_argument("--no-translate", action="store_true",
                    help="NMT çevirisini devre dışı bırak")
    rn.add_argument("--no-window", action="store_true",
                    help="Kamera penceresini gösterme")

    # status / cameras / test-sentence
    sub.add_parser("status", help="Veri seti + model durumu")
    sub.add_parser("cameras", help="Bağlı kameraları listele")
    ts = sub.add_parser("test-sentence", help="Cümle inşasını test et")
    ts.add_argument("--lang", choices=["tr", "en"], default="tr")
    ts.add_argument("--llm", default="qwen2.5:3b")
    ts.add_argument("words", nargs="+")

    # translate / nmt-langs
    tn = sub.add_parser("translate",
                        help="Kelimeleri cümle yapıp NMT ile çevir")
    tn.add_argument("--lang", choices=["tr", "en"], default="tr",
                    help="Kaynak dil")
    tn.add_argument("--target", choices=["tr", "en"], default="en",
                    help="Hedef dil")
    tn.add_argument("--llm", default="qwen2.5:3b")
    tn.add_argument("words", nargs="+")

    sub.add_parser("nmt-langs", help="Desteklenen NMT dil çiftleri")

    args = p.parse_args()

    dispatch = {
        "prep": cmd_prep,
        "train": cmd_train,
        "run": cmd_run,
        "status": cmd_status,
        "cameras": cmd_cameras,
        "test-sentence": cmd_test_sentence,
        "translate": cmd_translate,
        "nmt-langs": cmd_nmt_langs,
    }
    dispatch[args.cmd](args)


if __name__ == "__main__":
    main()
