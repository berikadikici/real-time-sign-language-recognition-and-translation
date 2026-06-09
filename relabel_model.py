"""
relabel_model.py
================
Eğitilmiş AUTSL modelinin etiketlerini Türkçe kelimelere günceller.

Model ağırlıkları DEĞİŞMEZ — sadece label_id → label_name eşlemesi günceller.
Eğitim sırasında kullanılan sınıf sıralaması korunur (CLASS_000 → CLASS_225 alfabetik).

Kullanım:
    python relabel_model.py
    # veya farklı CSV:
    python relabel_model.py --csv my_labels.csv --model models/model_autsl.pt
"""
from __future__ import annotations
import os, csv, json, argparse, shutil

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CSV = os.path.join(BASE_DIR, "autsl_class_map.csv")
DEFAULT_MODEL = os.path.join(BASE_DIR, "models", "model_autsl.pt")
DEFAULT_LABELS = os.path.join(BASE_DIR, "models", "labels_autsl.json")


def load_class_map(csv_path: str) -> dict[int, str]:
    """ClassId → Türkçe kelime eşlemesi yükler."""
    m = {}
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cid = row.get("ClassId") or row.get("classid")
            tr = row.get("TR") or row.get("tr") or row.get("label")
            if cid is not None and tr:
                try:
                    m[int(cid)] = tr.strip().upper()
                except ValueError:
                    pass
    return m


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default=DEFAULT_CSV, help="Class map CSV")
    p.add_argument("--model", default=DEFAULT_MODEL, help="Model checkpoint")
    p.add_argument("--labels", default=DEFAULT_LABELS, help="Labels JSON")
    p.add_argument("--no-backup", action="store_true", help="Yedek alma")
    args = p.parse_args()

    if not os.path.exists(args.csv):
        print(f"❌ CSV bulunamadı: {args.csv}")
        return
    if not os.path.exists(args.model):
        print(f"❌ Model bulunamadı: {args.model}")
        return

    print(f"📂 CSV     : {args.csv}")
    print(f"📂 Model   : {args.model}")
    print(f"📂 Labels  : {args.labels}")

    # Yedekle
    if not args.no_backup:
        bak = args.model + ".bak"
        if not os.path.exists(bak):
            shutil.copy2(args.model, bak)
            print(f"✓ Yedek alındı: {bak}")
        lbak = args.labels + ".bak"
        if os.path.exists(args.labels) and not os.path.exists(lbak):
            shutil.copy2(args.labels, lbak)
            print(f"✓ Yedek alındı: {lbak}")

    # Class map yükle
    class_map = load_class_map(args.csv)
    print(f"✓ {len(class_map)} class haritası yüklendi")

    # Modeli aç
    import torch
    ckpt = torch.load(args.model, map_location="cpu", weights_only=False)
    old_labels = ckpt["labels"]
    print(f"✓ Modelde {len(old_labels)} mevcut etiket: {old_labels[:5]}...")

    # Yeni etiketleri eski sıraya göre üret
    # Mevcut etiketler "CLASS_000", "CLASS_001"... formatında alfabetik sıralı
    new_labels = []
    missing = []
    for old in old_labels:
        if old.startswith("CLASS_"):
            try:
                cid = int(old.replace("CLASS_", ""))
                if cid in class_map:
                    new_labels.append(class_map[cid])
                else:
                    new_labels.append(old)
                    missing.append(cid)
            except ValueError:
                new_labels.append(old)
        else:
            # Zaten Türkçe etiket varsa olduğu gibi bırak
            new_labels.append(old)

    print(f"✓ {len(new_labels) - len(missing)} etiket Türkçe'ye çevrildi")
    if missing:
        print(f"⚠ {len(missing)} class için Türkçe karşılık yok: {missing[:5]}")

    # Modeli güncelle (sadece labels field'ı değişir, weights aynı)
    ckpt["labels"] = new_labels
    ckpt["n_classes"] = len(new_labels)
    torch.save(ckpt, args.model)
    print(f"✓ Model güncellendi: {args.model}")

    # Labels JSON'u da güncelle
    os.makedirs(os.path.dirname(args.labels), exist_ok=True)
    with open(args.labels, "w", encoding="utf-8") as f:
        json.dump(
            {"labels": new_labels,
             "label2idx": {l: i for i, l in enumerate(new_labels)}},
            f, ensure_ascii=False, indent=2,
        )
    print(f"✓ Labels JSON güncellendi: {args.labels}")

    print("\nÖrnek yeni etiketler:")
    for i, l in enumerate(new_labels[:15]):
        print(f"  {i:3d} → {l}")


if __name__ == "__main__":
    main()
