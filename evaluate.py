"""
evaluate.py
===========
Eğitilmiş bir işaret dili tanıma modelinin kapsamlı değerlendirmesi.

Üretilen çıktılar (figures/ klasörüne):
    - confusion_matrix_<dataset>.png       (heatmap)
    - per_class_accuracy_<dataset>.png     (bar chart)
    - top_confusions_<dataset>.png         (en sık karıştırılan 20 çift)
    - confidence_distribution_<dataset>.png (true/false ayrımıyla histogram)
    - classification_report_<dataset>.txt   (precision/recall/F1 per class)
    - metrics_<dataset>.json                (genel metrikler)
    - per_class_metrics_<dataset>.csv       (sınıf başına metrikler)

Kullanım:
    python evaluate.py --dataset autsl
    python evaluate.py --dataset asl_citizen
    python evaluate.py --dataset wlasl
"""
from __future__ import annotations
import os, json, argparse, time
from collections import Counter
import numpy as np

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
KP_DIR      = os.path.join(BASE_DIR, "data", "keypoints")
MODELS_DIR  = os.path.join(BASE_DIR, "models")
FIGURES_DIR = os.path.join(BASE_DIR, "figures")
SEED        = 42


def set_seed(seed: int = SEED):
    import random
    random.seed(seed); np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            torch.mps.manual_seed(seed)
    except ImportError:
        pass


# ── Veri hazırlama (train.py ile aynı) ────────────────────────────────────────
def prepare_val_data(dataset: str, seed: int = SEED,
                     val_split: float = 0.15, min_samples: int = 5):
    """train.py ile aynı stratified split'i kullanır → aynı val set."""
    from dataset_prep import build_label_list
    from train import load_sequences, pad, per_seq_center, compute_stats, standardize
    from sklearn.model_selection import train_test_split

    labels = build_label_list(KP_DIR, dataset=dataset, min_samples=min_samples)
    if not labels:
        raise ValueError(f"İşlenmiş veri yok: data/keypoints/{dataset}/")

    print(f"  Etiket sayısı: {len(labels)}")

    X_raw, y_raw = load_sequences(KP_DIR, dataset, labels)
    print(f"  Toplam orijinal sekans: {len(X_raw)}")

    # train.py ile aynı stratified split
    indices = np.arange(len(X_raw))
    tr_idx, val_idx = train_test_split(
        indices, test_size=val_split, stratify=y_raw, random_state=seed,
    )

    seq_len = 30
    X_val = np.array([pad(X_raw[i], seq_len) for i in val_idx], np.float32)
    y_val = y_raw[val_idx]
    print(f"  Validation set: {len(X_val)}")

    return X_val, y_val, labels


def load_model_and_normalize(model_path: str):
    """Checkpoint'i yükle ve normalize stats'larını dön."""
    import torch
    from train import build_model

    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    device = (torch.device("mps") if torch.backends.mps.is_available()
              else torch.device("cuda") if torch.cuda.is_available()
              else torch.device("cpu"))
    model = build_model(
        ckpt["input_size"], ckpt["hidden_size"], ckpt["n_classes"],
        ckpt.get("n_layers", 3),
    )
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    return model, ckpt["labels"], ckpt.get("feat_mean"), ckpt.get("feat_std"), device


# ── Inference ─────────────────────────────────────────────────────────────────
def run_inference(model, X_val, y_val, feat_mean, feat_std, device,
                  batch_size: int = 32):
    """Tüm val set üzerinde inference. Returns: y_pred, y_proba (N, n_classes)."""
    import torch
    from train import per_seq_center, standardize

    print("  Per-sequence centering...")
    X = per_seq_center(X_val)
    if feat_mean is not None and feat_std is not None:
        print("  Z-score standardization...")
        X = standardize(X, feat_mean, feat_std)

    print(f"  Inference ({len(X)} örnek)...")
    y_proba_all = []
    t0 = time.time()
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            xb = torch.from_numpy(X[i:i + batch_size]).to(device)
            logits = model(xb)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            y_proba_all.append(probs)
    y_proba = np.concatenate(y_proba_all, axis=0)
    y_pred = y_proba.argmax(axis=1)
    elapsed = time.time() - t0
    print(f"  Inference süresi: {elapsed:.2f}s ({elapsed/len(X)*1000:.1f} ms/sample)")
    return y_pred, y_proba


# ── Metrik hesaplama ──────────────────────────────────────────────────────────
def compute_metrics(y_true, y_pred, y_proba, labels):
    """Bütün metrikleri hesapla."""
    from sklearn.metrics import (
        accuracy_score, precision_recall_fscore_support,
        classification_report, top_k_accuracy_score,
    )

    metrics = {}

    metrics["top1_accuracy"] = float(accuracy_score(y_true, y_pred))

    n_classes = len(labels)
    if n_classes >= 3:
        try:
            metrics["top3_accuracy"] = float(top_k_accuracy_score(
                y_true, y_proba, k=3, labels=list(range(n_classes))))
        except Exception:
            metrics["top3_accuracy"] = None
    if n_classes >= 5:
        try:
            metrics["top5_accuracy"] = float(top_k_accuracy_score(
                y_true, y_proba, k=5, labels=list(range(n_classes))))
        except Exception:
            metrics["top5_accuracy"] = None

    # Macro & weighted
    p_macro, r_macro, f_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0)
    p_weighted, r_weighted, f_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0)

    metrics["precision_macro"] = float(p_macro)
    metrics["recall_macro"]    = float(r_macro)
    metrics["f1_macro"]        = float(f_macro)
    metrics["precision_weighted"] = float(p_weighted)
    metrics["recall_weighted"]    = float(r_weighted)
    metrics["f1_weighted"]        = float(f_weighted)

    # Per-class
    p_pc, r_pc, f_pc, support_pc = precision_recall_fscore_support(
        y_true, y_pred, average=None, labels=list(range(n_classes)),
        zero_division=0)
    per_class = {
        labels[i]: {
            "precision": float(p_pc[i]),
            "recall":    float(r_pc[i]),
            "f1":        float(f_pc[i]),
            "support":   int(support_pc[i]),
        }
        for i in range(n_classes)
    }

    metrics["n_classes"]   = n_classes
    metrics["n_samples"]   = int(len(y_true))
    metrics["random_baseline"] = 1.0 / n_classes

    return metrics, per_class


# ── Görselleştirme ────────────────────────────────────────────────────────────
def plot_confusion_matrix(y_true, y_pred, labels, out_path: str,
                          max_labels: int = 40):
    """Confusion matrix heatmap. Çok sınıf varsa en sık 40 sınıf gösterilir."""
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix

    # Eğer çok sınıf varsa en sık karşılaşılanları göster
    if len(labels) > max_labels:
        class_counts = Counter(y_true)
        top_classes = [c for c, _ in class_counts.most_common(max_labels)]
        mask = np.isin(y_true, top_classes)
        y_true_f = y_true[mask]
        y_pred_f = y_pred[mask]
        idx_map = {c: i for i, c in enumerate(top_classes)}
        y_true_idx = np.array([idx_map[c] for c in y_true_f])
        y_pred_idx = np.array([idx_map.get(p, -1) for p in y_pred_f])
        valid = y_pred_idx >= 0
        y_true_idx = y_true_idx[valid]
        y_pred_idx = y_pred_idx[valid]
        shown_labels = [labels[c] for c in top_classes]
        title_suffix = f" (top-{max_labels} sınıf)"
    else:
        y_true_idx = y_true
        y_pred_idx = y_pred
        shown_labels = labels
        title_suffix = ""

    cm = confusion_matrix(y_true_idx, y_pred_idx,
                          labels=list(range(len(shown_labels))))

    fig_size = max(8, min(24, len(shown_labels) * 0.4))
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))

    # Row-normalize
    cm_norm = cm.astype(np.float32)
    row_sums = cm_norm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    cm_norm = cm_norm / row_sums

    im = ax.imshow(cm_norm, cmap="Blues", aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(len(shown_labels)))
    ax.set_yticks(range(len(shown_labels)))
    ax.set_xticklabels(shown_labels, rotation=90, fontsize=7)
    ax.set_yticklabels(shown_labels, fontsize=7)
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)
    ax.set_title(f"Confusion Matrix (row-normalized){title_suffix}",
                 fontsize=14, pad=12)

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {out_path}")


def plot_per_class_accuracy(y_true, y_pred, labels, out_path: str,
                            top_n: int = 30):
    """Sınıf başına accuracy bar chart. Çok sınıfta en sık 30 sınıf."""
    import matplotlib.pyplot as plt

    class_acc = {}
    for cls_idx, cls_name in enumerate(labels):
        mask = y_true == cls_idx
        if mask.sum() > 0:
            class_acc[cls_name] = (y_pred[mask] == cls_idx).mean()
    sorted_classes = sorted(class_acc.items(), key=lambda x: -x[1])
    n_show = min(top_n, len(sorted_classes))
    top_classes = sorted_classes[:n_show]

    names = [c[0] for c in top_classes]
    accs = [c[1] for c in top_classes]
    colors = ["#2ca02c" if a >= 0.8 else
              "#ffaa00" if a >= 0.5 else
              "#d62728" for a in accs]

    fig, ax = plt.subplots(figsize=(max(8, n_show * 0.4), 6))
    bars = ax.bar(range(len(names)), accs, color=colors, edgecolor="black",
                  linewidth=0.5)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=90, fontsize=8)
    ax.set_ylabel("Per-class accuracy", fontsize=12)
    ax.set_ylim([0, 1.05])
    ax.set_title(f"Per-class accuracy (top-{n_show} by accuracy)",
                 fontsize=14)
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.grid(axis="y", linestyle=":", alpha=0.4)

    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{acc:.0%}", ha="center", va="bottom", fontsize=7)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {out_path}")


def plot_top_confusions(y_true, y_pred, labels, out_path: str,
                        top_n: int = 20):
    """En sık karıştırılan sınıf çiftleri (true ≠ pred)."""
    import matplotlib.pyplot as plt

    wrong_mask = y_true != y_pred
    pairs = list(zip(y_true[wrong_mask], y_pred[wrong_mask]))
    pair_counts = Counter(pairs)
    top_pairs = pair_counts.most_common(top_n)

    if not top_pairs:
        print("  ⚠ Hiç hata yok, top confusions atlanıyor")
        return

    names = [f"{labels[t]} → {labels[p]}" for (t, p), _ in top_pairs]
    counts = [c for _, c in top_pairs]

    fig, ax = plt.subplots(figsize=(10, max(4, len(names) * 0.4)))
    bars = ax.barh(range(len(names)), counts, color="#d62728",
                   edgecolor="black", linewidth=0.5)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Hata sayısı", fontsize=12)
    ax.set_title(f"Top-{len(names)} en sık karıştırılan sınıf çifti",
                 fontsize=14)
    ax.grid(axis="x", linestyle=":", alpha=0.4)

    for bar, c in zip(bars, counts):
        ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height() / 2,
                str(c), va="center", fontsize=9)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {out_path}")


def plot_confidence_distribution(y_true, y_pred, y_proba, out_path: str):
    """Doğru ve yanlış tahminlerin güven dağılımı."""
    import matplotlib.pyplot as plt

    confidences = y_proba.max(axis=1)
    correct_mask = y_true == y_pred

    correct_conf = confidences[correct_mask]
    wrong_conf = confidences[~correct_mask]

    fig, ax = plt.subplots(figsize=(10, 6))
    bins = np.linspace(0, 1, 31)
    ax.hist(correct_conf, bins=bins, alpha=0.6, label=f"Doğru ({len(correct_conf)})",
            color="#2ca02c", edgecolor="black", linewidth=0.5)
    ax.hist(wrong_conf, bins=bins, alpha=0.6, label=f"Yanlış ({len(wrong_conf)})",
            color="#d62728", edgecolor="black", linewidth=0.5)

    ax.set_xlabel("Confidence (max softmax probability)", fontsize=12)
    ax.set_ylabel("Örnek sayısı", fontsize=12)
    ax.set_title("Confidence dağılımı (doğru vs yanlış tahminler)",
                 fontsize=14)
    ax.axvline(0.45, color="black", linestyle="--", linewidth=1,
               label="Gate 3 eşiği (0.45)")
    ax.legend(fontsize=11)
    ax.grid(linestyle=":", alpha=0.4)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {out_path}")


# ── Rapor yazma ───────────────────────────────────────────────────────────────
def save_classification_report(y_true, y_pred, labels, out_path: str):
    from sklearn.metrics import classification_report
    report = classification_report(
        y_true, y_pred, target_names=labels, zero_division=0, digits=3)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"  ✓ {out_path}")


def save_per_class_csv(per_class: dict, out_path: str):
    import csv
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["class", "precision", "recall", "f1", "support"])
        for cls, m in sorted(per_class.items()):
            w.writerow([cls, f"{m['precision']:.4f}", f"{m['recall']:.4f}",
                        f"{m['f1']:.4f}", m["support"]])
    print(f"  ✓ {out_path}")


def save_metrics_json(metrics: dict, out_path: str):
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"  ✓ {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────
def evaluate(dataset: str, seed: int = SEED, batch_size: int = 32):
    set_seed(seed)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    model_path = os.path.join(MODELS_DIR, f"model_{dataset}.pt")
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model bulunamadı: {model_path}\n"
            f"Önce eğit: python main.py train --dataset {dataset}"
        )

    print(f"\n{'═'*60}")
    print(f"  Değerlendirme: {dataset.upper()}")
    print(f"{'═'*60}")

    # 1. Veri
    print("\n[1/5] Veri hazırlanıyor...")
    X_val, y_val, labels_data = prepare_val_data(dataset, seed=seed)

    # 2. Model
    print("\n[2/5] Model yükleniyor...")
    model, labels_ckpt, feat_mean, feat_std, device = load_model_and_normalize(model_path)
    print(f"  Cihaz: {device}")
    print(f"  Checkpoint etiketleri: {len(labels_ckpt)}")

    # Etiket eşitlik kontrolü
    if labels_data != labels_ckpt:
        print(f"  ⚠ Veri etiketleri ile checkpoint etiketleri tam eşleşmiyor.")
        print(f"  Checkpoint'in label listesi kullanılacak.")
    labels = labels_ckpt

    # 3. Inference
    print("\n[3/5] Inference çalışıyor...")
    y_pred, y_proba = run_inference(model, X_val, y_val, feat_mean, feat_std,
                                    device, batch_size=batch_size)

    # 4. Metrikler
    print("\n[4/5] Metrikler hesaplanıyor...")
    metrics, per_class = compute_metrics(y_val, y_pred, y_proba, labels)

    # 5. Görselleştirme + raporlar
    print("\n[5/5] Görselleştirme + rapor yazılıyor...")
    plot_confusion_matrix(y_val, y_pred, labels,
        os.path.join(FIGURES_DIR, f"confusion_matrix_{dataset}.png"))
    plot_per_class_accuracy(y_val, y_pred, labels,
        os.path.join(FIGURES_DIR, f"per_class_accuracy_{dataset}.png"))
    plot_top_confusions(y_val, y_pred, labels,
        os.path.join(FIGURES_DIR, f"top_confusions_{dataset}.png"))
    plot_confidence_distribution(y_val, y_pred, y_proba,
        os.path.join(FIGURES_DIR, f"confidence_distribution_{dataset}.png"))
    save_classification_report(y_val, y_pred, labels,
        os.path.join(FIGURES_DIR, f"classification_report_{dataset}.txt"))
    save_per_class_csv(per_class,
        os.path.join(FIGURES_DIR, f"per_class_metrics_{dataset}.csv"))
    save_metrics_json(metrics,
        os.path.join(FIGURES_DIR, f"metrics_{dataset}.json"))

    # ── Özet ──
    print(f"\n{'═'*60}")
    print(f"  ÖZET — {dataset.upper()}")
    print(f"{'═'*60}")
    print(f"  Validation set boyutu : {metrics['n_samples']}")
    print(f"  Sınıf sayısı          : {metrics['n_classes']}")
    print(f"  Random baseline       : {metrics['random_baseline']:.1%}")
    print(f"  Top-1 accuracy        : {metrics['top1_accuracy']:.1%}")
    if metrics.get("top3_accuracy") is not None:
        print(f"  Top-3 accuracy        : {metrics['top3_accuracy']:.1%}")
    if metrics.get("top5_accuracy") is not None:
        print(f"  Top-5 accuracy        : {metrics['top5_accuracy']:.1%}")
    print(f"  Precision (macro)     : {metrics['precision_macro']:.4f}")
    print(f"  Recall (macro)        : {metrics['recall_macro']:.4f}")
    print(f"  F1 (macro)            : {metrics['f1_macro']:.4f}")
    print(f"  Precision (weighted)  : {metrics['precision_weighted']:.4f}")
    print(f"  Recall (weighted)     : {metrics['recall_weighted']:.4f}")
    print(f"  F1 (weighted)         : {metrics['f1_weighted']:.4f}")
    print(f"\n  Tüm çıktılar: {FIGURES_DIR}/")
    print(f"{'═'*60}\n")
    return metrics


def main():
    p = argparse.ArgumentParser(description="İşaret dili tanıma modeli değerlendirmesi")
    p.add_argument("--dataset",
                   choices=["autsl", "wlasl", "asl_citizen"], required=True,
                   help="Değerlendirilecek model")
    p.add_argument("--seed", type=int, default=SEED,
                   help="Stratified split için seed (train.py ile aynı olmalı)")
    p.add_argument("--batch", type=int, default=32)
    args = p.parse_args()

    evaluate(args.dataset, seed=args.seed, batch_size=args.batch)


if __name__ == "__main__":
    main()
