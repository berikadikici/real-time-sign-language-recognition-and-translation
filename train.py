"""
train.py
========
İşaret dili keypoint dizilerinden Transformer+BiLSTM sınıflandırıcı eğitir.

AUTSL veya WLASL için ayrı modeller eğitir.

Kullanım:
    python main.py train --dataset autsl
    python main.py train --dataset wlasl
"""
from __future__ import annotations

import os, json, time, random, argparse
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
KP_DIR   = os.path.join(BASE_DIR, "data", "keypoints")
SEED     = 42


def set_seed(seed: int = SEED):
    random.seed(seed); np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            torch.mps.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


# ── Model (Transformer + BiLSTM) ──────────────────────────────────────────────
def build_model(input_size: int, hidden: int, n_classes: int, n_layers: int = 3):
    import torch.nn as nn

    class SignModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj  = nn.Linear(input_size, hidden)
            self.norm0 = nn.LayerNorm(hidden)
            self.input_dropout = nn.Dropout(0.2)

            enc_layer = nn.TransformerEncoderLayer(
                d_model=hidden, nhead=max(1, hidden // 64),
                dim_feedforward=hidden * 4,
                dropout=0.3, batch_first=True,
            )
            self.transformer = nn.TransformerEncoder(enc_layer, num_layers=2)

            self.lstm = nn.LSTM(
                hidden, hidden // 2,
                num_layers=n_layers, batch_first=True,
                bidirectional=True,
                dropout=0.5 if n_layers > 1 else 0.0,
            )
            self.classifier = nn.Sequential(
                nn.LayerNorm(hidden),
                nn.Dropout(0.5),
                nn.Linear(hidden, n_classes),
            )

        def forward(self, x):
            x = self.norm0(self.proj(x))
            x = self.input_dropout(x)
            x = self.transformer(x)
            x, _ = self.lstm(x)
            return self.classifier(x.mean(dim=1) + x.max(dim=1).values)

    return SignModel()


# ── Veri yükleme ──────────────────────────────────────────────────────────────
def load_sequences(kp_dir: str, dataset: str, labels: list[str]):
    label2idx = {l: i for i, l in enumerate(labels)}
    X, y = [], []
    dd = os.path.join(kp_dir, dataset)
    for g in os.listdir(dd):
        if g not in label2idx:
            continue
        gd = os.path.join(dd, g)
        for sf in os.listdir(gd):
            sp = os.path.join(gd, sf, "sequence.npy")
            if not os.path.exists(sp):
                continue
            seq = np.load(sp).astype(np.float32)
            X.append(seq); y.append(label2idx[g])
    if not X:
        raise ValueError(f"Veri yok → {dd}")
    return X, np.array(y, np.int64)


def pad(seq: np.ndarray, seq_len: int) -> np.ndarray:
    if len(seq) < seq_len:
        seq = np.concatenate([seq, np.zeros((seq_len - len(seq), seq.shape[1]), np.float32)])
    return seq[:seq_len].astype(np.float32)


def augment(seq: np.ndarray, rng) -> np.ndarray:
    s = seq.copy()
    if rng.random() < 0.7:
        s += rng.normal(0, 0.015, s.shape).astype(np.float32)
    if rng.random() < 0.5 and len(s) > 6:
        off = rng.integers(1, 4)
        if rng.random() < 0.5:
            s = np.concatenate([s[off:], np.zeros((off, s.shape[1]), np.float32)])
        else:
            s = np.concatenate([np.zeros((off, s.shape[1]), np.float32), s[:-off]])
    return s


def per_seq_center(X: np.ndarray) -> np.ndarray:
    out = X.copy()
    for i in range(len(out)):
        s = out[i]
        nz = np.abs(s).sum(axis=1) > 1e-6
        if nz.sum() > 0:
            out[i, nz] = s[nz] - s[nz].mean(axis=0, keepdims=True)
    return out


def compute_stats(X: np.ndarray):
    flat = X.reshape(-1, X.shape[-1])
    nz = np.abs(flat).sum(axis=1) > 1e-6
    flat = flat[nz]
    return flat.mean(axis=0).astype(np.float32), (flat.std(axis=0).astype(np.float32) + 1e-6)


def standardize(X, mean, std):
    zm = np.abs(X).sum(axis=-1, keepdims=True) <= 1e-6
    out = ((X - mean) / std).astype(np.float32)
    out[np.broadcast_to(zm, out.shape)] = 0.0
    return out


# ── Eğitim ────────────────────────────────────────────────────────────────────
def train(
    dataset: str,
    kp_dir: str = KP_DIR,
    seq_len: int = 30,
    hidden: int = 128,
    n_layers: int = 3,
    epochs: int = 100,
    lr: float = 5e-4,
    batch: int = 32,
    val_split: float = 0.15,
    min_samples: int = 5,
    aug_factor: int = 3,
    warmup: int = 5,
    seed: int = SEED,
):
    set_seed(seed)
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    from sklearn.model_selection import train_test_split

    from dataset_prep import build_label_list

    out_dir = os.path.join(BASE_DIR, "models")
    os.makedirs(out_dir, exist_ok=True)
    model_path = os.path.join(out_dir, f"model_{dataset}.pt")
    label_path = os.path.join(out_dir, f"labels_{dataset}.json")

    all_labels = build_label_list(kp_dir, dataset=dataset, min_samples=min_samples)
    if not all_labels:
        raise ValueError(f"İşlenmiş {dataset} verisi yok. Önce: python main.py prep --dataset {dataset}")

    print(f"\n{'═'*60}")
    print(f"  {dataset.upper()} Eğitimi (seed={seed})")
    print(f"{'═'*60}")
    print(f"  Etiket   : {len(all_labels)}")
    print(f"  Hidden   : {hidden}")
    print(f"  Epoch    : {epochs} (warmup {warmup})")
    print(f"  LR       : {lr}")
    print(f"  Batch    : {batch}")
    print(f"{'═'*60}\n")

    print("Veri yükleniyor...")
    X_list, y = load_sequences(kp_dir, dataset, all_labels)
    print(f"  Orijinal: {len(X_list)}")

    indices = np.arange(len(X_list))
    tr_idx, val_idx = train_test_split(
        indices, test_size=val_split, stratify=y, random_state=seed,
    )

    X_tr_raw = [X_list[i] for i in tr_idx]
    y_tr_raw = y[tr_idx]
    X_val = np.array([pad(X_list[i], seq_len) for i in val_idx], np.float32)
    y_val = y[val_idx]

    # Augment
    rng = np.random.default_rng(seed)
    X_aug, y_aug = [], []
    for seq, lbl in zip(X_tr_raw, y_tr_raw):
        base = pad(seq, seq_len)
        X_aug.append(base); y_aug.append(lbl)
        for _ in range(aug_factor - 1):
            X_aug.append(augment(base, rng)); y_aug.append(lbl)
    X_tr = np.array(X_aug, np.float32)
    y_tr = np.array(y_aug, np.int64)
    print(f"  Train: {len(X_tr)} (augmented x{aug_factor}), Val: {len(X_val)}")

    # Centering + standardize
    X_tr  = per_seq_center(X_tr)
    X_val = per_seq_center(X_val)
    mean, std = compute_stats(X_tr)
    X_tr  = standardize(X_tr, mean, std)
    X_val = standardize(X_val, mean, std)

    device = (torch.device("mps") if torch.backends.mps.is_available()
              else torch.device("cuda") if torch.cuda.is_available()
              else torch.device("cpu"))
    print(f"  Cihaz: {device}\n")

    tr_ds  = TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr))
    val_ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
    tr_ld  = DataLoader(tr_ds, batch_size=batch, shuffle=True, drop_last=True)
    val_ld = DataLoader(val_ds, batch_size=batch, shuffle=False)

    input_size = X_tr.shape[2]
    model = build_model(input_size, hidden, len(all_labels), n_layers).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parametre: {n_params:,}\n")

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    def lr_lambda(e):
        if e < warmup:
            return (e + 1) / warmup
        prog = (e - warmup) / max(1, epochs - warmup)
        return 0.5 * (1 + np.cos(np.pi * prog))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)

    best_val = 0.0
    no_improve = 0
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        tr_loss = tr_correct = tr_total = 0
        for xb, yb in tr_ld:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tr_loss += loss.item()
            tr_correct += (logits.argmax(1) == yb).sum().item()
            tr_total   += len(yb)
        sched.step()

        model.eval()
        v_correct = v_total = 0
        with torch.no_grad():
            for xb, yb in val_ld:
                xb, yb = xb.to(device), yb.to(device)
                v_correct += (model(xb).argmax(1) == yb).sum().item()
                v_total   += len(yb)

        tr_acc = tr_correct / max(1, tr_total)
        val_acc = v_correct / max(1, v_total)
        elapsed = time.time() - t0
        print(f"  Epoch {epoch:>3}/{epochs}  "
              f"loss={tr_loss/len(tr_ld):.4f}  "
              f"tr={tr_acc:.1%}  val={val_acc:.1%}  [{elapsed:.0f}s]")

        if val_acc > best_val:
            best_val = val_acc
            torch.save({
                "state_dict":      model.to("cpu").state_dict(),
                "labels":          all_labels,
                "input_size":      int(input_size),
                "hidden_size":     hidden,
                "n_classes":       len(all_labels),
                "sequence_length": seq_len,
                "n_layers":        n_layers,
                "val_acc":         val_acc,
                "feat_mean":       mean,
                "feat_std":        std,
                "dataset":         dataset,
            }, model_path)
            model.to(device)
            print(f"    ✓ Kaydedildi (val={val_acc:.1%})")
            no_improve = 0
        else:
            no_improve += 1
        if no_improve >= 30:
            print(f"  ⏹ Early stop (30 epoch iyileşme yok)")
            break

    with open(label_path, "w", encoding="utf-8") as f:
        json.dump({"labels": all_labels,
                   "label2idx": {l: i for i, l in enumerate(all_labels)}},
                  f, ensure_ascii=False, indent=2)

    print(f"\n{'═'*60}")
    print(f"  En iyi val_acc : {best_val:.1%}")
    print(f"  Model          : {model_path}")
    print(f"{'═'*60}")
    return best_val


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["autsl", "wlasl"], required=True)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--seed", type=int, default=SEED)
    args = p.parse_args()

    train(
        dataset = args.dataset,
        epochs  = args.epochs,
        hidden  = args.hidden,
        lr      = args.lr,
        batch   = args.batch,
        seed    = args.seed,
    )


if __name__ == "__main__":
    main()
