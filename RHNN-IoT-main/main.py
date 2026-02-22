import argparse
import os
import sys
import time
import warnings
from sklearn.exceptions import InconsistentVersionWarning, UndefinedMetricWarning

# Suppress specific warnings
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    from data_loader import BENIGN_SETS, load_and_preprocess
except Exception:
    try:
        from EBAO1 import BENIGN_SETS, load_and_preprocess
    except Exception:
        BENIGN_SETS = {}
        load_and_preprocess = None

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _safe_dataset_tag(name: str) -> str:
    s = str(name).strip().replace(os.sep, "_")
    s = s.replace(" ", "_").replace("/", "_").replace("\\", "_")
    while "__" in s:
        s = s.replace("__", "_")
    return s


def _infer_normal_indices(class_names):
    normal_indices = []
    for idx, name in enumerate(class_names):
        name_lower = str(name).lower()
        if ("benign" in name_lower) or ("normal" in name_lower) or (name_lower in {"nonvpn", "non-vpn", "non-tor", "nontor"}):
            normal_indices.append(idx)
    return normal_indices


def _save_confusion_matrix_image(cm, class_names, out_path, title):
    # Sanitize inputs to prevent Glyph warnings (e.g., replace En Dash with Hyphen)
    title = str(title).replace("\u2013", "-").replace("\x96", "-")
    class_names = [str(c).replace("\u2013", "-").replace("\x96", "-") for c in class_names]

    fig, ax = plt.subplots(1, 1, figsize=(max(8, len(class_names) * 0.8), max(6, len(class_names) * 0.7)))
    
    # Calculate row-normalized confusion matrix for color mapping
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_norm = np.nan_to_num(cm_norm)  # Handle division by zero if any
    
    im = ax.imshow(cm_norm, interpolation="nearest", cmap=plt.cm.Blues, vmin=0, vmax=1)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(title)
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_ylabel("True")
    ax.set_xlabel("Pred")

    # Add text annotations
    thresh = 0.5  # Threshold for normalized values (0-1 range)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            count = cm[i, j]
            total = cm[i, :].sum()
            percent = (count / total * 100) if total > 0 else 0
            val_norm = cm_norm[i, j]
            
            text = f"{count}\n({percent:.1f}%)"
            
            ax.text(j, i, text,
                    ha="center", va="center",
                    color="white" if val_norm > thresh else "black",
                    fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def make_sequences(X, y, seq_len):
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64)
    if len(X) < seq_len:
        return np.zeros((0, seq_len, X.shape[1]), dtype=np.float32), np.zeros((0,), dtype=np.int64)
    out_X = np.lib.stride_tricks.sliding_window_view(X, (seq_len, X.shape[1]))[:, 0, :, :]
    out_X = np.ascontiguousarray(out_X)
    out_y = y[seq_len - 1 :]
    return out_X, out_y


class SeqDataset(Dataset):
    def __init__(self, X_seq, y_seq):
        self.X = torch.from_numpy(np.asarray(X_seq, dtype=np.float32))
        self.y = torch.from_numpy(np.asarray(y_seq, dtype=np.int64))

    def __len__(self):
        return int(self.y.shape[0])

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class RHNNClassifier(nn.Module):
    def __init__(self, in_dim, hidden, num_layers, num_classes, dropout):
        super().__init__()
        self.rnn = nn.GRU(
            input_size=in_dim,
            hidden_size=hidden,
            num_layers=int(num_layers),
            batch_first=True,
            dropout=float(dropout) if int(num_layers) > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x):
        _, h = self.rnn(x)
        last = h[-1]
        return self.head(last)


def _evaluate(model, loader, device, class_names, dataset_name):
    model.eval()
    normal_indices = _infer_normal_indices(class_names)
    all_true = []
    all_pred = []
    all_prob = []

    t0 = time.perf_counter()
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            probs = torch.softmax(logits, dim=1)
            pred = torch.argmax(probs, dim=1)
            all_true.append(yb.detach().cpu().numpy())
            all_pred.append(pred.detach().cpu().numpy())
            all_prob.append(probs.detach().cpu().numpy())
    t1 = time.perf_counter()

    if len(all_true) == 0:
        return {}, np.zeros((0,), dtype=int), np.zeros((0,), dtype=int), np.zeros((0, 0), dtype=float)

    y_true = np.concatenate(all_true).astype(int)
    y_pred = np.concatenate(all_pred).astype(int)
    y_prob = np.concatenate(all_prob).astype(float)

    acc = float((y_true == y_pred).mean()) if len(y_true) else 0.0
    prec = float(precision_score(y_true, y_pred, average="weighted", zero_division=0)) if len(y_true) else 0.0
    rec = float(recall_score(y_true, y_pred, average="weighted", zero_division=0)) if len(y_true) else 0.0
    f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0)) if len(y_true) else 0.0
    f1_m = float(f1_score(y_true, y_pred, average="macro", zero_division=0)) if len(y_true) else 0.0

    auc = float("nan")
    present = np.unique(y_true)
    if present.size >= 2:
        if len(class_names) == 2:
            if present.size == 2:
                try:
                    auc = float(roc_auc_score(y_true, y_prob[:, 1]))
                except Exception:
                    auc = float("nan")
        else:
            try:
                auc = float(
                    roc_auc_score(
                        y_true,
                        y_prob,
                        multi_class="ovr",
                        average="macro",
                        labels=np.arange(len(class_names)),
                    )
                )
            except Exception:
                aucs = []
                for c in present:
                    bin_true = (y_true == int(c)).astype(int)
                    if np.unique(bin_true).size < 2:
                        continue
                    try:
                        aucs.append(float(roc_auc_score(bin_true, y_prob[:, int(c)])))
                    except Exception:
                        continue
                if len(aucs) > 0:
                    auc = float(np.mean(aucs))

    is_true_normal = np.isin(y_true, normal_indices) if normal_indices else np.zeros_like(y_true, dtype=bool)
    is_pred_normal = np.isin(y_pred, normal_indices) if normal_indices else np.zeros_like(y_pred, dtype=bool)
    fp = int(np.logical_and(is_true_normal, ~is_pred_normal).sum())
    tn = int(np.logical_and(is_true_normal, is_pred_normal).sum())
    far = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0

    attack_mask = ~is_true_normal
    asa = float((y_pred[attack_mask] == y_true[attack_mask]).mean()) if attack_mask.any() else float("nan")

    metrics = {
        "acc": acc,
        "prec_weighted": prec,
        "recall_weighted": rec,
        "f1_weighted": f1,
        "f1_macro": f1_m,
        "auc_macro_ovr": auc,
        "far": far,
        "asa": asa,
        "inference_time_s": float(t1 - t0),
        "n_samples": int(len(y_true)),
        "dataset": str(dataset_name),
    }
    return metrics, y_true, y_pred, y_prob


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-name", default=os.getenv("DATASET_NAME", "CIC-IDS2017"))
    parser.add_argument("--data-dir", default=os.getenv("DATA_DIR", "data/CIC-IDS2017/TrafficLabelling_"))
    parser.add_argument("--baseline-dir", default=os.getenv("BASELINE_DIR", "baseline"))
    parser.add_argument("--seq-len", type=int, default=int(os.getenv("SEQ_LEN", "10")))
    parser.add_argument("--epochs", type=int, default=int(os.getenv("EPOCHS", "10")))
    parser.add_argument("--batch-size", type=int, default=int(os.getenv("BATCH_SIZE", "256")))
    parser.add_argument("--hidden", type=int, default=int(os.getenv("HIDDEN", "128")))
    parser.add_argument("--layers", type=int, default=int(os.getenv("LAYERS", "2")))
    parser.add_argument("--dropout", type=float, default=float(os.getenv("DROPOUT", "0.2")))
    parser.add_argument("--lr", type=float, default=float(os.getenv("LR", "0.001")))
    parser.add_argument("--weight-decay", type=float, default=float(os.getenv("WEIGHT_DECAY", "1e-5")))
    parser.add_argument("--fs-sample-size", type=int, default=int(os.getenv("FS_SAMPLE_SIZE", "5000")))
    parser.add_argument("--test-ratio", type=float, default=float(os.getenv("TEST_RATIO", "0.2")))
    parser.add_argument("--random-state", type=int, default=int(os.getenv("RANDOM_STATE", "42")))
    parser.add_argument("--max-train", type=int, default=int(os.getenv("MAX_TRAIN", "0")))
    parser.add_argument("--max-test", type=int, default=int(os.getenv("MAX_TEST", "0")))
    args = parser.parse_args()

    if load_and_preprocess is None:
        raise RuntimeError("EBAO1.load_and_preprocess not available")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    np.random.seed(args.random_state)
    torch.manual_seed(args.random_state)

    X_val, y_val, X_train, y_train, X_test, y_test, _, le, class_names = load_and_preprocess(
        args.data_dir,
        args.dataset_name,
        fs_sample_size=args.fs_sample_size,
        test_ratio=args.test_ratio,
        random_state=args.random_state,
    )

    if args.max_train and args.max_train > 0:
        X_train = X_train[: args.max_train]
        y_train = y_train[: args.max_train]
    if args.max_test and args.max_test > 0:
        X_test = X_test[: args.max_test]
        y_test = y_test[: args.max_test]

    Xtr, ytr = make_sequences(X_train, y_train, args.seq_len)
    Xte, yte = make_sequences(X_test, y_test, args.seq_len)

    train_loader = DataLoader(SeqDataset(Xtr, ytr), batch_size=args.batch_size, shuffle=True, drop_last=False)
    test_loader = DataLoader(SeqDataset(Xte, yte), batch_size=args.batch_size, shuffle=False, drop_last=False)

    val_loader = test_loader
    if X_val is not None and y_val is not None:
        Xv, yv = make_sequences(X_val, y_val, args.seq_len)
        if len(yv) > 0:
            val_loader = DataLoader(SeqDataset(Xv, yv), batch_size=args.batch_size, shuffle=False)
            print(f"Using separate validation set ({len(yv)} sequences).")
        else:
            print("Validation set empty after sequencing. Using test set for validation.")
    else:
        print("Using test set for validation (early stopping).")

    in_dim = int(X_train.shape[1]) if len(X_train.shape) == 2 else int(Xtr.shape[-1])
    if len(class_names) > 0:
        num_classes = len(class_names)
    else:
        num_classes = int(len(le.classes_)) if hasattr(le, "classes_") else int(np.max(y_train) + 1)

    model = RHNNClassifier(
        in_dim=in_dim,
        hidden=args.hidden,
        num_layers=args.layers,
        num_classes=num_classes,
        dropout=args.dropout,
    ).to(device)

    class_counts = np.bincount(ytr.astype(int), minlength=num_classes) if len(ytr) else np.ones((num_classes,), dtype=int)
    weights = 1.0 / np.sqrt(np.maximum(class_counts, 1))
    weights = weights / weights.sum() * num_classes
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_f1 = -1.0
    run_tag = time.strftime("%Y%m%d-%H%M%S")
    run_dir = os.path.join(args.baseline_dir, "RHNN", _safe_dataset_tag(args.dataset_name), run_tag)
    model_dir = os.path.join(run_dir, "model")
    cm_dir = os.path.join(run_dir, "confusion_matrix")
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(cm_dir, exist_ok=True)

    patience = 10
    no_improve_count = 0

    for epoch in range(int(args.epochs)):
        model.train()
        total_loss = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().cpu().item()) * int(xb.size(0))

        avg_loss = total_loss / max(1, len(train_loader.dataset))
        metrics, _, _, _ = _evaluate(model, val_loader, device, class_names, args.dataset_name)
        f1_m = float(metrics.get("f1_macro", float("nan")))
        print(f"Epoch {epoch+1}/{args.epochs} | loss={avg_loss:.6f} | f1_macro={f1_m:.6f}", flush=True)

        if np.isfinite(f1_m) and f1_m > best_f1:
            best_f1 = f1_m
            no_improve_count = 0
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "args": vars(args),
                    "class_names": list(class_names),
                },
                os.path.join(model_dir, "best_model.pt"),
            )
        else:
            no_improve_count += 1
            if no_improve_count >= patience:
                print(f"Early stopping triggered at epoch {epoch+1} (no improvement for {patience} epochs).")
                break

    # Load best model for final testing
    best_model_path = os.path.join(model_dir, "best_model.pt")
    if os.path.exists(best_model_path):
        print(f"Loading best model from {best_model_path} for final evaluation on Test set...")
        checkpoint = torch.load(best_model_path)
        model.load_state_dict(checkpoint["state_dict"])
    
    model.eval()
    metrics, y_true, y_pred, _ = _evaluate(model, test_loader, device, class_names, args.dataset_name)
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(num_classes))
    np.save(os.path.join(cm_dir, "cm_counts.npy"), cm)
    _save_confusion_matrix_image(cm, list(class_names), os.path.join(cm_dir, "cm_counts.png"), "Confusion Matrix (Counts)")

    print("-" * 50)
    for k in ["acc", "prec_weighted", "recall_weighted", "f1_weighted", "f1_macro", "auc_macro_ovr", "far", "asa", "inference_time_s", "n_samples"]:
        if k in metrics:
            print(f"{k:<25} | {metrics[k]}")
    print("-" * 50)
    print(f"Saved baseline run dir: {run_dir}", flush=True)


if __name__ == "__main__":
    main()

# 运行 Darknet2020
# python RHNN-IoT-main/main.py --dataset-name Darknet2020 --epochs 50

# # 运行 CIC-IDS2017
# python RHNN-IoT-main/main.py --dataset-name CIC-IDS2017 --epochs 50

# # 运行 UNSW-NB15
# python RHNN-IoT-main/main.py --dataset-name UNSW-NB15 --epochs 50

# # 运行 ISCX-IDS2012
# python RHNN-IoT-main/main.py --dataset-name ISCX-IDS2012 --epochs 50