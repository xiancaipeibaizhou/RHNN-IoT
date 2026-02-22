import argparse
import os
import time
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import MessagePassing

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

# -----------------------------------------------------------------------------
# Model Definition (PyG Version)
# -----------------------------------------------------------------------------

class EGraphSAGELayer(MessagePassing):
    def __init__(self, ndim_in, edims, ndim_out, activation):
        super(EGraphSAGELayer, self).__init__(aggr='mean')
        self.activation = activation
        # W_msg takes concatenated node features (neighbor) and edge features
        self.W_msg = nn.Linear(ndim_in + edims, ndim_out)
        # W_apply takes concatenated original node features and aggregated neighbor features
        self.W_apply = nn.Linear(ndim_in + ndim_out, ndim_out)

    def forward(self, x, edge_index, edge_attr):
        # x: [N, ndim_in]
        # edge_index: [2, E]
        # edge_attr: [E, edims]
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_j, edge_attr):
        # x_j: [E, ndim_in] (features of source nodes)
        # edge_attr: [E, edims]
        # Concat source node features and edge features
        tmp = torch.cat([x_j, edge_attr], dim=1)
        return self.W_msg(tmp)

    def update(self, aggr_out, x):
        # aggr_out: [N, ndim_out]
        # x: [N, ndim_in]
        # Concat original node features and aggregated neighbor features
        tmp = torch.cat([x, aggr_out], dim=1)
        return self.activation(self.W_apply(tmp))

class MLPPredictor(nn.Module):
    def __init__(self, in_features, out_classes):
        super().__init__()
        # Predictor takes concatenated features of two nodes (src and dst)
        self.W = nn.Linear(in_features * 2, out_classes)

    def forward(self, x, edge_index):
        # x: [N, in_features]
        # edge_index: [2, E]
        row, col = edge_index
        # Get features for source and destination nodes
        h_src = x[row]
        h_dst = x[col]
        # Concatenate and predict
        score = self.W(torch.cat([h_src, h_dst], dim=1))
        return score

class EGraphSAGE(nn.Module):
    def __init__(self, ndim_in, ndim_out, edim, activation, dropout, n_classes):
        super(EGraphSAGE, self).__init__()
        self.sage = EGraphSAGELayer(ndim_in, edim, ndim_out, activation)
        self.pred = MLPPredictor(ndim_out, n_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, edge_attr):
        h = self.sage(x, edge_index, edge_attr)
        h = self.dropout(h)
        return self.pred(h, edge_index)

# -----------------------------------------------------------------------------
# Data Loading
# -----------------------------------------------------------------------------

def load_processed_data(data_dir, dataset_name):
    """
    Load processed PyG data.
    """
    print(f"Loading data for {dataset_name} from {data_dir}...")
    dataset_path = os.path.join(data_dir, dataset_name)
    
    train_path = os.path.join(dataset_path, 'train_graphs.pt')
    val_path = os.path.join(dataset_path, 'val_graphs.pt')
    test_path = os.path.join(dataset_path, 'test_graphs.pt')
    
    # Load PyG data
    # weights_only=False is required for loading PyG Data objects in newer torch versions
    train_graphs = torch.load(train_path, weights_only=False)
    val_graphs = torch.load(val_path, weights_only=False)
    test_graphs = torch.load(test_path, weights_only=False)
    
    print(f"Loaded {len(train_graphs)} training graphs, {len(val_graphs)} validation graphs, {len(test_graphs)} test graphs.")
    return train_graphs, val_graphs, test_graphs

# -----------------------------------------------------------------------------
# Training and Evaluation
# -----------------------------------------------------------------------------

def _infer_normal_indices(class_names):
    normal_indices = []
    for idx, name in enumerate(class_names):
        name_lower = str(name).lower()
        if ("benign" in name_lower) or ("normal" in name_lower) or (name_lower in {"nonvpn", "non-vpn", "non-tor", "nontor"}):
            normal_indices.append(idx)
    if not normal_indices and len(class_names) > 0:
         # Fallback: assume class 0 is normal if no keyword matched
         normal_indices.append(0)
    return normal_indices

def evaluate(model, dataloader, device, class_names, dataset_name):
    model.eval()
    normal_indices = _infer_normal_indices(class_names)
    
    all_true = []
    all_pred = []
    all_prob = []
    total_loss = 0
    criterion = nn.CrossEntropyLoss()
    
    t0 = time.perf_counter()
    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            x = batch.x
            edge_index = batch.edge_index
            edge_attr = batch.edge_attr
            # Use edge_labels if available
            if hasattr(batch, 'edge_labels'):
                labels = batch.edge_labels.long()
            elif hasattr(batch, 'y'):
                labels = batch.y.long()
            else:
                continue
            
            logits = model(x, edge_index, edge_attr)
            loss = criterion(logits, labels)
            total_loss += loss.item()
            
            probs = F.softmax(logits, dim=1)
            _, predicted = torch.max(logits, 1)
            
            all_true.extend(labels.cpu().numpy())
            all_pred.extend(predicted.cpu().numpy())
            all_prob.extend(probs.cpu().numpy())
            
    t1 = time.perf_counter()
    avg_loss = total_loss / len(dataloader) if len(dataloader) > 0 else 0
    
    y_true = np.array(all_true)
    y_pred = np.array(all_pred)
    y_prob = np.array(all_prob)
    
    if len(y_true) == 0:
         return {}, np.zeros((0,), dtype=int), np.zeros((0,), dtype=int), np.zeros((0, 0), dtype=float)

    # Calculate metrics matching RHNN-IoT-main
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
                # Fallback for some edge cases
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
        "val_loss": avg_loss,
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
    parser = argparse.ArgumentParser(description='E-GraphSAGE Training (PyG Version)')
    parser.add_argument('--dataset', type=str, default='unsw_nb15', help='Dataset name')
    parser.add_argument('--data_dir', type=str, default='/root/project/reon/processed_data', help='Path to processed data')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--hidden_dim', type=int, default=64, help='Hidden dimension')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate')
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience')
    parser.add_argument('--output_dir', type=str, default='baseline/E-GraphSAGE', help='Output directory for results')
    args = parser.parse_args()
    args.output_dir = os.path.join(args.output_dir, args.dataset)
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        
    # Load data
    train_graphs, val_graphs, test_graphs = load_processed_data(args.data_dir, args.dataset)
    
    # Create dataloaders
    train_loader = DataLoader(train_graphs, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_graphs, batch_size=args.batch_size, shuffle=False)
    
    # Get dimensions
    sample_g = train_graphs[0]
    ndim_in = sample_g.x.shape[1]
    edim = sample_g.edge_attr.shape[1]
    
    # Determine number of classes
    all_labels = []
    for g in train_graphs:
        if hasattr(g, 'edge_labels'):
            all_labels.extend(g.edge_labels.tolist())
        elif hasattr(g, 'y'):
            all_labels.extend(g.y.tolist())
            
    if not all_labels:
         # Fallback if labels are not pre-loaded (should not happen with processed data)
         print("Warning: No labels found in data!")
         n_classes = 2
    else:
        n_classes = len(set(all_labels))
        
    print(f"Input Node Dim: {ndim_in}, Edge Dim: {edim}, Num Classes: {n_classes}")
    
    # Initialize model
    model = EGraphSAGE(ndim_in, args.hidden_dim, edim, F.relu, args.dropout, n_classes)
    model = model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    # Update output dir with timestamp
    run_tag = time.strftime("%Y%m%d-%H%M%S")
    args.output_dir = os.path.join(args.output_dir, run_tag)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        
    print(f"Output directory: {args.output_dir}")

    # Training Loop
    best_val_f1 = 0
    patience_counter = 0
    best_model_path = os.path.join(args.output_dir, 'best_model.pt')
    
    # Try to load label encoder for class names
    label_encoder_path = os.path.join(args.data_dir, args.dataset, 'label_encoder.pkl')
    class_names = []
    if os.path.exists(label_encoder_path):
        try:
            with open(label_encoder_path, 'rb') as f:
                le = pickle.load(f)
                class_names = le.classes_
            print(f"Loaded class names: {class_names}")
        except Exception as e:
            print(f"Failed to load label encoder with pickle: {e}")
            # Try loading with pandas (sometimes works for different python versions)
            try:
                import pandas as pd
                le = pd.read_pickle(label_encoder_path)
                class_names = le.classes_
                print(f"Loaded class names with pandas: {class_names}")
            except Exception as e2:
                print(f"Failed to load label encoder with pandas: {e2}")

    if not len(class_names):
        print("Using numerical class indices as names.")
        class_names = [str(i) for i in range(n_classes)]

    print("Starting training...")
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        start_time = time.time()
        
        for batch in train_loader:
            batch = batch.to(device)
            x = batch.x
            edge_index = batch.edge_index
            edge_attr = batch.edge_attr
            if hasattr(batch, 'edge_labels'):
                labels = batch.edge_labels.long()
            elif hasattr(batch, 'y'):
                labels = batch.y.long()
            
            optimizer.zero_grad()
            logits = model(x, edge_index, edge_attr)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_train_loss = total_loss / len(train_loader)
        
        # Validation
        metrics, _, _, _ = evaluate(model, val_loader, device, class_names, args.dataset)
        val_loss = metrics.get('val_loss', 0)
        val_acc = metrics.get('acc', 0)
        val_f1_macro = metrics.get('f1_macro', 0)

        epoch_time = time.time() - start_time
        print(f"Epoch {epoch+1}/{args.epochs} | Time: {epoch_time:.2f}s | "
              f"Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f} | "
              f"Val Acc: {val_acc:.4f} | Val F1 (Macro): {val_f1_macro:.4f}")
        
        # Early Stopping using F1 Macro
        if val_f1_macro > best_val_f1:
            best_val_f1 = val_f1_macro
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            print("  New best model saved!")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stopping triggered after {epoch+1} epochs.")
                break
                
    # Load best model
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path))
        
    # Final Test
    print("Final Evaluation on Test Set...")
    metrics, y_true, y_pred, y_prob = evaluate(model, test_loader, device, class_names, args.dataset)
    
    print("-" * 30)
    print("Detailed Test Metrics:")
    for k, v in metrics.items():
        if k != 'dataset':
             print(f"{k}: {v}")
    print("-" * 30)
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(n_classes))
    np.save(os.path.join(args.output_dir, "cm_counts.npy"), cm)
    _save_confusion_matrix_image(cm, class_names, os.path.join(args.output_dir, 'confusion_matrix.png'), "Confusion Matrix (Counts)")
    print(f"Confusion matrix saved to {os.path.join(args.output_dir, 'confusion_matrix.png')}")

if __name__ == '__main__':
    main()

# 运行 Darknet2020
# python3 /root/project/reon/E-GraphSAGE-master/E-GraphSAGE-master/E-GraphSAGE/standard/train_egraphsage.py --dataset darknet2020 --epochs 1 

# # 运行 CIC-IDS2017
# python3 /root/project/reon/E-GraphSAGE-master/E-GraphSAGE-master/E-GraphSAGE/standard/train_egraphsage.py --dataset cic_ids2017 --epochs 100 

# # 运行 UNSW-NB15
# python3 /root/project/reon/E-GraphSAGE-master/E-GraphSAGE-master/E-GraphSAGE/standard/train_egraphsage.py --dataset unsw_nb15 --epochs 100 

# # 运行 ISCX-IDS2012
# python3 /root/project/reon/E-GraphSAGE-master/E-GraphSAGE-master/E-GraphSAGE/standard/train_egraphsage.py --dataset iscx_ids2012 --epochs 100         
