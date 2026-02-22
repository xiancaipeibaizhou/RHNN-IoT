
import torch
import torch.nn as nn
import numpy as np
import argparse
import os
import time
import pickle
from pathlib import Path
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, precision_score, recall_score, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import copy

# --- Utility Functions ---

def _infer_normal_indices(class_names):
    """
    Infers the indices of 'normal' classes based on their names.
    Returns a list of indices.
    """
    normal_indices = []
    for idx, name in enumerate(class_names):
        name_lower = str(name).lower()
        if ("benign" in name_lower) or ("normal" in name_lower) or (name_lower in {"nonvpn", "non-vpn", "non-tor", "nontor"}):
            normal_indices.append(idx)
    
    # Fallback: if no class matches, assume class 0 is normal
    if not normal_indices and len(class_names) > 0:
         normal_indices.append(0)
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

# ET-BERT style Transformer Model adapted for continuous features
class ETBERT(nn.Module):
    def __init__(self, input_dim, d_model=128, nhead=4, num_layers=2, num_classes=2, dropout=0.1):
        super(ETBERT, self).__init__()
        self.d_model = d_model
        
        # Project continuous features to d_model
        self.embedding = nn.Linear(input_dim, d_model)
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, num_classes)
        )
        
    def forward(self, x):
        # x shape: (batch_size, input_dim)
        # Treat as sequence length 1
        x = self.embedding(x)  # (batch_size, d_model)
        x = x.unsqueeze(1)     # (batch_size, 1, d_model)
        
        output = self.transformer_encoder(x) # (batch_size, 1, d_model)
        
        # Pooling (just take the single token)
        output = output.squeeze(1) # (batch_size, d_model)
        
        logits = self.classifier(output) # (batch_size, num_classes)
        return logits

def get_pyg_data(dataset_name):
    processed_dir = Path('/root/project/reon/processed_data') / dataset_name
    
    # Load PyG graph lists
    # Use weights_only=False to avoid future warning/error if not using trusted source, but local files are safe
    train_list = torch.load(processed_dir / 'train_graphs.pt', weights_only=False)
    val_list = torch.load(processed_dir / 'val_graphs.pt', weights_only=False)
    test_list = torch.load(processed_dir / 'test_graphs.pt', weights_only=False)
    
    def process_list(graph_list):
        features = []
        labels = []
        for data in graph_list:
            # Combine node features and edge features
            # Assuming 1 node and self-loops or simple structure where we can concat mean/sum
            # Based on previous TGN check, x is [1, 4], edge_attr is [1, 77]
            # We concat them to get 81 features
            if data.x.shape[0] > 0:
                x_feat = data.x.mean(dim=0) # Mean pooling if multiple nodes
            else:
                x_feat = torch.zeros(4) # Fallback
                
            if data.edge_attr is not None and data.edge_attr.shape[0] > 0:
                e_feat = data.edge_attr.mean(dim=0) # Mean pooling if multiple edges
            else:
                e_feat = torch.zeros(77) # Fallback (should match edge_dim)
            
            combined = torch.cat([x_feat, e_feat], dim=0)
            features.append(combined.numpy())
            
            # Label
            # If y is graph label
            if hasattr(data, 'y') and data.y is not None:
                 labels.append(data.y.item() if data.y.numel() == 1 else data.y[0].item())
            # If label is on edge (TGN case)
            elif hasattr(data, 'edge_labels') and data.edge_labels is not None:
                 labels.append(data.edge_labels.item() if data.edge_labels.numel() == 1 else data.edge_labels[0].item())
            else:
                 labels.append(0) # Default
                 
        return np.stack(features), np.array(labels)

    train_X, train_y = process_list(train_list)
    val_X, val_y = process_list(val_list)
    test_X, test_y = process_list(test_list)
    
    # Try to load label encoder for class names
    label_encoder_path = processed_dir / 'label_encoder.pkl'
    raw_class_names = None
    if label_encoder_path.exists():
        try:
            with open(label_encoder_path, 'rb') as f:
                le = pickle.load(f)
                raw_class_names = le.classes_
            print(f"Loaded class names: {raw_class_names}")
        except Exception as e:
            print(f"Failed to load label encoder with pickle: {e}")
            try:
                import pandas as pd
                le = pd.read_pickle(label_encoder_path)
                raw_class_names = le.classes_
                print(f"Loaded class names with pandas: {raw_class_names}")
            except Exception as e2:
                print(f"Failed to load label encoder with pandas: {e2}")

    return train_X, train_y, val_X, val_y, test_X, test_y, raw_class_names

class EarlyStopMonitor:
    def __init__(self, max_round=5, higher_better=True, tolerance=1e-10):
        self.max_round = max_round
        self.num_round = 0
        self.epoch_count = 0
        self.best_epoch = 0
        self.last_best = None
        self.higher_better = higher_better
        self.tolerance = tolerance

    def early_stop_check(self, curr_val):
        if not self.higher_better:
            curr_val *= -1
        
        if self.last_best is None:
            self.last_best = curr_val
            self.num_round = 0
            self.best_epoch = self.epoch_count
        elif (curr_val - self.last_best) / np.abs(self.last_best) > self.tolerance:
            self.last_best = curr_val
            self.num_round = 0
            self.best_epoch = self.epoch_count
        else:
            self.num_round += 1
        
        self.epoch_count += 1
        return self.num_round >= self.max_round

def train_etbert(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    print(f"Loading data from {args.dataset}...")
    train_X, train_y, val_X, val_y, test_X, test_y, raw_class_names = get_pyg_data(args.dataset)
    
    # Remap labels
    unique_labels = np.unique(np.concatenate([train_y, val_y, test_y]))
    
    # Resolve class names
    if raw_class_names is not None:
        # Map unique_labels (which are indices in raw_class_names) to names
        # Ensure we access raw_class_names with integer indices
        class_names = [str(raw_class_names[int(l)]) for l in unique_labels]
    else:
        class_names = [str(int(l)) for l in unique_labels]
    print(f"Active Class Names: {class_names}")

    label_map = {l: i for i, l in enumerate(unique_labels)}
    train_y = np.array([label_map[l] for l in train_y])
    val_y = np.array([label_map[l] for l in val_y])
    test_y = np.array([label_map[l] for l in test_y])
    
    num_classes = len(unique_labels)
    input_dim = train_X.shape[1]
    print(f"Input dim: {input_dim}, Num classes: {num_classes}")
    print(f"Train size: {len(train_X)}, Val size: {len(val_X)}, Test size: {len(test_X)}")
    
    # Convert to Tensor
    train_X = torch.FloatTensor(train_X).to(device)
    train_y = torch.LongTensor(train_y).to(device)
    val_X = torch.FloatTensor(val_X).to(device)
    val_y = torch.LongTensor(val_y).to(device)
    test_X = torch.FloatTensor(test_X).to(device)
    test_y = torch.LongTensor(test_y).to(device)
    
    # Create Datasets and Loaders
    train_dataset = torch.utils.data.TensorDataset(train_X, train_y)
    val_dataset = torch.utils.data.TensorDataset(val_X, val_y)
    test_dataset = torch.utils.data.TensorDataset(test_X, test_y)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.bs, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.bs, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.bs, shuffle=False)
    
    # Model
    model = ETBERT(input_dim=input_dim, d_model=args.d_model, nhead=args.nhead, 
                   num_layers=args.n_layer, num_classes=num_classes, dropout=args.dropout).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()
    
    early_stopper = EarlyStopMonitor(max_round=args.patience, higher_better=True)
    
    # Training Loop
    best_model_state = None
    
    for epoch in range(args.n_epoch):
        model.train()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())
            
        train_loss = total_loss / len(train_loader)
        train_acc = accuracy_score(all_labels, all_preds)
        train_f1 = f1_score(all_labels, all_preds, average='macro')
        
        # Validation
        model.eval()
        val_loss = 0
        val_preds = []
        val_labels = []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                logits = model(X_batch)
                loss = criterion(logits, y_batch)
                val_loss += loss.item()
                preds = torch.argmax(logits, dim=1)
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(y_batch.cpu().numpy())
        
        val_loss /= len(val_loader)
        val_acc = accuracy_score(val_labels, val_preds)
        val_f1 = f1_score(val_labels, val_preds, average='macro')
        
        print(f"Epoch {epoch+1}/{args.n_epoch} | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Train F1: {train_f1:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}")
        
        if early_stopper.early_stop_check(val_f1):
            print(f"Early stopping at epoch {epoch+1}")
            break
        
        if val_f1 == early_stopper.last_best:
            best_model_state = copy.deepcopy(model.state_dict())
            
    # Test with best model
    if best_model_state is not None:
        print("Loading best model for testing...")
        model.load_state_dict(best_model_state)
        # Save best model
        output_dir = os.path.join('baseline', 'ET-BERT', args.dataset)
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(output_dir, 'etbert_best_model.pth'))
    
    model.eval()
    test_loss = 0
    test_preds = []
    test_labels = []
    test_probs = []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            logits = model(X_batch)
            probs = torch.softmax(logits, dim=1)
            loss = criterion(logits, y_batch)
            test_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            test_preds.extend(preds.cpu().numpy())
            test_labels.extend(y_batch.cpu().numpy())
            test_probs.extend(probs.cpu().numpy())
            
    test_loss /= len(test_loader)
    
    # Metrics
    test_labels = np.array(test_labels)
    test_preds = np.array(test_preds)
    test_probs = np.array(test_probs)
    
    acc = accuracy_score(test_labels, test_preds)
    f1_m = f1_score(test_labels, test_preds, average='macro', zero_division=0)
    precision_w = precision_score(test_labels, test_preds, average='weighted', zero_division=0)
    recall_w = recall_score(test_labels, test_preds, average='weighted', zero_division=0)
    f1_w = f1_score(test_labels, test_preds, average='weighted', zero_division=0)
    
    try:
        if len(np.unique(test_labels)) == 2:
            if test_probs.shape[1] == 2:
                 auc = roc_auc_score(test_labels, test_probs[:, 1])
            else:
                 auc = 0.0
        else:
            auc = roc_auc_score(test_labels, test_probs, multi_class='ovr', average='macro')
    except Exception as e:
        print(f"AUC calculation failed: {e}")
        auc = 0.0
        
    # ASA & FAR
    normal_indices = _infer_normal_indices(class_names)
    print(f"Normal indices inferred: {normal_indices} (Names: {[class_names[i] for i in normal_indices]})")
    
    attack_mask = np.isin(test_labels, normal_indices, invert=True)
    if np.sum(attack_mask) > 0:
        asa = accuracy_score(test_labels[attack_mask], test_preds[attack_mask])
    else:
        asa = 0.0
        
    normal_mask = np.isin(test_labels, normal_indices)
    if np.sum(normal_mask) > 0:
        pred_attacks = np.isin(test_preds[normal_mask], normal_indices, invert=True)
        false_alarms = np.sum(pred_attacks)
        far = false_alarms / np.sum(normal_mask)
    else:
        far = 0.0
        
    print("-" * 30)
    print("Detailed Test Metrics:")
    print(f"Loss: {test_loss:.4f}")
    print(f"ACC: {acc:.4f}")
    print(f"APR (Weighted Precision): {precision_w:.4f}")
    print(f"RE (Weighted Recall): {recall_w:.4f}")
    print(f"AUC: {auc:.4f}")
    print(f"F1 (Weighted): {f1_w:.4f}")
    print(f"F1 (Macro): {f1_m:.4f}")
    print(f"FAR: {far:.4f}")
    print(f"ASA: {asa:.4f}")
    print("-" * 30)
    
    # Confusion Matrix
    cm_count = confusion_matrix(test_labels, test_preds)
    cm_norm = confusion_matrix(test_labels, test_preds, normalize='true')
    print("Confusion Matrix (Counts):")
    print(cm_count)
    print("Confusion Matrix (Percentages):")
    print(cm_norm)
    
    # Save Confusion Matrices
    _save_confusion_matrix_image(cm_count, class_names, os.path.join(output_dir, 'confusion_matrix.png'), "Confusion Matrix (Counts)")
    _save_confusion_matrix_image(cm_norm, class_names, os.path.join(output_dir, 'confusion_matrix_normalized.png'), "Confusion Matrix (Normalized)")
    
    print(f"Confusion matrices saved to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='unsw_nb15', help='Dataset name')
    parser.add_argument('--n_epoch', type=int, default=50, help='Number of epochs')
    parser.add_argument('--bs', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--d_model', type=int, default=128, help='Transformer hidden dimension')
    parser.add_argument('--nhead', type=int, default=4, help='Number of attention heads')
    parser.add_argument('--n_layer', type=int, default=2, help='Number of transformer layers')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay')
    
    args = parser.parse_args()
    train_etbert(args)
