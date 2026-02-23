
import argparse
import torch
import torch.nn as nn
import numpy as np
import os
import pickle
from pathlib import Path
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, precision_score, recall_score, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import copy
from tqdm import tqdm
import time

# --- Utility Functions ---

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

def calculate_laplacian_with_self_loop(matrix):
    # If matrix is already on GPU, we should create eye on same device
    eye = torch.eye(matrix.size(0)).to(matrix.device)
    matrix = matrix + eye
    row_sum = matrix.sum(1)
    d_inv_sqrt = torch.pow(row_sum, -0.5).flatten()
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0
    d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
    normalized_laplacian = (
        matrix.matmul(d_mat_inv_sqrt).transpose(0, 1).matmul(d_mat_inv_sqrt)
    )
    return normalized_laplacian

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

# --- Modified T-GCN Model ---

class TGCNGraphConvolution(nn.Module):
    def __init__(self, adj, num_gru_units: int, output_dim: int, input_dim: int, bias: float = 0.0):
        super(TGCNGraphConvolution, self).__init__()
        self._num_gru_units = num_gru_units
        self._output_dim = output_dim
        self._input_dim = input_dim
        self._bias_init_value = bias
        
        # Calculate Laplacian once. Note: adj should be on correct device before passing or moved later.
        # We register buffer so it moves with model.
        if not isinstance(adj, torch.Tensor):
            adj = torch.FloatTensor(adj)
        self.register_buffer(
            "laplacian", calculate_laplacian_with_self_loop(adj)
        )
        
        # Weights: input is [x, h]. x has input_dim, h has num_gru_units.
        self.weights = nn.Parameter(
            torch.FloatTensor(self._input_dim + self._num_gru_units, self._output_dim)
        )
        self.biases = nn.Parameter(torch.FloatTensor(self._output_dim))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weights)
        nn.init.constant_(self.biases, self._bias_init_value)

    def forward(self, inputs, hidden_state):
        batch_size, num_nodes, _ = inputs.shape 
        # inputs (batch_size, num_nodes, input_dim)
        # hidden_state (batch_size, num_nodes, num_gru_units)
        
        # [x, h] (batch_size, num_nodes, input_dim + num_gru_units)
        concatenation = torch.cat((inputs, hidden_state), dim=2)
        
        # (num_nodes, input_dim + num_gru_units, batch_size)
        concatenation = concatenation.transpose(0, 1).transpose(1, 2)
        
        # (num_nodes, (input_dim + num_gru_units) * batch_size)
        concatenation = concatenation.reshape(
            (num_nodes, (self._input_dim + self._num_gru_units) * batch_size)
        )
        
        # A[x, h]
        a_times_concat = self.laplacian @ concatenation
        
        # Reshape back: (num_nodes, input_dim + num_gru_units, batch_size)
        a_times_concat = a_times_concat.reshape(
            (num_nodes, self._input_dim + self._num_gru_units, batch_size)
        )
        # (batch_size, num_nodes, input_dim + num_gru_units)
        a_times_concat = a_times_concat.transpose(0, 2).transpose(1, 2)
        
        # (batch_size * num_nodes, input_dim + num_gru_units)
        a_times_concat = a_times_concat.reshape(
            (batch_size * num_nodes, self._input_dim + self._num_gru_units)
        )
        
        # A[x, h]W + b -> (batch_size * num_nodes, output_dim)
        outputs = a_times_concat @ self.weights + self.biases
        
        # (batch_size, num_nodes * output_dim)
        outputs = outputs.reshape((batch_size, num_nodes * self._output_dim))
        return outputs

class TGCNCell(nn.Module):
    def __init__(self, adj, input_dim: int, hidden_dim: int):
        super(TGCNCell, self).__init__()
        self._input_dim = input_dim
        self._hidden_dim = hidden_dim
        
        if not isinstance(adj, torch.Tensor):
            adj = torch.FloatTensor(adj)
        self.register_buffer("adj", adj)
        
        self.graph_conv1 = TGCNGraphConvolution(
            self.adj, self._hidden_dim, self._hidden_dim * 2, self._input_dim, bias=1.0
        )
        self.graph_conv2 = TGCNGraphConvolution(
            self.adj, self._hidden_dim, self._hidden_dim, self._input_dim
        )

    def forward(self, inputs, hidden_state):
        # [r, u] = sigmoid(A[x, h]W + b)
        concatenation = torch.sigmoid(self.graph_conv1(inputs, hidden_state))
        
        # r, u shape: (batch_size, num_nodes * num_gru_units) -> need to reshape?
        # graph_conv1 output is (batch_size, num_nodes * 2 * hidden_dim)
        # We split into r and u.
        
        # Wait, graph_conv1 output_dim was hidden_dim * 2.
        # So it returns (batch_size, num_nodes * hidden_dim * 2).
        # We need to reshape to split per node?
        # Or just split the last dim?
        
        batch_size = inputs.shape[0]
        num_nodes = inputs.shape[1]
        
        concatenation = concatenation.reshape((batch_size, num_nodes, 2 * self._hidden_dim))
        r, u = torch.chunk(concatenation, chunks=2, dim=2)
        
        # r, u: (batch_size, num_nodes, hidden_dim)
        
        # c = tanh(A[x, (r * h)W + b])
        c = torch.tanh(self.graph_conv2(inputs, r * hidden_state))
        c = c.reshape((batch_size, num_nodes, self._hidden_dim))
        
        # h := u * h + (1 - u) * c
        new_hidden_state = u * hidden_state + (1.0 - u) * c
        
        return new_hidden_state, new_hidden_state

class TGCN(nn.Module):
    def __init__(self, adj, hidden_dim: int, input_dim: int, num_classes: int, dropout: float = 0.0, **kwargs):
        super(TGCN, self).__init__()
        self._input_dim = input_dim
        self._hidden_dim = hidden_dim
        self._num_classes = num_classes
        
        if not isinstance(adj, torch.Tensor):
            adj = torch.FloatTensor(adj)
        self.register_buffer("adj", adj)
        
        self.tgcn_cell = TGCNCell(self.adj, input_dim, hidden_dim)
        
        # Classifier
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, inputs):
        # inputs: (batch_size, seq_len, num_nodes, input_dim)
        batch_size, seq_len, num_nodes, input_dim = inputs.shape
        
        hidden_state = torch.zeros(batch_size, num_nodes, self._hidden_dim).type_as(inputs)
        
        for i in range(seq_len):
            # step_input: (batch_size, num_nodes, input_dim)
            step_input = inputs[:, i, :, :]
            output, hidden_state = self.tgcn_cell(step_input, hidden_state)
        
        # hidden_state: (batch_size, num_nodes, hidden_dim)
        # We need to classify based on active node.
        # However, we don't pass active node index here.
        # Strategy: Return hidden states, let training loop pick.
        # Or: Apply classifier to all nodes, return (batch_size, num_nodes, num_classes)
        
        logits = self.classifier(self.dropout(hidden_state))
        return logits

# --- Data Loading ---

def get_data_and_adj(dataset_name, seq_len=10):
    processed_dir = Path('/root/project/reon/processed_data') / dataset_name
    print(f"Loading data from {processed_dir}...")
    
    train_list = torch.load(processed_dir / 'train_graphs.pt', weights_only=False)
    val_list = torch.load(processed_dir / 'val_graphs.pt', weights_only=False)
    test_list = torch.load(processed_dir / 'test_graphs.pt', weights_only=False)
    
    # 1. Identify all unique nodes
    all_data = train_list + val_list + test_list
    node_map = {}
    next_id = 0
    
    for data in all_data:
        if data.n_id.dim() == 0:
            raw_id = data.n_id.item()
        else:
            raw_id = data.n_id[0].item()
        if raw_id not in node_map:
            node_map[raw_id] = next_id
            next_id += 1
            
    num_nodes = len(node_map)
    print(f"Total unique nodes: {num_nodes}")
    
    # 2. Build Adjacency Matrix
    # Assume fully connected + self-loops for robustness if topology is unknown
    adj = np.ones((num_nodes, num_nodes), dtype=np.float32)
    # Or just Identity? T-GCN paper uses spatial adj. 
    # With 4 nodes, fully connected is fine.
    
    # 3. Create Sequences
    # Feature dim
    # Assuming standard processing: x (4) + edge_attr (77) -> 81
    input_dim = 81
    
    def process_split(graph_list, seq_len):
        X = []
        y = []
        active_nodes = [] # To know which node to classify
        
        # We slide over the list.
        # Each step corresponds to one graph (event).
        # At step t, we have event E_t.
        # We look back seq_len steps: E_{t-seq_len+1} ... E_t.
        # Target is label of E_t.
        
        # Convert all graphs to (node_idx, feature, label)
        events = []
        for data in graph_list:
            # Node ID
            if data.n_id.dim() == 0: raw_id = data.n_id.item()
            else: raw_id = data.n_id[0].item()
            node_idx = node_map[raw_id]
            
            # Feature
            if data.x.shape[0] > 0: x_feat = data.x.mean(dim=0)
            else: x_feat = torch.zeros(4)
            if data.edge_attr is not None and data.edge_attr.shape[0] > 0: e_feat = data.edge_attr.mean(dim=0)
            else: e_feat = torch.zeros(77)
            feature = torch.cat([x_feat, e_feat], dim=0).numpy()
            
            # Label
            if hasattr(data, 'y') and data.y is not None:
                 lbl = data.y.item() if data.y.numel() == 1 else data.y[0].item()
            elif hasattr(data, 'edge_labels') and data.edge_labels is not None:
                 lbl = data.edge_labels.item() if data.edge_labels.numel() == 1 else data.edge_labels[0].item()
            else:
                 lbl = 0
            
            events.append((node_idx, feature, lbl))
            
        # Create windows
        # Start from seq_len-1
        num_samples = len(events) - seq_len + 1
        if num_samples <= 0:
            return None, None, None
            
        for i in range(num_samples):
            # Window: events[i : i+seq_len]
            window_events = events[i : i+seq_len]
            
            # Construct input tensor for this window: (seq_len, num_nodes, input_dim)
            # Init with zeros (or last known value? Zeros implies no activity which is distinct)
            # Using zeros is safer for sparse events.
            seq_tensor = np.zeros((seq_len, num_nodes, input_dim), dtype=np.float32)
            
            for t, (n_idx, feat, _) in enumerate(window_events):
                seq_tensor[t, n_idx, :] = feat
                
            # Target is the label of the last event in window
            target_label = window_events[-1][2]
            target_node = window_events[-1][0]
            
            X.append(seq_tensor)
            y.append(target_label)
            active_nodes.append(target_node)
            
        return np.array(X), np.array(y), np.array(active_nodes)

    train_X, train_y, train_nodes = process_split(train_list, seq_len)
    val_X, val_y, val_nodes = process_split(val_list, seq_len)
    test_X, test_y, test_nodes = process_split(test_list, seq_len)
    
    # Load label encoder if available
    label_encoder_path = processed_dir / 'label_encoder.pkl'
    class_names = []
    if label_encoder_path.exists():
        try:
            with open(label_encoder_path, 'rb') as f:
                le = pickle.load(f, encoding='latin1')
                class_names = le.classes_
            print(f"Loaded class names: {class_names}")
        except Exception as e:
            print(f"Failed to load label encoder with pickle: {e}")
            try:
                import pandas as pd
                le = pd.read_pickle(label_encoder_path)
                class_names = list(le.classes_)
                print(f"Loaded class names with pandas: {class_names}")
            except Exception as e2:
                print(f"Failed to load label encoder with pandas: {e2}")
    
    if not class_names:
        print("Using numerical class indices as names.")
        if dataset_name == 'unsw_nb15':
            print("Trying to use hardcoded class names for UNSW-NB15 (Alphabetical)...")
            # Standard UNSW-NB15 classes in alphabetical order
            possible_names = ['Analysis', 'Backdoor', 'DoS', 'Exploits', 'Fuzzers', 'Generic', 'Normal', 'Reconnaissance', 'Shellcode', 'Worms']
            class_names = possible_names
            print(f"Using hardcoded class names: {class_names}")
            
    return (train_X, train_y, train_nodes), (val_X, val_y, val_nodes), (test_X, test_y, test_nodes), adj, input_dim, num_nodes, class_names

# --- Main Training Loop ---

def train_tgcn(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    (tr_X, tr_y, tr_nodes), (va_X, va_y, va_nodes), (te_X, te_y, te_nodes), adj, input_dim, num_nodes, class_names = \
        get_data_and_adj(args.dataset, args.seq_len)
        
    print(f"Train size: {len(tr_X)}, Val size: {len(va_X)}, Test size: {len(te_X)}")
    print(f"Input dim: {input_dim}, Num Nodes: {num_nodes}")
    
    # Remap labels
    unique_labels = np.unique(np.concatenate([tr_y, va_y, te_y]))
    label_map = {l: i for i, l in enumerate(unique_labels)}
    tr_y = np.array([label_map[l] for l in tr_y])
    va_y = np.array([label_map[l] for l in va_y])
    te_y = np.array([label_map[l] for l in te_y])
    num_classes = len(unique_labels)
    print(f"Num classes: {num_classes}")
    
    # Convert to Tensor
    tr_X = torch.FloatTensor(tr_X).to(device)
    tr_y = torch.LongTensor(tr_y).to(device)
    tr_nodes = torch.LongTensor(tr_nodes).to(device)
    
    va_X = torch.FloatTensor(va_X).to(device)
    va_y = torch.LongTensor(va_y).to(device)
    va_nodes = torch.LongTensor(va_nodes).to(device)
    
    te_X = torch.FloatTensor(te_X).to(device)
    te_y = torch.LongTensor(te_y).to(device)
    te_nodes = torch.LongTensor(te_nodes).to(device)
    
    adj = torch.FloatTensor(adj).to(device)
    
    # Dataset/Loader
    batch_size = args.bs
    train_dataset = torch.utils.data.TensorDataset(tr_X, tr_y, tr_nodes)
    val_dataset = torch.utils.data.TensorDataset(va_X, va_y, va_nodes)
    test_dataset = torch.utils.data.TensorDataset(te_X, te_y, te_nodes)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Model
    model = TGCN(adj=adj, hidden_dim=args.hidden_dim, input_dim=input_dim, 
                 num_classes=num_classes, dropout=args.dropout).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()
    
    # Standardized timestamped output directory
    run_tag = time.strftime("%Y%m%d-%H%M%S")
    output_dir = os.path.join('baseline', 'T-GCN', args.dataset, run_tag)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    best_val_f1 = 0
    patience_counter = 0
    best_model_path = os.path.join(output_dir, 'tgcn_best_model.pth')
    
    for epoch in range(args.n_epoch):
        model.train()

        total_loss = 0
        all_preds = []
        all_labels = []
        
        for X_batch, y_batch, nodes_batch in train_loader:
            optimizer.zero_grad()
            # X_batch: (B, Seq, Nodes, Dim)
            logits = model(X_batch) # (B, Nodes, Classes)
            
            # Select active nodes
            # logits[b, nodes_batch[b], :]
            # We can use gather
            # nodes_batch: (B, ) -> (B, 1, Classes)
            # But simpler:
            selected_logits = logits[torch.arange(logits.size(0)), nodes_batch]
            
            loss = criterion(selected_logits, y_batch)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            preds = torch.argmax(selected_logits, dim=1)
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
            for X_batch, y_batch, nodes_batch in val_loader:
                logits = model(X_batch)
                selected_logits = logits[torch.arange(logits.size(0)), nodes_batch]
                loss = criterion(selected_logits, y_batch)
                val_loss += loss.item()
                preds = torch.argmax(selected_logits, dim=1)
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(y_batch.cpu().numpy())
        
        val_loss /= len(val_loader)
        val_acc = accuracy_score(val_labels, val_preds)
        val_f1 = f1_score(val_labels, val_preds, average='macro')
        
        print(f"Epoch {epoch+1}/{args.n_epoch} | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Train F1: {train_f1:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}")
        
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"  New best model saved! F1: {val_f1:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

    # Test
    if os.path.exists(best_model_path):
        print("Loading best model for testing...")
        model.load_state_dict(torch.load(best_model_path))

        
    model.eval()
    test_loss = 0
    test_preds = []
    test_labels = []
    test_probs = []
    with torch.no_grad():
        for X_batch, y_batch, nodes_batch in test_loader:
            logits = model(X_batch)
            selected_logits = logits[torch.arange(logits.size(0)), nodes_batch]
            probs = torch.softmax(selected_logits, dim=1)
            loss = criterion(selected_logits, y_batch)
            test_loss += loss.item()
            preds = torch.argmax(selected_logits, dim=1)
            test_preds.extend(preds.cpu().numpy())
            test_labels.extend(y_batch.cpu().numpy())
            test_probs.extend(probs.cpu().numpy())
            
    test_loss /= len(test_loader)
    
    # Metrics
    y_true = np.array(test_labels)
    y_pred = np.array(test_preds)
    y_prob = np.array(test_probs)
    
    if not class_names:
        class_names = [str(l) for l in unique_labels]
        
    normal_indices = _infer_normal_indices(class_names)
    
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

    print("-" * 30)
    print("Detailed Test Metrics:")
    print(f"Loss: {test_loss:.4f}")
    print(f"ACC: {acc:.4f}")
    print(f"APR (Weighted Precision): {prec:.4f}")
    print(f"RE (Weighted Recall): {rec:.4f}")
    print(f"AUC: {auc:.4f}")
    print(f"F1 (Weighted): {f1:.4f}")
    print(f"F1 (Macro): {f1_m:.4f}")
    print(f"FAR: {far:.4f}")
    print(f"ASA: {asa:.4f}")
    print("-" * 30)
    
    # Confusion Matrix
    cm_count = confusion_matrix(y_true, y_pred, labels=np.arange(num_classes))
    np.save(os.path.join(output_dir, "cm_counts.npy"), cm_count)
    
    _save_confusion_matrix_image(cm_count, class_names, os.path.join(output_dir, 'confusion_matrix.png'), "Confusion Matrix (Counts)")
    print(f"Confusion matrix saved to {os.path.join(output_dir, 'confusion_matrix.png')}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='unsw_nb15', help='Dataset name')
    parser.add_argument('--n_epoch', type=int, default=50, help='Number of epochs')
    parser.add_argument('--bs', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--hidden_dim', type=int, default=64, help='Hidden dimension')
    parser.add_argument('--seq_len', type=int, default=10, help='Sequence length')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience')
    parser.add_argument('--weight_decay', type=float, default=1.5e-3, help='Weight decay')
    
    args = parser.parse_args()
    train_tgcn(args)
