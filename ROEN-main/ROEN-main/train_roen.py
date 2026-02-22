
import argparse
import torch
import torch.nn as nn
import numpy as np
import os
import time
import pickle
from pathlib import Path
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, precision_score, recall_score, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import copy
from tqdm import tqdm
from torch_geometric.nn import GCNConv

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

# --- Modified ROEN Model ---

class ROEN(nn.Module):
    def __init__(self, node_in_channels, edge_in_channels, hidden_channels_node, hidden_channels_edge, mlp_hidden_channels, num_edge_classes, num_nodes=4):
        super(ROEN, self).__init__()
        
        self.num_nodes = num_nodes
        self.num_edges = num_nodes * num_nodes # Fully connected
        
        # Add two MLP layers before GCN for processing node features
        self.mlp_node_fc1 = nn.Linear(node_in_channels, hidden_channels_node)
        self.mlp_node_fc2 = nn.Linear(hidden_channels_node, hidden_channels_node)
        
        # Add two MLP layers before GCN for processing edge features
        self.mlp_edge_fc1 = nn.Linear(edge_in_channels, hidden_channels_edge)
        self.mlp_edge_fc2 = nn.Linear(hidden_channels_edge, hidden_channels_edge)

        # GCN layers for processing node features
        self.gcn_node_layers1 = GCNConv(hidden_channels_node, hidden_channels_node)
        self.gcn_node_layers2 = GCNConv(hidden_channels_node, hidden_channels_node)
        
        # Fully connected layers for processing edge features
        self.edge_fc1 = nn.Linear(hidden_channels_edge, hidden_channels_edge)
        self.edge_fc2 = nn.Linear(hidden_channels_edge, hidden_channels_edge)

        # LSTM for processing temporal information of node features
        # Note: We use batch_first=True, input (Batch, Seq, Features)
        # Here Batch = NumNodes, Seq = SeqLen. 
        # So input shape to LSTM should be (NumNodes, SeqLen, Hidden)
        self.lstm_node = nn.LSTM(input_size=hidden_channels_node, hidden_size=hidden_channels_node, batch_first=True)
        
        # LSTM for processing temporal information of edge features
        # Input shape: (NumEdges, SeqLen, Hidden)
        self.lstm_edge = nn.LSTM(input_size=hidden_channels_edge, hidden_size=hidden_channels_edge, batch_first=True)

        # Final MLP for edge classification
        # Input: Node_Src + Node_Dst + Edge_Feat
        self.mlp_classifier_fc1 = nn.Linear(2 * hidden_channels_node + hidden_channels_edge, mlp_hidden_channels)  
        self.mlp_classifier_fc2 = nn.Linear(mlp_hidden_channels, num_edge_classes)

        self.lstm_hidden_node = hidden_channels_node  
        self.lstm_hidden_edge = hidden_channels_edge
        
        # Fixed Edge Index for Fully Connected Graph (including self loops)
        # We register it as a buffer so it moves to device
        src = []
        dst = []
        for i in range(num_nodes):
            for j in range(num_nodes):
                src.append(i)
                dst.append(j)
        self.register_buffer("edge_index", torch.tensor([src, dst], dtype=torch.long))

    def forward(self, x_seq, edge_attr_seq):
        # x_seq: (Batch, SeqLen, NumNodes, NodeIn)
        # edge_attr_seq: (Batch, SeqLen, NumEdges, EdgeIn)
        
        batch_size, seq_len, num_nodes, _ = x_seq.shape
        _, _, num_edges, _ = edge_attr_seq.shape
        
        # We need to process each time step
        # To batch efficiently, we can combine Batch and SeqLen dimensions?
        # Or process sequentially.
        # Since GCN is spatial, we can process (Batch * SeqLen) graphs in parallel.
        
        # Reshape to (Batch * SeqLen, NumNodes, NodeIn)
        x_flat = x_seq.view(batch_size * seq_len, num_nodes, -1)
        # Reshape to (Batch * SeqLen, NumEdges, EdgeIn)
        edge_attr_flat = edge_attr_seq.view(batch_size * seq_len, num_edges, -1)
        
        # Process MLPs
        x_emb = torch.relu(self.mlp_node_fc1(x_flat))
        x_emb = torch.relu(self.mlp_node_fc2(x_emb))
        
        edge_emb = torch.relu(self.mlp_edge_fc1(edge_attr_flat))
        edge_emb = torch.relu(self.mlp_edge_fc2(edge_emb))
        
        # GCN Processing
        # GCNConv expects (TotalNodes, Dim) and EdgeIndex.
        # Here we have batches of graphs.
        # We can use PyG's batching or just loop if Batch*SeqLen is small?
        # Or reshape to (Batch*SeqLen*NumNodes, Dim) and adjust edge_index.
        
        # Construct Batch Edge Index
        # self.edge_index is (2, NumEdges).
        # We need to replicate it for Batch * SeqLen graphs.
        total_graphs = batch_size * seq_len
        
        # (2, NumEdges) -> (2, NumEdges * TotalGraphs)
        # Offset node indices by GraphIndex * NumNodes
        # This can be precomputed or computed on fly.
        
        # Efficient Batch Edge Index Construction
        base_edge_index = self.edge_index # (2, NumEdges)
        # Offsets: [0, NumNodes, 2*NumNodes, ...]
        offsets = torch.arange(total_graphs, device=x_seq.device) * num_nodes
        offsets = offsets.view(-1, 1, 1) # (TotalGraphs, 1, 1)
        
        # (TotalGraphs, 2, NumEdges)
        batch_edge_index = base_edge_index.unsqueeze(0) + offsets
        # (2, TotalGraphs * NumEdges)
        batch_edge_index = batch_edge_index.permute(1, 0, 2).reshape(2, -1)
        
        # Flatten x to (TotalGraphs * NumNodes, Dim)
        x_batch = x_emb.view(-1, x_emb.size(-1))
        
        # GCN 1
        x_batch = torch.relu(self.gcn_node_layers1(x_batch, batch_edge_index))
        # GCN 2
        x_batch = torch.relu(self.gcn_node_layers2(x_batch, batch_edge_index))
        
        # Edge FC layers
        # (TotalGraphs * NumEdges, Dim)
        edge_batch = edge_emb.view(-1, edge_emb.size(-1))
        edge_batch = torch.relu(self.edge_fc1(edge_batch))
        edge_batch = torch.relu(self.edge_fc2(edge_batch))
        
        # Reshape back to Sequence
        # x_batch: (Batch * SeqLen * NumNodes, Hidden) -> (Batch, SeqLen, NumNodes, Hidden)
        x_out = x_batch.view(batch_size, seq_len, num_nodes, -1)
        # edge_batch: (Batch * SeqLen * NumEdges, Hidden) -> (Batch, SeqLen, NumEdges, Hidden)
        edge_out = edge_batch.view(batch_size, seq_len, num_edges, -1)
        
        # LSTM Processing
        # LSTM expects (Batch, Seq, Feature).
        # We want to run LSTM for each Node (across time) and each Edge (across time).
        # So we treat (Batch * NumNodes) as the LSTM Batch dimension.
        
        # Permute x_out to (Batch, NumNodes, SeqLen, Hidden)
        x_lstm_in = x_out.permute(0, 2, 1, 3).reshape(batch_size * num_nodes, seq_len, -1)
        
        # Permute edge_out to (Batch, NumEdges, SeqLen, Hidden)
        edge_lstm_in = edge_out.permute(0, 2, 1, 3).reshape(batch_size * num_edges, seq_len, -1)
        
        # Run LSTM
        # Outputs: (Batch*NumNodes, SeqLen, Hidden)
        lstm_out_node, _ = self.lstm_node(x_lstm_in)
        lstm_out_edge, _ = self.lstm_edge(edge_lstm_in)
        
        # Reshape back to (Batch, NumNodes, SeqLen, Hidden)
        lstm_out_node = lstm_out_node.view(batch_size, num_nodes, seq_len, -1)
        # Reshape back to (Batch, NumEdges, SeqLen, Hidden)
        lstm_out_edge = lstm_out_edge.view(batch_size, num_edges, seq_len, -1)
        
        # Permute back to (Batch, SeqLen, NumNodes, Hidden)
        lstm_out_node = lstm_out_node.permute(0, 2, 1, 3)
        lstm_out_edge = lstm_out_edge.permute(0, 2, 1, 3)
        
        # Classification
        # We need to classify each edge at each time step (or just last step).
        # We'll compute for all steps.
        
        # Construct Edge Features for Classification
        # For each edge (u, v), we need Node_u, Node_v, and Edge_uv features.
        
        # Use edge_index to gather node features.
        # self.edge_index is (2, NumEdges).
        src_indices = self.edge_index[0] # (NumEdges,)
        dst_indices = self.edge_index[1] # (NumEdges,)
        
        # Gather src node features: (Batch, SeqLen, NumEdges, Hidden)
        # We expand node features to edges.
        src_features = lstm_out_node[:, :, src_indices, :]
        dst_features = lstm_out_node[:, :, dst_indices, :]
        
        # Concatenate: (Batch, SeqLen, NumEdges, 2*NodeHidden + EdgeHidden)
        final_features = torch.cat([src_features, dst_features, lstm_out_edge], dim=-1)
        
        # MLP Classifier
        preds = torch.relu(self.mlp_classifier_fc1(final_features))
        logits = self.mlp_classifier_fc2(preds) # (Batch, SeqLen, NumEdges, NumClasses)
        
        return logits

# --- Data Loading ---

def get_data_roen(dataset_name, seq_len=10):
    processed_dir = Path('/root/project/reon/processed_data') / dataset_name
    print(f"Loading data from {processed_dir}...")
    
    train_list = torch.load(processed_dir / 'train_graphs.pt', weights_only=False)
    val_list = torch.load(processed_dir / 'val_graphs.pt', weights_only=False)
    test_list = torch.load(processed_dir / 'test_graphs.pt', weights_only=False)
    
    all_data = train_list + val_list + test_list
    node_map = {}
    next_id = 0
    for data in all_data:
        if data.n_id.dim() == 0: raw_id = data.n_id.item()
        else: raw_id = data.n_id[0].item()
        if raw_id not in node_map:
            node_map[raw_id] = next_id
            next_id += 1
    num_nodes = len(node_map)
    print(f"Total unique nodes: {num_nodes}")
    
    # We assume fully connected graph
    num_edges = num_nodes * num_nodes
    
    # Map (src, dst) to edge index in our dense representation (0..NumEdges-1)
    # Since we construct edges by looping i then j, index = i * NumNodes + j
    
    def process_split(graph_list, seq_len):
        X_seq = []
        E_seq = []
        Y_seq = [] # Target labels (for the active edge)
        Mask_seq = [] # Mask to indicate which edge is active
        
        events = []
        for data in graph_list:
            if data.n_id.dim() == 0: raw_id = data.n_id.item()
            else: raw_id = data.n_id[0].item()
            node_idx = node_map[raw_id]
            
            # Node Feature (Active Node)
            # We only have features for the active node?
            # T-GCN assumed so. Let's assume other nodes have zero features?
            # Or use last known? Zero for now (impulse).
            x_feat = data.x.mean(dim=0) if data.x.shape[0] > 0 else torch.zeros(4)
            
            # Edge Feature (Active Edge)
            # data.edge_index has the active edge(s).
            e_feat = data.edge_attr.mean(dim=0) if (data.edge_attr is not None and data.edge_attr.shape[0] > 0) else torch.zeros(77)
            
            # Label
            if hasattr(data, 'y') and data.y is not None:
                lbl = data.y.item() if data.y.numel() == 1 else data.y[0].item()
            elif hasattr(data, 'edge_labels') and data.edge_labels is not None:
                lbl = data.edge_labels.item() if data.edge_labels.numel() == 1 else data.edge_labels[0].item()
            else:
                lbl = 0
            
            # Identify active edge indices
            # data.edge_index contains edges in LOCAL node IDs (0..K).
            # But wait, T-GCN remapped raw_id to global node_idx.
            # Here, the graph structure in `data` might be local.
            # We need to map `data.edge_index` to global nodes.
            # But `data` usually contains 1-hop neighborhood or just the edge.
            # If `data` is just one edge, then `data.edge_index` is [[0], [1]].
            # And `data.x` has 2 rows.
            # We need the GLOBAL IDs of these nodes.
            # `data.n_id` usually contains global IDs if it's a subgraph.
            # If `data.n_id` is a scalar (center node), then where are the neighbors?
            # PyG `NeighborLoader` returns subgraphs where `n_id` is the mapping to global.
            # But here `data.n_id` seems to be the center node ID (scalar or 1-element list).
            # This implies `data` is centered on ONE node.
            # And `edge_index` connects this node to neighbors?
            # Or is it a self-loop?
            
            # Let's assume the event is: Node `node_idx` is active.
            # And it interacts with itself? Or neighbors?
            # T-GCN ignored topology and just used `node_idx`.
            # ROEN needs edge features.
            # If `data.edge_index` is present, use it.
            # If `data` has mapping to global IDs, we use it.
            # If not, we can only map the center node.
            
            # Heuristic:
            # Assume `data.x` corresponds to `node_idx`.
            # Assume `data.edge_index` connects `node_idx` to `node_idx` (self-loop) if no other info.
            # OR, if `data` has multiple nodes, we need their global IDs.
            # But `train_tgcn.py` only extracted `node_idx` from `n_id`.
            
            # Let's check `data` structure from T-GCN run:
            # "Total unique nodes: 4".
            # "node x shape [1,4]".
            # This implies `data` has only 1 node!
            # So `edge_index` must be empty or self-loop?
            # If `edge_attr` exists (shape [1,77]), there MUST be an edge.
            # So it must be a self-loop `(0, 0)` in local indices.
            # Which maps to `(node_idx, node_idx)` in global indices.
            
            active_edges = []
            if data.edge_index is not None and data.edge_index.size(1) > 0:
                # Assuming local index 0 corresponds to `node_idx`.
                # If there are other nodes, we don't know their IDs.
                # But since `x` is [1, 4], there is only 1 node.
                # So it IS a self-loop.
                active_edges.append((node_idx, node_idx))
            else:
                # If no edge index but edge attr exists?
                # Assume self-loop on center node.
                active_edges.append((node_idx, node_idx))
            
            events.append({
                'node_idx': node_idx,
                'x_feat': x_feat,
                'e_feat': e_feat,
                'active_edges': active_edges,
                'label': lbl
            })

        # Create windows
        num_samples = len(events) - seq_len + 1
        if num_samples <= 0: return None, None, None, None
        
        for i in range(num_samples):
            window = events[i : i+seq_len]
            
            # Prepare tensors for this window
            # X: (SeqLen, NumNodes, 4)
            X_w = torch.zeros(seq_len, num_nodes, 4)
            # E: (SeqLen, NumEdges, 77)
            E_w = torch.zeros(seq_len, num_edges, 77)
            # Mask: (SeqLen, NumEdges)
            Mask_w = torch.zeros(seq_len, num_edges)
            # Y: (SeqLen, NumEdges) - though we only care about active ones
            Y_w = torch.zeros(seq_len, num_edges, dtype=torch.long)
            
            for t, ev in enumerate(window):
                # Set Node Feature
                n_idx = ev['node_idx']
                X_w[t, n_idx, :] = ev['x_feat']
                
                # Set Edge Features
                e_feat = ev['e_feat']
                lbl = ev['label']
                
                for (u, v) in ev['active_edges']:
                    # Global edge index = u * NumNodes + v
                    e_idx = u * num_nodes + v
                    E_w[t, e_idx, :] = e_feat
                    Mask_w[t, e_idx] = 1.0
                    Y_w[t, e_idx] = lbl
            
            X_seq.append(X_w)
            E_seq.append(E_w)
            Mask_seq.append(Mask_w)
            Y_seq.append(Y_w)
            
        return torch.stack(X_seq), torch.stack(E_seq), torch.stack(Mask_seq), torch.stack(Y_seq)

    
    # Load label encoder if available
    label_encoder_path = processed_dir / 'label_encoder.pkl'
    class_names = []
    if label_encoder_path.exists():
        try:
            with open(label_encoder_path, 'rb') as f:
                le = pickle.load(f)
                class_names = list(le.classes_)
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
        # Fallback handled in main loop if needed, but get_data_roen doesn't return num_classes explicitly (it calculates it).
        # We'll handle fallback in train_roen.
        pass

    tr_X, tr_E, tr_M, tr_Y = process_split(train_list, seq_len)
    va_X, va_E, va_M, va_Y = process_split(val_list, seq_len)
    te_X, te_E, te_M, te_Y = process_split(test_list, seq_len)
    
    return (tr_X, tr_E, tr_M, tr_Y), (va_X, va_E, va_M, va_Y), (te_X, te_E, te_M, te_Y), num_nodes, raw_class_names

# --- Main Training Loop ---

def train_roen(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    (tr_X, tr_E, tr_M, tr_Y), (va_X, va_E, va_M, va_Y), (te_X, te_E, te_M, te_Y), num_nodes, raw_class_names = \
        get_data_roen(args.dataset, args.seq_len)
        
    print(f"Train size: {len(tr_X)}")
    
    # Remap labels
    # Collect all valid labels from masked positions
    all_labels_list = []
    for Y, M in [(tr_Y, tr_M), (va_Y, va_M), (te_Y, te_M)]:
        valid_lbls = Y[M == 1]
        all_labels_list.append(valid_lbls)
    
    all_labels = torch.cat(all_labels_list)
    unique_labels = torch.unique(all_labels).sort()[0]
    
    # Resolve class names
    active_class_names = []
    if class_names:
        # Map unique_labels (which are indices in class_names) to names
        try:
            active_class_names = [str(class_names[int(l)]) for l in unique_labels]
        except IndexError:
            print("Warning: Unique label index out of range for class_names. Fallback to indices.")
            active_class_names = [str(int(l)) for l in unique_labels]
    else:
        active_class_names = [str(int(l)) for l in unique_labels]
    
    # Update class_names variable to be used later for plotting
    class_names = active_class_names
    print(f"Active Class Names: {class_names}")

    label_map = {int(l): i for i, l in enumerate(unique_labels)}
    num_classes = len(unique_labels)
    print(f"Num classes: {num_classes}")
    print(f"Label Map: {label_map}")
    
    # Helper to remap
    def remap_labels(Y):
        # Y is tensor. We use a lookup tensor if values are small.
        # Assuming max label is not too huge.
        if unique_labels.numel() == 0: return Y
        max_val = int(unique_labels.max().item())
        lookup = torch.zeros(max_val + 1, dtype=torch.long)
        # Default to 0 or -1? 
        # Since we only care about M=1 positions, the value at M=0 doesn't matter much 
        # as long as it doesn't crash lookup.
        # But if we use lookup[Y], Y at M=0 must be valid index.
        # Y is initialized to 0. So 0 must be in range.
        
        for k, v in label_map.items():
            lookup[k] = v
            
        return lookup[Y.long()]

    tr_Y = remap_labels(tr_Y)
    va_Y = remap_labels(va_Y)
    te_Y = remap_labels(te_Y)
    
    # Dataset
    train_dataset = torch.utils.data.TensorDataset(tr_X, tr_E, tr_M, tr_Y)
    val_dataset = torch.utils.data.TensorDataset(va_X, va_E, va_M, va_Y)
    test_dataset = torch.utils.data.TensorDataset(te_X, te_E, te_M, te_Y)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.bs, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.bs, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.bs, shuffle=False)
    
    # Model
    model = ROEN(node_in_channels=4, 
                 edge_in_channels=77, 
                 hidden_channels_node=args.hidden_dim, 
                 hidden_channels_edge=args.hidden_dim, 
                 mlp_hidden_channels=args.hidden_dim, 
                 num_edge_classes=num_classes,
                 num_nodes=num_nodes).to(device)
                 
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss(reduction='none') # We will mask it manually
    
    # Standardized timestamped output directory
    run_tag = time.strftime("%Y%m%d-%H%M%S")
    output_dir = os.path.join('baseline', 'ROEN', args.dataset, run_tag)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    best_val_f1 = 0
    patience_counter = 0
    best_model_path = os.path.join(output_dir, 'roen_best_model.pth')
    
    for epoch in range(args.n_epoch):
        model.train()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        for X, E, M, Y in train_loader:
            X, E, M, Y = X.to(device), E.to(device), M.to(device), Y.to(device)
            
            optimizer.zero_grad()
            logits = model(X, E) # (Batch, Seq, NumEdges, NumClasses)
            
            # We only care about the LAST step? Or all steps?
            # Typically for sequence classification, we care about the last step.
            # But the mask might have events at any step.
            # T-GCN used last step.
            # Let's use last step for classification.
            
            last_logits = logits[:, -1, :, :] # (Batch, NumEdges, NumClasses)
            last_Y = Y[:, -1, :] # (Batch, NumEdges)
            last_M = M[:, -1, :] # (Batch, NumEdges)
            
            # Flatten to compute loss on active edges only
            loss_unreduced = criterion(last_logits.reshape(-1, num_classes), last_Y.reshape(-1))
            mask_flat = last_M.reshape(-1)
            
            loss = (loss_unreduced * mask_flat).sum() / (mask_flat.sum() + 1e-8)
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
            # Metrics (Active edges only)
            active_indices = mask_flat.bool()
            if active_indices.any():
                preds = torch.argmax(last_logits.reshape(-1, num_classes), dim=1)
                all_preds.extend(preds[active_indices].cpu().numpy())
                all_labels.extend(last_Y.reshape(-1)[active_indices].cpu().numpy())
        
        train_loss = total_loss / len(train_loader)
        train_acc = accuracy_score(all_labels, all_preds) if all_labels else 0
        train_f1 = f1_score(all_labels, all_preds, average='macro') if all_labels else 0
        
        # Validation
        model.eval()
        val_loss = 0
        val_preds = []
        val_labels = []
        with torch.no_grad():
            for X, E, M, Y in val_loader:
                X, E, M, Y = X.to(device), E.to(device), M.to(device), Y.to(device)
                logits = model(X, E)
                last_logits = logits[:, -1, :, :]
                last_Y = Y[:, -1, :]
                last_M = M[:, -1, :]
                
                loss_unreduced = criterion(last_logits.reshape(-1, num_classes), last_Y.reshape(-1))
                mask_flat = last_M.reshape(-1)
                loss = (loss_unreduced * mask_flat).sum() / (mask_flat.sum() + 1e-8)
                val_loss += loss.item()
                
                active_indices = mask_flat.bool()
                if active_indices.any():
                    preds = torch.argmax(last_logits.reshape(-1, num_classes), dim=1)
                    val_preds.extend(preds[active_indices].cpu().numpy())
                    val_labels.extend(last_Y.reshape(-1)[active_indices].cpu().numpy())

        val_loss /= len(val_loader)
        val_acc = accuracy_score(val_labels, val_preds) if val_labels else 0
        val_f1 = f1_score(val_labels, val_preds, average='macro') if val_labels else 0
        
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
    test_preds = []
    test_labels = []
    test_probs = []
    with torch.no_grad():
        for X, E, M, Y in test_loader:
            X, E, M, Y = X.to(device), E.to(device), M.to(device), Y.to(device)
            logits = model(X, E)
            last_logits = logits[:, -1, :, :]
            last_Y = Y[:, -1, :]
            last_M = M[:, -1, :]
            
            # Calculate probabilities
            probs = torch.softmax(last_logits.reshape(-1, num_classes), dim=1)
            
            active_indices = last_M.reshape(-1).bool()
            if active_indices.any():
                preds = torch.argmax(last_logits.reshape(-1, num_classes), dim=1)
                test_preds.extend(preds[active_indices].cpu().numpy())
                test_labels.extend(last_Y.reshape(-1)[active_indices].cpu().numpy())
                test_probs.extend(probs[active_indices].cpu().numpy())
                
    test_acc = accuracy_score(test_labels, test_preds)
    test_f1 = f1_score(test_labels, test_preds, average='macro', zero_division=0)
    
    # New metrics
    test_labels = np.array(test_labels)
    test_preds = np.array(test_preds)
    test_probs = np.array(test_probs)
    
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
        # False Alarm: Predicted attack (not normal) when true is normal
        pred_attacks = np.isin(test_preds[normal_mask], normal_indices, invert=True)
        false_alarms = np.sum(pred_attacks)
        far = false_alarms / np.sum(normal_mask)
    else:
        far = 0.0

    print("-" * 30)
    print("Detailed Test Results:")
    print(f"Accuracy: {test_acc:.4f}")
    print(f"APR (Weighted Precision): {precision_w:.4f}")
    print(f"RE (Weighted Recall): {recall_w:.4f}")
    print(f"AUC: {auc:.4f}")
    print(f"F1 (Weighted): {f1_w:.4f}")
    print(f"F1 Score (Macro): {test_f1:.4f}")
    print(f"FAR: {far:.4f}")
    print(f"ASA: {asa:.4f}")
    print("-" * 30)
    
    # Confusion Matrix (Counts & Percentages)
    cm_count = confusion_matrix(test_labels, test_preds)
    cm_norm = cm_count.astype('float') / cm_count.sum(axis=1)[:, np.newaxis]
    
    _save_confusion_matrix_image(cm_count, class_names, os.path.join(output_dir, 'confusion_matrix.png'), "Confusion Matrix (Counts)")
    print(f"Confusion matrix saved to {os.path.join(output_dir, 'confusion_matrix.png')}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='unsw_nb15')
    parser.add_argument('--n_epoch', type=int, default=50)
    parser.add_argument('--bs', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--seq_len', type=int, default=10)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    
    args = parser.parse_args()
    train_roen(args)
