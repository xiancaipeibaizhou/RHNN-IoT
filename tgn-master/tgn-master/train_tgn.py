import math
import logging
import time
import sys
import argparse
import torch
import numpy as np
import os
import pickle
from pathlib import Path
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score, confusion_matrix, accuracy_score, precision_score, recall_score
import matplotlib.pyplot as plt
import seaborn as sns

# Add current directory to sys.path
sys.path.append(str(Path(__file__).parent))

from model.tgn import TGN
from utils.utils import EarlyStopMonitor, RandEdgeSampler, get_neighbor_finder
from utils.data_processing import compute_time_statistics

# Set random seed
torch.manual_seed(0)
np.random.seed(0)

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

class TGNData:
    def __init__(self, sources, destinations, timestamps, edge_idxs, labels):
        self.sources = sources
        self.destinations = destinations
        self.timestamps = timestamps
        self.edge_idxs = edge_idxs
        self.labels = labels
        self.n_interactions = len(sources)
        self.unique_nodes = set(sources) | set(destinations)
        self.n_unique_nodes = len(self.unique_nodes)

def get_pyg_data(dataset_name):
    """
    Load PyG data and convert to TGN format.
    Treats each graph in the list as a single interaction event (self-loop).
    """
    processed_dir = Path('/root/project/reon/processed_data') / dataset_name
    
    print(f"Loading data from {processed_dir}...")
    train_list = torch.load(processed_dir / 'train_graphs.pt', weights_only=False)
    val_list = torch.load(processed_dir / 'val_graphs.pt', weights_only=False)
    test_list = torch.load(processed_dir / 'test_graphs.pt', weights_only=False)
    
    # Load label encoder if available
    label_encoder_path = processed_dir / 'label_encoder.pkl'
    class_names = []
    if label_encoder_path.exists():
        try:
            with open(label_encoder_path, 'rb') as f:
                le = pickle.load(f)
                class_names = le.classes_
            print(f"Loaded class names: {class_names}")
        except Exception as e:
            print(f"Failed to load label encoder with pickle: {e}")
            try:
                import pandas as pd
                le = pd.read_pickle(label_encoder_path)
                class_names = le.classes_
                print(f"Loaded class names with pandas: {class_names}")
            except Exception as e2:
                print(f"Failed to load label encoder with pandas: {e2}")

    # Combine all lists to build node mapping
    all_data = train_list + val_list + test_list
    
    # Build node mapping
    node_map = {}
    next_id = 0
    
    sources = []
    destinations = []
    timestamps = []
    edge_idxs = []
    labels = []
    edge_features_list = []
    
    # Process all data sequentially
    # Assign sequential timestamps: 0, 1, 2...
    # Treat each graph as one event: source = n_id[0], dest = n_id[0] (self-loop)
    # Edge features = cat(x, edge_attr)
    
    for i, data in enumerate(all_data):
        # Get node ID (assume first element of n_id is the target node)
        if data.n_id.dim() == 0:
            raw_id = data.n_id.item()
        else:
            raw_id = data.n_id[0].item()
            
        if raw_id not in node_map:
            node_map[raw_id] = next_id
            next_id += 1
        
        mapped_id = node_map[raw_id]
        
        sources.append(mapped_id)
        destinations.append(mapped_id) # Self-loop
        timestamps.append(float(i)) # Sequential time
        edge_idxs.append(i) # Sequential edge index
        
        # Labels
        if hasattr(data, 'y') and data.y is not None:
             # Handle if y is a tensor
             lbl = data.y.item() if data.y.numel() == 1 else data.y[0].item()
             labels.append(lbl)
        elif hasattr(data, 'edge_labels') and data.edge_labels is not None:
             lbl = data.edge_labels.item() if data.edge_labels.numel() == 1 else data.edge_labels[0].item()
             labels.append(lbl)
        else:
             labels.append(0) # Default
             
        # Features
        # Combine x (node feat) and edge_attr
        # x: [1, 4], edge_attr: [1, 77] -> [81]
        x_feat = data.x[0] if data.x.dim() == 2 else data.x
        e_feat = data.edge_attr[0] if data.edge_attr.dim() == 2 else data.edge_attr
        
        combined_feat = torch.cat([x_feat, e_feat], dim=0).numpy()
        edge_features_list.append(combined_feat)
        
    # Convert to numpy
    sources = np.array(sources)
    destinations = np.array(destinations)
    timestamps = np.array(timestamps)
    edge_idxs = np.array(edge_idxs)
    labels = np.array(labels)
    edge_features = np.stack(edge_features_list)
    
    # Remap labels to 0..K-1
    unique_labels = np.unique(labels)
    label_map = {l: i for i, l in enumerate(unique_labels)}
    labels = np.array([label_map[l] for l in labels])
    print(f"Remapped labels: {unique_labels} -> {np.unique(labels)}")
    
    # Node features: Initialize with zeros or random
    # Since we combined x into edge features, we can use dummy node features
    # But TGN expects node features. Let's use zeros.
    # Size should be compatible with TGN input.
    # Let's use 100-dim zeros as default in TGN arguments, or match edge feature dim?
    # TGN args allow specifying dimensions.
    # We'll use 81 dim for node features too (initialized with zeros)
    num_nodes = next_id
    node_features = np.zeros((num_nodes, edge_features.shape[1]), dtype=np.float32)
    
    # Create Data objects for train/val/test
    num_train = len(train_list)
    num_val = len(val_list)
    num_test = len(test_list)
    
    full_data = TGNData(sources, destinations, timestamps, edge_idxs, labels)
    
    train_data = TGNData(sources[:num_train], destinations[:num_train], 
                         timestamps[:num_train], edge_idxs[:num_train], labels[:num_train])
                         
    val_data = TGNData(sources[num_train:num_train+num_val], 
                       destinations[num_train:num_train+num_val],
                       timestamps[num_train:num_train+num_val],
                       edge_idxs[num_train:num_train+num_val],
                       labels[num_train:num_train+num_val])
                       
    test_data = TGNData(sources[num_train+num_val:], 
                        destinations[num_train+num_val:],
                        timestamps[num_train+num_val:],
                        edge_idxs[num_train+num_val:],
                        labels[num_train+num_val:])
    
    if len(class_names) == 0:
        class_names = [str(i) for i in range(len(unique_labels))]
                        
    return full_data, node_features, edge_features, train_data, val_data, test_data, class_names

def train_tgn(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load data
    full_data, node_features, edge_features, train_data, val_data, test_data, class_names = \
        get_pyg_data(args.dataset)
        
    num_classes = len(np.unique(full_data.labels))
    print(f"Num classes: {num_classes}")
    
    # Determine dimensions from data
    node_dim = node_features.shape[1]
    edge_dim = edge_features.shape[1]
    print(f"Node dim: {node_dim}, Edge dim: {edge_dim}")
    
    # Update args to match data dimensions
    args.node_dim = node_dim
    args.memory_dim = node_dim # Memory dim must match node dim for TGN implementation
    args.message_dim = node_dim # Set message dim to node dim for simplicity
    
    # Initialize TGN
    # Neighbor finder
    train_ngh_finder = get_neighbor_finder(train_data, args.uniform)
    full_ngh_finder = get_neighbor_finder(full_data, args.uniform)
    
    # Model
    tgn = TGN(neighbor_finder=train_ngh_finder, 
              node_features=node_features,
              edge_features=edge_features, 
              device=device,
              n_layers=args.n_layer,
              n_heads=args.n_head, 
              dropout=args.drop_out, 
              use_memory=args.use_memory,
              message_dimension=args.message_dim, 
              memory_dimension=args.memory_dim,
              embedding_module_type=args.embedding_module,
              message_function=args.message_function,
              aggregator_type=args.aggregator,
              memory_updater_type=args.memory_updater,
              n_neighbors=args.n_degree,
              mean_time_shift_src=0, std_time_shift_src=1, 
              mean_time_shift_dst=0, std_time_shift_dst=1,
              use_destination_embedding_in_message=args.use_destination_embedding_in_message,
              use_source_embedding_in_message=args.use_source_embedding_in_message,
              dyrep=args.dyrep)
              
    tgn = tgn.to(device)
    
    # Decoder (MLP for edge classification)
    embedding_dim = node_features.shape[1]
    decoder = torch.nn.Sequential(
        torch.nn.Linear(embedding_dim * 2, embedding_dim),
        torch.nn.ReLU(),
        torch.nn.Linear(embedding_dim, num_classes)
    ).to(device)
    
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(list(tgn.parameters()) + list(decoder.parameters()), 
                                 lr=args.lr)
                                 
    # Output Directory
    run_tag = time.strftime("%Y%m%d-%H%M%S")
    output_dir = os.path.join('baseline', 'TGN', args.dataset, run_tag)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Training Loop
    best_val_f1 = 0
    patience_counter = 0
    best_model_path = os.path.join(output_dir, 'tgn_best_model.pth')
    best_decoder_path = os.path.join(output_dir, 'tgn_best_decoder.pth')
    
    for epoch in range(args.n_epoch):
        tgn.memory.__init_memory__()
        
        tgn.train()
        decoder.train()
        
        tgn.neighbor_finder = train_ngh_finder
        
        num_batch = math.ceil(len(train_data.sources) / args.bs)
        
        total_loss = 0
        preds = []
        true_labels = []
        
        for k in range(num_batch):
            s_idx = k * args.bs
            e_idx = min((k + 1) * args.bs, len(train_data.sources))
            
            batch_src = train_data.sources[s_idx:e_idx]
            batch_dst = train_data.destinations[s_idx:e_idx]
            batch_ts = train_data.timestamps[s_idx:e_idx]
            batch_ei = train_data.edge_idxs[s_idx:e_idx]
            batch_y = torch.from_numpy(train_data.labels[s_idx:e_idx]).long().to(device)
            
            optimizer.zero_grad()
            
            src_emb, dst_emb, _ = tgn.compute_temporal_embeddings(
                batch_src, batch_dst, batch_dst, batch_ts, batch_ei, args.n_degree)
                
            decoder_input = torch.cat([src_emb, dst_emb], dim=1)
            logits = decoder(decoder_input)
            
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            tgn.memory.detach_memory()
            
            total_loss += loss.item()
            preds.append(logits.argmax(dim=1).cpu().numpy())
            true_labels.append(batch_y.cpu().numpy())
            
        train_loss = total_loss / num_batch
        train_preds = np.concatenate(preds)
        train_labels = np.concatenate(true_labels)
        train_acc = accuracy_score(train_labels, train_preds)
        train_f1 = f1_score(train_labels, train_preds, average='macro')
        
        print(f"Epoch {epoch+1}/{args.n_epoch} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Train F1: {train_f1:.4f}")
        
        # Validation
        val_acc, val_f1, val_loss = eval_tgn(tgn, decoder, val_data, full_ngh_finder, device, args, criterion)
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}")
        
        if early_stopper.early_stop_check(val_f1):
            print("Early stopping!")
            break
            
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            
            torch.save(tgn.state_dict(), os.path.join(output_dir, 'tgn_best_model.pth'))
            torch.save(decoder.state_dict(), os.path.join(output_dir, 'tgn_best_decoder.pth'))
            
    # Test
    print("Loading best model for testing...")
    
    tgn.load_state_dict(torch.load(os.path.join(output_dir, 'tgn_best_model.pth')))
    decoder.load_state_dict(torch.load(os.path.join(output_dir, 'tgn_best_decoder.pth')))
    
    test_acc, test_f1, test_loss, test_preds, test_labels, test_probs = eval_tgn(tgn, decoder, test_data, full_ngh_finder, device, args, criterion, return_preds=True)
    
    # Metrics Calculation
    acc = accuracy_score(test_labels, test_preds)
    f1_m = f1_score(test_labels, test_preds, average='macro', zero_division=0)
    precision_w = precision_score(test_labels, test_preds, average='weighted', zero_division=0)
    recall_w = recall_score(test_labels, test_preds, average='weighted', zero_division=0)
    f1_w = f1_score(test_labels, test_preds, average='weighted', zero_division=0)
    
    try:
        if len(np.unique(test_labels)) == 2:
            auc = roc_auc_score(test_labels, test_probs[:, 1])
        else:
            auc = roc_auc_score(test_labels, test_probs, multi_class='ovr', average='macro')
    except:
        auc = 0.0
        
    # ASA & FAR
    normal_indices = _infer_normal_indices(class_names)
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
    print("Detailed Test Results:")
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
    
    # Confusion Matrix (Counts & Percentages)
    cm_count = confusion_matrix(test_labels, test_preds)
    
    _save_confusion_matrix_image(cm_count, class_names, os.path.join(output_dir, 'confusion_matrix.png'), "Confusion Matrix (Counts)")
    print(f"Confusion matrix saved to {os.path.join(output_dir, 'confusion_matrix.png')}")

def eval_tgn(tgn, decoder, data, ngh_finder, device, args, criterion, return_preds=False):
    tgn.eval()
    decoder.eval()
    
    # Important: During validation/test, we must process the stream sequentially from where training left off?
    # TGN memory is persistent during inference.
    # So we should NOT reset memory if we are continuing the stream.
    # But for standard validation, we usually run from start (or from train end).
    # Here, we assume validation follows training.
    # We should use 'full_ngh_finder' which contains all edges (train+val+test) so we can find neighbors from past.
    
    tgn.neighbor_finder = ngh_finder
    
    preds = []
    true_labels = []
    probs_list = []
    total_loss = 0
    num_batch = math.ceil(len(data.sources) / args.bs)
    
    with torch.no_grad():
        for k in range(num_batch):
            s_idx = k * args.bs
            e_idx = min((k + 1) * args.bs, len(data.sources))
            
            batch_src = data.sources[s_idx:e_idx]
            batch_dst = data.destinations[s_idx:e_idx]
            batch_ts = data.timestamps[s_idx:e_idx]
            batch_ei = data.edge_idxs[s_idx:e_idx]
            batch_y = torch.from_numpy(data.labels[s_idx:e_idx]).long().to(device)
            
            src_emb, dst_emb, _ = tgn.compute_temporal_embeddings(
                batch_src, batch_dst, batch_dst, batch_ts, batch_ei, args.n_degree)
                
            decoder_input = torch.cat([src_emb, dst_emb], dim=1)
            logits = decoder(decoder_input)
            probs = torch.softmax(logits, dim=1)
            loss = criterion(logits, batch_y)
            
            total_loss += loss.item()
            preds.append(logits.argmax(dim=1).cpu().numpy())
            true_labels.append(batch_y.cpu().numpy())
            probs_list.append(probs.cpu().numpy())
            
    all_preds = np.concatenate(preds)
    all_labels = np.concatenate(true_labels)
    all_probs = np.concatenate(probs_list)
    
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro')
    avg_loss = total_loss / num_batch
    
    if return_preds:
        return acc, f1, avg_loss, all_preds, all_labels, all_probs
    return acc, f1, avg_loss

if __name__ == '__main__':
    parser = argparse.ArgumentParser('TGN training')
    parser.add_argument('--dataset', type=str, default='unsw_nb15')
    parser.add_argument('--bs', type=int, default=32)
    parser.add_argument('--n_epoch', type=int, default=50)
    parser.add_argument('--n_layer', type=int, default=1)
    parser.add_argument('--n_head', type=int, default=2)
    parser.add_argument('--drop_out', type=float, default=0.1)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--message_dim', type=int, default=100)
    parser.add_argument('--memory_dim', type=int, default=172)
    parser.add_argument('--n_degree', type=int, default=10)
    parser.add_argument('--use_memory', action='store_true', default=True)
    parser.add_argument('--embedding_module', type=str, default="graph_attention")
    parser.add_argument('--message_function', type=str, default="identity")
    parser.add_argument('--memory_updater', type=str, default="gru")
    parser.add_argument('--aggregator', type=str, default="last")
    parser.add_argument('--uniform', action='store_true', default=True)
    parser.add_argument('--use_destination_embedding_in_message', action='store_true', default=False)
    parser.add_argument('--use_source_embedding_in_message', action='store_true', default=False)
    parser.add_argument('--dyrep', action='store_true', default=False)
    
    args = parser.parse_args()
    
    # Create results dir
    Path("results").mkdir(parents=True, exist_ok=True)
    
    train_tgn(args)
