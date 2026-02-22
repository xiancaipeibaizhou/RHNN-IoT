import os
import numpy as np
import joblib

def load_and_preprocess(data_dir, dataset_name, fs_sample_size=None, test_ratio=None, random_state=None):
    # Map friendly names to folder names if necessary
    name_map = {
        "CIC-IDS2017": "cic_ids2017",
        "UNSW-NB15": "unsw_nb15",
        "Darknet2020": "darknet2020_block",
        "ISCX-IDS2012": "iscx_ids2012"
    }
    
    folder_name = name_map.get(dataset_name, dataset_name)
    
    # Try absolute path first
    base_path = "/root/project/reon/processed_data"
    dataset_path = os.path.join(base_path, folder_name)
    
    if not os.path.exists(dataset_path):
        # Try relative path
        dataset_path = os.path.join("processed_data", folder_name)
    
    if not os.path.exists(dataset_path):
         # Try looking in parent directory
        dataset_path = os.path.join("..", "processed_data", folder_name)

    npz_path = os.path.join(dataset_path, "flattened_data.npz")
    le_path = os.path.join(dataset_path, "label_encoder.pkl")
    
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"Dataset {dataset_name} not found at {npz_path}")
        
    print(f"Loading {dataset_name} from {npz_path}...")
    data = np.load(npz_path)
    X_train = data['X_train']
    y_train = data['y_train']
    X_test = data['X_test']
    y_test = data['y_test']
    X_val = data['X_val'] if 'X_val' in data else None
    y_val = data['y_val'] if 'y_val' in data else None
    
    le = None
    class_names = []
    if os.path.exists(le_path):
        try:
            le = joblib.load(le_path)
            if hasattr(le, 'classes_'):
                class_names = le.classes_
        except:
            print("Warning: Could not load label encoder.")
    
    if len(class_names) == 0:
        if dataset_name == "ISCX-IDS2012":
            class_names = ["Normal", "Attack"]
        else:
            unique = np.unique(np.concatenate([y_train, y_test]))
            class_names = [str(x) for x in unique]
        
    # Return values matching original EBAO1 signature:
    # _, _, X_train, y_train, X_test, y_test, _, le, class_names
    # We use the first two slots for X_val, y_val if available
    return X_val, y_val, X_train, y_train, X_test, y_test, None, le, class_names

BENIGN_SETS = {}
