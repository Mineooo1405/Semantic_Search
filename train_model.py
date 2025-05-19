import os
import sys
import json
import random
import traceback
from datetime import datetime as DateTime

import numpy as np
import pandas as pd
import torch
import matchzoo as mz
from matchzoo.embedding import load_from_file # Changed import
# matchzoo.utils.EarlyStopping and AverageMeter are used internally by mz.Trainer

# --- Global Configuration & Constants ---
SEED = 42 # Seed for reproducibility
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    # For full reproducibility, though might impact performance slightly
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

# --- Helper Functions ---
def jsonify_value(value):
    if isinstance(value, mz.tasks.Ranking):
        return {
            "task_type": "Ranking",
            "loss": value.loss.__class__.__name__ if value.loss else None,
            "metrics": [m.__class__.__name__ for m in value.metrics] if value.metrics else []
        }
    elif isinstance(value, np.ndarray):
        return f"<numpy_array shape={value.shape} dtype={value.dtype}>"
    elif isinstance(value, mz.ParamTable):
        return {str(k): jsonify_value(v) for k, v in value.items()}
    elif isinstance(value, mz.losses.Loss):
        loss_config = {}
        if hasattr(value, 'get_config'):
            loss_config = value.get_config()
        if hasattr(value, 'margin'): 
            loss_config['margin'] = jsonify_value(value.margin)
        if hasattr(value, 'num_neg'):
            loss_config['num_neg'] = jsonify_value(value.num_neg)
        return {
            "loss_class": value.__class__.__name__,
            "config": loss_config
        }
    elif isinstance(value, mz.metrics.Metric):
        return value.__class__.__name__
    elif isinstance(value, (list, tuple)):
        return [jsonify_value(item) for item in value]
    elif isinstance(value, dict):
        return {str(k): jsonify_value(v) for k, v in value.items()}
    elif torch.is_tensor(value):
        return f"<torch_tensor shape={value.shape} dtype={value.dtype} device={value.device}>"
    elif isinstance(value, type):
        return value.__name__
    elif isinstance(value, (int, float, str, bool)) or value is None:
        return value
    elif isinstance(value, np.integer):
        return int(value)
    elif isinstance(value, np.floating):
        return float(value)
    elif isinstance(value, DateTime):
        return value.isoformat()
    elif any(lib_name in str(type(value)).lower() for lib_name in ['torch', 'matchzoo', 'numpy']):
        return f"Object_Type:{type(value).__name__}"
    try:
        return str(value)
    except Exception:
        return f"Unserializable_Object_Type:{type(value).__name__}"

def get_validated_train_data_path():
    """
    Placeholder: User MUST ensure this function is correctly defined.
    It should return the absolute path to the training data file
    and a string identifier for the data chunk/type for folder naming.
    Example: return "/path/to/train_triplets.tsv", "ms_marco_data"
    """
    # Attempting to use 'train.tsv' from the workspace root as a default.
    # PLEASE VERIFY AND ADJUST THIS PATH AND CHUNK TYPE.
    workspace_root = "D:/SemanticSearch" # Assuming this is the workspace root
    default_train_path = os.path.join(workspace_root, "TrainingData_MatchZoo_BEIR/msmarco_semantic-grouping/train_2/msmarco_semantic-grouping_train_triplets.tsv")
    default_chunk_type = "train_tsv_data"

    if os.path.exists(default_train_path):
        print(f"Using default training data path: {default_train_path}")
        return default_train_path, default_chunk_type
    else:
        print(f"WARNING: Default training data path {default_train_path} not found.")
        print("Please ensure get_validated_train_data_path() returns correct values.")
        # Fallback to a dummy path to allow script structure to be generated
        # User MUST ensure this path is correct for the script to run.
        return "PLEASE_SET_CORRECT_PATH/train_data.tsv", "default_chunk"

# --- Global Configurations ---
_validated_train_data_path, chunk_type_for_folder = get_validated_train_data_path()
TRAIN_DATA_PATH = _validated_train_data_path

GLOVE_100D_PATH = "D:/SemanticSearch/embedding/glove.6B/glove.6B.100d.txt"
GLOVE_300D_PATH = "D:/SemanticSearch/embedding/glove.6B/glove.6B.300d.txt"
GLOVE_PATHS = {
    "100D": GLOVE_100D_PATH,
    "300D": GLOVE_300D_PATH
}

MODELS_TO_TRAIN = ["KNRM", "MatchLSTM", "ArcI", "DRMM", "MatchPyramid", "ConvKNRM"]

current_date_str = DateTime.now().strftime("%Y%m%d")
base_trained_models_dir = "D:/SemanticSearch/TrainedModels" # User's original path

GENERAL_BATCH_SIZE = 64 
GENERAL_EPOCHS = 10     
GENERAL_LEARNING_RATE = 1e-3
HINGE_MARGIN = 1.0
GENERAL_PATIENCE = 10 
GENERAL_EARLY_STOPPING_KEY = 'ndcg@3' # Make sure this metric is calculated
GENERAL_VALIDATE_INTERVAL = None # Validate every epoch by default
GENERAL_CLIP_NORM = None # No gradient clipping by default

def get_mz_loss_class(class_name_str):
    if hasattr(mz.losses, class_name_str):
        return getattr(mz.losses, class_name_str)
    raise ValueError(f"Loss class {class_name_str} not found in matchzoo.losses")

def get_torch_optimizer_class(class_name_str):
    if hasattr(torch.optim, class_name_str):
        return getattr(torch.optim, class_name_str)
    raise ValueError(f"Optimizer class {class_name_str} not found in torch.optim")

ALL_MODEL_CONFIGS = {
    "KNRM": {
        "batch_size": GENERAL_BATCH_SIZE, "epochs": GENERAL_EPOCHS, "learning_rate": GENERAL_LEARNING_RATE, 
        "patience": GENERAL_PATIENCE, "early_stopping_key": GENERAL_EARLY_STOPPING_KEY,
        "loss_fn_class_name": "RankHingeLoss", "loss_params": {"margin": HINGE_MARGIN},
        "optimizer_class_name": "Adadelta", "optimizer_params": {}, # Changed to Adadelta
        "clip_norm": GENERAL_CLIP_NORM, "validate_interval": GENERAL_VALIDATE_INTERVAL,
        "embedding_dim": 100, "glove_path_key": "100D",
        "fixed_length_left_hint": 10, "fixed_length_right_hint": 40,
        "hyperparameters": {
            'kernel_num': 11,
            'sigma': 0.1,
            'exact_sigma': 0.001
        },
        "preprocessor_type": "BasicPreprocessor", "preprocessor_params": {"filter_low_freq": 2},
        "dataset_mode": "point", "dataset_num_dup": 1, "dataset_num_neg": 0 
    },
    "MatchLSTM": {
        "model_class_name": "MatchLSTM", 
        "fixed_length_left_hint": 30, "fixed_length_right_hint": 100,
        "embedding_dim": 300, "glove_path_key": "300D", # Changed embedding
        "batch_size": GENERAL_BATCH_SIZE, "epochs": GENERAL_EPOCHS, "learning_rate": GENERAL_LEARNING_RATE,
        "patience": 5, "early_stopping_key": "map", 
        "loss_fn_class_name": "RankCrossEntropyLoss", "loss_params": {"num_neg": 10}, # Changed loss
        "optimizer_class_name": "Adadelta", "optimizer_params": {}, # Changed optimizer
        "preprocessor_type": "ModelDefault", "preprocessor_params": {},
        "dataset_mode": "pair", "dataset_num_dup": 5, "dataset_num_neg": 10, # Changed dataset params
        "hyperparameters": {
            'rnn_type': 'lstm',
            'hidden_size': 200,
            'lstm_layer': 1,
            'dropout': 0.2,
            'drop_lstm': False,
            'concat_lstm': True,
            'mask_value': 0  # Added mask_value
        }
    },
    "ArcI": {
        "batch_size": GENERAL_BATCH_SIZE, "epochs": GENERAL_EPOCHS, "learning_rate": GENERAL_LEARNING_RATE,
        "patience": GENERAL_PATIENCE, "early_stopping_key": GENERAL_EARLY_STOPPING_KEY,
        "loss_fn_class_name": "RankHingeLoss", "loss_params": {"margin": HINGE_MARGIN}, # Changed loss
        "optimizer_class_name": "Adadelta", "optimizer_params": {}, # Changed optimizer
        "clip_norm": GENERAL_CLIP_NORM, "validate_interval": GENERAL_VALIDATE_INTERVAL,
        "embedding_dim": 300, "glove_path_key": "300D",
        "fixed_length_left_hint": 10,
        "fixed_length_right_hint": 100,
        "hyperparameters": { # Updated to strictly follow text examples
            'left_filters': [32],
            'left_kernel_sizes': [3],
            'left_pool_sizes': [2], # Text example: [2]
            'right_filters': [32],
            'right_kernel_sizes': [3],
            'right_pool_sizes': [4],
            'conv_activation_func': 'relu',
            'mlp_num_layers': 1,
            'mlp_num_units': 64,    # Text example: 64
            'mlp_num_fan_out': 32,  # Text example: 32
            'mlp_activation_func': 'relu',
            'dropout_rate': 0.5     # Text example: 0.5
        },
        "preprocessor_type": "ModelDefault", "preprocessor_params": {}, 
        "dataset_mode": "pair", "dataset_num_dup": 2, "dataset_num_neg": 1 # num_dup/neg from notebook
    },
    "DRMM": {
        "model_class_name": "DRMM",
        "fixed_length_left_hint": 30, "fixed_length_right_hint": 30,
        "embedding_dim": 300, "glove_path_key": "300D",
        "batch_size": 32, "epochs": GENERAL_EPOCHS, "learning_rate": GENERAL_LEARNING_RATE,
        "patience": 5, "early_stopping_key": "map", 
        "loss_fn_class_name": "RankCrossEntropyLoss", "loss_params": {'num_neg': 10},
        "optimizer_class_name": "Adadelta", "optimizer_params": {},
        "preprocessor_type": "BasicPreprocessor", "preprocessor_params": {}, # Changed: DRMM uses BasicPreprocessor in v1.1.1
        "dataset_mode": "pair", "dataset_num_dup": 5, "dataset_num_neg": 10,
        "hyperparameters": {
            'hist_bin_size': 30, # Moved: hist_bin_size is a model hyperparameter
            'mlp_num_layers': 1,
            'mlp_num_units': 5,
            'mlp_num_fan_out': 1,
            'mlp_activation_func': 'tanh',
            'mask_value': 0
        }
    },
    "MatchPyramid": {
        "model_class_name": "MatchPyramid",
        "fixed_length_left_hint": 30, "fixed_length_right_hint": 30,
        "embedding_dim": 300, "glove_path_key": "300D", # Changed embedding
        "batch_size": GENERAL_BATCH_SIZE, "epochs": GENERAL_EPOCHS, "learning_rate": GENERAL_LEARNING_RATE,
        "patience": 5, "early_stopping_key": "map", 
        "loss_fn_class_name": "RankCrossEntropyLoss", "loss_params": {"num_neg": 1}, # Changed loss
        "optimizer_class_name": "Adam", "optimizer_params": {}, # Adam is correct per notebook
        "preprocessor_type": "ModelDefault", 
        "preprocessor_params": {}, 
        "dataset_mode": "pair", "dataset_num_dup": 2, "dataset_num_neg": 1, # Changed dataset_num_dup
        "hyperparameters": { # Updated dropout
            'kernel_count': [16, 32],
            'kernel_size': [[3, 3], [3, 3]],
            'dpool_size': [3, 10],
            'activation': 'relu',
            'dropout_rate': 0.1  # From notebook example
        }
    },
    "ConvKNRM": {
        "batch_size": GENERAL_BATCH_SIZE, "epochs": GENERAL_EPOCHS, "learning_rate": GENERAL_LEARNING_RATE,
        "patience": GENERAL_PATIENCE, "early_stopping_key": GENERAL_EARLY_STOPPING_KEY,
        "loss_fn_class_name": "RankHingeLoss", "loss_params": {"margin": HINGE_MARGIN}, # Changed loss
        "optimizer_class_name": "Adadelta", "optimizer_params": {}, # Changed optimizer
        "clip_norm": GENERAL_CLIP_NORM, "validate_interval": GENERAL_VALIDATE_INTERVAL,
        "embedding_dim": 300, "glove_path_key": "300D",
        "fixed_length_left_hint": 50, 
        "fixed_length_right_hint": 100,
        "hyperparameters": { # Already aligned with text example
            'filters': 128,
            'conv_activation_func': 'tanh',
            'max_ngram': 3,
            'use_crossmatch': True,
            'kernel_num': 11,
            'sigma': 0.1,
            'exact_sigma': 0.001
        },
        "preprocessor_type": "ModelDefault", "preprocessor_params": {},
        "dataset_mode": "pair", "dataset_num_dup": 5, "dataset_num_neg": 1 # Changed dataset_num_dup
    }
}

# --- Load Data Once ---
print(f"Loading training data from: {TRAIN_DATA_PATH}")
raw_data_triplets = []
try:
    with open(TRAIN_DATA_PATH, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f):
            parts = line.strip().split('\t')
            if len(parts) == 3: # query, positive_doc, negative_doc
                raw_data_triplets.append((parts[0], parts[1], parts[2]))
            else:
                print(f"Warning: Line {line_num+1} in {TRAIN_DATA_PATH} is malformed, expected 3 tab-separated parts: {line.strip()}")
except FileNotFoundError:
    print(f"FATAL: Training data file not found at {TRAIN_DATA_PATH}. Please check the path. Exiting.")
    sys.exit(1)
except Exception as e_load_data:
    print(f"FATAL: Error reading training data file {TRAIN_DATA_PATH}: {e_load_data}. Exiting.")
    print(traceback.format_exc())
    sys.exit(1)

if not raw_data_triplets:
    print(f"FATAL: No data loaded from {TRAIN_DATA_PATH}. Ensure the file is not empty and correctly formatted. Exiting.")
    sys.exit(1)
print(f"Loaded {len(raw_data_triplets)} triplets from file.")

# Create a DataFrame for mz.pack()
all_data_for_pack = []
query_to_id = {}
doc_to_id = {}
query_id_counter = 0
doc_id_counter = 0

for query_text, positive_text, negative_text in raw_data_triplets:
    # Query
    if query_text not in query_to_id:
        query_to_id[query_text] = f"q_{query_id_counter}"
        query_id_counter += 1
    q_id = query_to_id[query_text]

    # Positive Document
    if positive_text not in doc_to_id:
        doc_to_id[positive_text] = f"d_{doc_id_counter}"
        doc_id_counter += 1
    pos_doc_id = doc_to_id[positive_text]
    all_data_for_pack.append({'text_left': query_text, 'text_right': positive_text, 'id_left': q_id, 'id_right': pos_doc_id, 'label': 1})

    # Negative Document
    if negative_text not in doc_to_id:
        doc_to_id[negative_text] = f"d_{doc_id_counter}"
        doc_id_counter += 1
    neg_doc_id = doc_to_id[negative_text]
    all_data_for_pack.append({'text_left': query_text, 'text_right': negative_text, 'id_left': q_id, 'id_right': neg_doc_id, 'label': 0})

if not all_data_for_pack:
    print(f"FATAL: No data pairs generated for DataPack. Check raw_data_triplets. Exiting.")
    sys.exit(1)

packed_df = pd.DataFrame(all_data_for_pack)
print(f"Created DataFrame with {len(packed_df)} pairs for DataPack.")

# --- Split data into training and validation sets (e.g., 90/10) ---
from sklearn.model_selection import train_test_split
# Stratify by label if it's binary and reasonably balanced.
# If using ranking with many queries, might need grouped CV by query ID.
# For now, simple split.
stratify_col = packed_df['label'] if 'label' in packed_df.columns and len(packed_df['label'].unique()) > 1 else None
train_df, dev_df = train_test_split(packed_df, test_size=0.1, random_state=SEED, stratify=stratify_col)

print(f"Train DataFrame size: {len(train_df)}")
print(f"Validation DataFrame size: {len(dev_df)}")

# Create Raw DataPacks
train_pack_raw = mz.pack(train_df, task='ranking') # Specify task for clarity
dev_pack_raw = mz.pack(dev_df, task='ranking')

print(f"Raw Train DataPack created. Relations: {len(train_pack_raw.relation)}, Left: {len(train_pack_raw.left)}, Right: {len(train_pack_raw.right)}")
print(f"Raw Dev DataPack created. Relations: {len(dev_pack_raw.relation)}, Left: {len(dev_pack_raw.left)}, Right: {len(dev_pack_raw.right)}")

# --- User input to specify start model ---
user_start_model_name = input(f"Enter model name to start training from (e.g., DRMM), or press Enter to start from the beginning: ").strip()

# --- Main Training Loop ---
start_training_flag = not bool(user_start_model_name)

for model_name_key in MODELS_TO_TRAIN:
    if not start_training_flag and model_name_key == user_start_model_name:
        start_training_flag = True
    
    if not start_training_flag:
        print(f"Skipping model {model_name_key} as per user request.")
        continue

    print(f"\n{'='*20} Training model: {model_name_key} {'='*20}")
    config = ALL_MODEL_CONFIGS[model_name_key]
    
    model_save_dir = os.path.join(base_trained_models_dir, chunk_type_for_folder, current_date_str, model_name_key)
    os.makedirs(model_save_dir, exist_ok=True)
    print(f"Model artifacts will be saved in: {model_save_dir}")

    try:
        # 1. Preprocessing
        print("Preprocessing data...")
        preprocessor_params_conf = config.get("preprocessor_params", {})
        
        fixed_length_left = config["fixed_length_left_hint"]
        fixed_length_right = config["fixed_length_right_hint"]

        if config["preprocessor_type"] == "BasicPreprocessor":
            preprocessor = mz.preprocessors.BasicPreprocessor(
                truncated_length_left=fixed_length_left, # Changed: Use truncated_length_left
                truncated_length_right=fixed_length_right, # Changed: Use truncated_length_right
                **preprocessor_params_conf
            )
        elif config["preprocessor_type"] == "ModelDefault":
            actual_model_class_name_for_prep = config.get("model_class_name", model_name_key)
            try:
                mz_model_class_for_prep = getattr(mz.models, actual_model_class_name_for_prep)
            except AttributeError:
                # Try common variations if class name in config doesn't match mz.models directly
                if actual_model_class_name_for_prep == "MatchLSTM": # Common case
                    mz_model_class_for_prep = getattr(mz.models, "MatchLSTMModel") 
                else: # Re-raise if still not found
                    raise
            preprocessor = mz_model_class_for_prep.get_default_preprocessor(**preprocessor_params_conf)
            # For default preprocessors, fixed lengths are often set after instantiation or during transform
            # Ensure they are used:
            if hasattr(preprocessor, 'fixed_length_left'):
                 preprocessor.fixed_length_left = fixed_length_left
            if hasattr(preprocessor, 'fixed_length_right'):
                 preprocessor.fixed_length_right = fixed_length_right
        else:
            raise ValueError(f"Unsupported preprocessor type: {config['preprocessor_type']}")

        # Fit on training data, then transform both train and dev
        # For ModelDefault, fixed_lengths might be passed to transform if not set on preprocessor
        train_pack_processed = preprocessor.fit_transform(train_pack_raw, verbose=0)
        dev_pack_processed = preprocessor.transform(dev_pack_raw, verbose=0)
        
        preprocessor_save_path = os.path.join(model_save_dir, "preprocessor.dill")
        preprocessor.save(preprocessor_save_path)
        print(f"Preprocessor saved to {preprocessor_save_path}")

        # 2. Embedding
        glove_path = GLOVE_PATHS[config["glove_path_key"]]
        print(f"Loading GloVe embeddings from: {glove_path} for dim {config['embedding_dim']}")
        
        # Robustly get term_index from preprocessor context
        term_index_map = preprocessor.context.get('term_index')
        if term_index_map is None:
            vocab_unit = preprocessor.context.get('vocab_unit')
            if vocab_unit and hasattr(vocab_unit, 'state') and 'term_index' in vocab_unit.state:
                term_index_map = vocab_unit.state['term_index']
            elif vocab_unit and isinstance(vocab_unit, dict) and 'term_index' in vocab_unit: # if vocab_unit is a dict-like object
                 term_index_map = vocab_unit['term_index']
            # Fallback for older MatchZoo versions or custom preprocessors if term_index is directly on vocab_unit
            elif vocab_unit and hasattr(vocab_unit, 'term_index') and isinstance(vocab_unit.term_index, dict):
                 term_index_map = vocab_unit.term_index


        if term_index_map is None:
            keys_in_context = list(preprocessor.context.keys()) if hasattr(preprocessor, 'context') and preprocessor.context else []
            raise ValueError(f"Could not retrieve term_index from preprocessor context for model {model_name_key}. Available keys: {keys_in_context}")

        embedding_matrix = load_from_file(glove_path, mode='glove').build_matrix(term_index_map)
        print(f"Embedding matrix shape: {embedding_matrix.shape}")

        # 3. Task
        # Common metrics for ranking. Ensure your early_stopping_key is among these.
        ranking_metrics = [
            mz.metrics.MeanAveragePrecision(), 
            mz.metrics.NormalizedDiscountedCumulativeGain(k=3), 
            mz.metrics.NormalizedDiscountedCumulativeGain(k=5)
        ]
        if config["dataset_mode"] == "point" and model_name_key == "KNRM": # Specific metrics for KNRM point mode if needed
            # For point mode with binary labels, these might be more informative
            # However, RankHingeLoss still implies ranking. MAP/NDCG can still be used if labels are 0/1.
            # ranking_metrics = [mz.metrics.Accuracy(), mz.metrics.Precision(), mz.metrics.Recall()]
            pass # Using common ranking metrics for now.

        task = mz.tasks.Ranking(
            losses=get_mz_loss_class(config["loss_fn_class_name"])(**config["loss_params"]),
            metrics=ranking_metrics
        )

        # 4. Model Setup
        actual_model_class_name = config.get("model_class_name", model_name_key)
        try:
            mz_model_class = getattr(mz.models, actual_model_class_name)
        except AttributeError:
            if actual_model_class_name == "MatchLSTM":
                 mz_model_class = getattr(mz.models, "MatchLSTMModel")
            else:
                raise
        
        model = mz_model_class()
        model.params['task'] = task
        model.params['embedding'] = embedding_matrix
        # Crucial: ensure model knows the fixed lengths for its architecture
        # model.params['fixed_length_left'] = preprocessor.context.get('fixed_length_left', fixed_length_left) # Removed: Causes KeyError for some models
        # model.params['fixed_length_right'] = preprocessor.context.get('fixed_length_right', fixed_length_right) # Removed: Causes KeyError for some models
        
        model.params.update(config["hyperparameters"])
        model.build()
        print(f"Model {actual_model_class_name} built. Parameters summary:")
        for param_key, param_val in model.params.items():
            if param_key == 'embedding':
                print(f"  {param_key}: <matrix shape {param_val.shape if hasattr(param_val, 'shape') else 'N/A'}>")
            else:
                print(f"  {param_key}: {jsonify_value(param_val)}")


        # 5. Training
        optimizer_class = get_torch_optimizer_class(config["optimizer_class_name"])
        optimizer = optimizer_class(model.parameters(), lr=config["learning_rate"], **config.get("optimizer_params", {}))

        train_loader_callbacks = []
        if config["dataset_mode"] == "pair":
            train_loader_callbacks.append(mz.dataloader.callbacks.PairGenerator(
                num_dup=config.get("dataset_num_dup", 1), 
                num_neg=config.get("dataset_num_neg", 1),
                # relation_file=train_pack_processed.relation_file # Not needed if DataPack is already processed
            ))
        
        train_loader = mz.dataloader.DataLoader(
            dataset=train_pack_processed,
            stage='train',
            mode=config["dataset_mode"],
            batch_size=config["batch_size"],
            num_workers=0, # Set to 0 for easier debugging
            callbacks=train_loader_callbacks
        )
        
        # For validation, mode is typically 'point' to evaluate each item.
        # The dev_pack_processed already contains (text_left, text_right, label) pairs.
        dev_loader = mz.dataloader.DataLoader(
            dataset=dev_pack_processed,
            stage='eval',
            mode='point', # Evaluate each pair as a point for ranking metrics
            batch_size=config["batch_size"], # Can be larger for validation
            num_workers=0,
            callbacks=[] # No pair generation needed for eval
        )

        trainer = mz.Trainer(
            model=model,
            optimizer=optimizer,
            trainloader=train_loader,
            validloader=dev_loader,
            epochs=config["epochs"],
            patience=config.get("patience", GENERAL_PATIENCE),
            save_dir=model_save_dir,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            monitor=config.get("early_stopping_key", GENERAL_EARLY_STOPPING_KEY),
            clip_norm=config.get("clip_norm", GENERAL_CLIP_NORM),
            validate_interval=config.get("validate_interval", GENERAL_VALIDATE_INTERVAL)
        )

        print(f"Starting training for {actual_model_class_name}...")
        trainer.run()
        
        print(f"Training finished for {actual_model_class_name}.")
        # Trainer saves the best model. To save the final one explicitly:
        # model.save(model_save_dir) # This saves model.pt and model.config
        print(f"Model {actual_model_class_name} (best or final) saved in {model_save_dir}")

        # Save comprehensive training configuration
        full_config_to_save = {
            "model_name_key": model_name_key,
            "actual_model_class_name": actual_model_class_name,
            "config_used": config,
            "glove_path_used": glove_path,
            "preprocessor_context": {k: jsonify_value(v) for k, v in preprocessor.context.items()},
            "trainer_best_score": trainer.best_score if hasattr(trainer, 'best_score') else None,
            "model_final_params": {k: jsonify_value(v) for k, v in model.params.items()}
        }
        config_save_path = os.path.join(model_save_dir, "training_details.json")
        with open(config_save_path, 'w', encoding='utf-8') as f_cfg:
            json.dump(full_config_to_save, f_cfg, indent=4, default=str) # Use str as a simpler default for jsonify
        print(f"Training details for {actual_model_class_name} saved to {config_save_path}")

    except Exception as e_train_model:
        print(f"ERROR during processing model {model_name_key}: {e_train_model}")
        print(traceback.format_exc())
        print(f"Skipping to next model if any.")

print(f"\n{'='*20} All specified models processed. {'='*20}")