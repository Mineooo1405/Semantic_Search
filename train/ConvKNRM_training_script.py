import matchzoo as mz
import nltk
import pandas as pd
import numpy as np
import torch
import os
from pathlib import Path
import json # Added for saving config
import sys # Added
import shutil # Added for rmtree
import argparse # Added for command-line arguments
from datetime import datetime # ADDED to fix missing import

# from transform_data import transform_to_matchzoo_format # Removed unused import

print(f"MatchZoo version: {mz.__version__}")
print(f"PyTorch version: {torch.__version__}")
print(f"NumPy version: {np.__version__}")
print(f"Pandas version: {pd.__version__}")

# --- Argument Parsing --- Added
parser = argparse.ArgumentParser(description="ConvKNRM Training Script")
parser.add_argument("--train_file", type=str, required=True, help="Path to the training data file.")
parser.add_argument("--dev_file", type=str, required=True, help="Path to the development/validation data file.")
parser.add_argument("--test_file", type=str, required=True, help="Path to the test data file.")
args = parser.parse_args()

# --- Script Configuration ---
TRAIN_FILE_PATH = args.train_file
DEV_FILE_PATH = args.dev_file
TEST_FILE_PATH = args.test_file

EMBEDDING_FILE_PATH = r"D:/SemanticSearch/embedding/glove.6B/glove.6B.300d.txt"
EMBEDDING_DIMENSION = 300

# --- Determine the dataset name part and set up output directory --- Added and moved up
dataset_name_part = os.path.basename(os.path.dirname(TRAIN_FILE_PATH)) # Takes parent directory of train file
model_save_dir_base = "trained_convknrm_model"
full_save_dir = Path(model_save_dir_base) / dataset_name_part
full_save_dir.mkdir(parents=True, exist_ok=True)
print(f"[ConvKNRM Script] All outputs will be saved in: {full_save_dir}")

MODEL_SAVE_PATH = full_save_dir / "convknrm_model.pt" # MODIFIED name for clarity
PREPROCESSOR_SAVE_PATH = full_save_dir / "convknrm_preprocessor.dill" # MODIFIED name for clarity
CONFIG_SAVE_PATH = full_save_dir / "convknrm_config.json" # MODIFIED name for clarity

# ConvKNRM specific parameters (from conv_knrm.ipynb)
BATCH_SIZE = 32 # CHANGED from 20
EPOCHS = 10
FILTERS = 128
CONV_ACTIVATION_FUNC = 'tanh'
MAX_NGRAM = 3
USE_CROSSMATCH = True
KERNEL_NUM = 11
SIGMA = 0.1
EXACT_SIGMA = 0.001
CLIP_NORM = 10
SCHEDULER_STEP_SIZE = 3

# --- Setup NLTK ---
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    print("NLTK 'punkt' tokenizer not found. Attempting to download...")
    nltk_data_dir = Path(os.getcwd()) / '.nltk_data'
    nltk_data_dir.mkdir(parents=True, exist_ok=True)
    if str(nltk_data_dir) not in nltk.data.path:
        nltk.data.path.append(str(nltk_data_dir))
    try:
        nltk.download('punkt', download_dir=str(nltk_data_dir))
        print(f"'punkt' tokenizer downloaded to {nltk_data_dir} or already available there.")
    except Exception as e:
        print(f"Failed to download 'punkt': {e}. Please ensure NLTK can download data or install 'punkt' manually.")
        sys.exit(1) # Changed to sys.exit

# --- Task Definition (from init.ipynb, used by conv_knrm.ipynb) ---
print("Defining ranking task for ConvKNRM...")
ranking_task = mz.tasks.Ranking(losses=mz.losses.RankHingeLoss(num_neg=4, margin=0.2)) # CHANGED
ranking_task.metrics = [
    mz.metrics.NormalizedDiscountedCumulativeGain(k=3),
    mz.metrics.NormalizedDiscountedCumulativeGain(k=5),
    mz.metrics.MeanAveragePrecision()
]
print(f"`ranking_task` initialized with loss: {ranking_task.losses[0]} and metrics: {ranking_task.metrics}")

# --- Helper function to load data from qrels format (query, document, label) ---
def load_data_from_qrels_format(file_path, delimiter='\t'): # Changed to '\\t' for tab
    print(f"Loading qrels data from: {file_path} with delimiter '{repr(delimiter)}'") # Used repr for clarity
    records = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                parts = line.strip().split(delimiter)
                if len(parts) == 3: # query, document, label
                    query, doc, label_str = parts[0], parts[1], parts[2]
                    try:
                        label = int(label_str)
                        # MatchZoo's Ranking task typically expects 0 for non-relevant, 1 for relevant.
                        if label not in (0, 1):
                            print(f"Warning: Line {i+1}: Label '{label_str}' is not 0 or 1. Treating as 0 (non-relevant). Modify if other behavior is desired.")
                            label = 0 
                        records.append({'text_left': query, 'text_right': doc, 'label': label})
                    except ValueError:
                        print(f"Skipping line {i+1} due to non-integer label: {label_str}")
                else:
                    print(f"Skipping malformed line {i+1} (expected 3 parts, got {len(parts)}): {line.strip()}")
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return [] 
    except Exception as e:
        print(f"Error reading or processing file {file_path}: {e}")
        return []
    print(f"Loaded {len(records)} query-document-label records from {file_path}")
    return records

# --- Load CUSTOM Dataset ---
print("Loading CUSTOM dataset (query, document, label format)...")
# The TRAIN_FILE_PATH, DEV_FILE_PATH, TEST_FILE_PATH will be set by the master script
source_train_data = load_data_from_qrels_format(TRAIN_FILE_PATH)
source_dev_data = load_data_from_qrels_format(DEV_FILE_PATH) # Dev data should also be in qrels format
source_test_data = load_data_from_qrels_format(TEST_FILE_PATH) # Test data should also be in qrels format

if not source_train_data:
    print("CRITICAL: Training data is empty. Exiting.")
    sys.exit(1) # Changed to sys.exit
if not source_dev_data:
    print("CRITICAL: Development/Validation data is empty. Exiting.")
    sys.exit(1) # Changed to sys.exit

# The load_triplet_data_from_tsv function now returns a list of dicts with 'text_left', 'text_right', 'label'
train_df = pd.DataFrame(source_train_data)
dev_df = pd.DataFrame(source_dev_data)
test_df = pd.DataFrame(source_test_data) if source_test_data else pd.DataFrame(columns=['text_left', 'text_right', 'label'])

# Ensure dataframes are not empty before packing
if train_df.empty:
    print("CRITICAL: Training DataFrame is empty. Exiting.")
    sys.exit(1) # Changed to sys.exit
if dev_df.empty:
    print("CRITICAL: Development DataFrame is empty. Exiting.")
    sys.exit(1) # Changed to sys.exit

train_pack_raw = mz.pack(train_df, task=ranking_task)
dev_pack_raw = mz.pack(dev_df, task=ranking_task)

if not test_df.empty:
    test_pack_raw = mz.pack(test_df, task=ranking_task)
else:
    print("WARNING: Test DataFrame is empty. Using dev_pack_raw for test_pack_raw.")
    test_pack_raw = dev_pack_raw

print(f"Train DataPack created with {len(train_pack_raw)} entries.")
print(f"Dev (Validation) DataPack created with {len(dev_pack_raw)} entries.")
print(f"Test DataPack created with {len(test_pack_raw)} entries (may be using dev data if original test was empty).")

# --- Preprocessing (from conv_knrm.ipynb) ---
print("Preprocessing data for ConvKNRM...")
preprocessor = mz.models.ConvKNRM.get_default_preprocessor()
train_pack_processed = preprocessor.fit_transform(train_pack_raw)
dev_pack_processed = preprocessor.transform(dev_pack_raw)
test_pack_processed = preprocessor.transform(test_pack_raw)
print("Data preprocessed.")
print(f"Preprocessor context (vocab size, etc.): {preprocessor.context}")

# --- Embedding Setup (USING YOUR CUSTOM EMBEDDINGS) ---
print(f"Setting up CUSTOM GloVe embeddings from: {EMBEDDING_FILE_PATH}")
if not os.path.exists(EMBEDDING_FILE_PATH):
    print(f"ERROR: Embedding file not found: {EMBEDDING_FILE_PATH}")
    print("Using DUMMY random embeddings as a fallback.")
    term_index_for_dummy = preprocessor.context['vocab_unit'].state['term_index']
    embedding_matrix = np.random.rand(len(term_index_for_dummy) + 1, EMBEDDING_DIMENSION)
else:
    custom_embedding = mz.embedding.load_from_file(EMBEDDING_FILE_PATH, mode='glove')
    term_index = preprocessor.context['vocab_unit'].state['term_index']
    embedding_matrix = custom_embedding.build_matrix(term_index)
    print(f"Embedding matrix built from custom file. Shape: {embedding_matrix.shape}")

    # Normalize embedding_matrix (from conv_knrm.ipynb)
    print("Normalizing embedding matrix...")
    l2_norm = np.sqrt((embedding_matrix * embedding_matrix).sum(axis=1))
    epsilon = 1e-8 # To prevent division by zero for zero vectors
    embedding_matrix = embedding_matrix / (l2_norm[:, np.newaxis] + epsilon)
    embedding_matrix = np.nan_to_num(embedding_matrix, nan=0.0)
    print(f"Normalized embedding matrix shape: {embedding_matrix.shape}")

# --- Dataset and DataLoader (from conv_knrm.ipynb) ---
print("Creating Datasets and DataLoaders...")
trainset = mz.dataloader.Dataset(
    data_pack=train_pack_processed,
    mode='pair',
    num_dup=5,      # From conv_knrm.ipynb trainset
    num_neg=4,      # CHANGED From conv_knrm.ipynb trainset (RankHingeLoss implies 1 neg)
    batch_size=BATCH_SIZE,
    resample=True,
    sort=False,
    shuffle=True    # Generally good for training
)
validset = mz.dataloader.Dataset(
    data_pack=dev_pack_processed, # Using dev_pack for validation
    batch_size=BATCH_SIZE,
    shuffle=False   # No need to shuffle validation data
)

padding_callback = mz.models.ConvKNRM.get_default_padding_callback()

trainloader = mz.dataloader.DataLoader(
    dataset=trainset,
    stage='train',
    callback=padding_callback
)
validloader = mz.dataloader.DataLoader(
    dataset=validset,
    stage='dev', # For validation
    callback=padding_callback
)
print("Datasets and DataLoaders created.")

# --- Model Setup (from conv_knrm.ipynb) ---
print("Setting up ConvKNRM model...")
model = mz.models.ConvKNRM()

model.params['task'] = ranking_task
model.params['embedding'] = embedding_matrix
model.params['embedding_freeze'] = False # CHANGED FROM True
model.params['filters'] = FILTERS # Added to match notebook
model.params['conv_activation_func'] = CONV_ACTIVATION_FUNC # Added to match notebook
model.params['max_ngram'] = MAX_NGRAM # Added to match notebook
model.params['use_crossmatch'] = USE_CROSSMATCH # Added to match notebook
model.params['kernel_num'] = KERNEL_NUM
model.params['sigma'] = SIGMA
model.params['exact_sigma'] = EXACT_SIGMA

model.build()
print(model)
print(f"Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

# --- Trainer Setup (from conv_knrm.ipynb) ---
print("Setting up Trainer...")
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=SCHEDULER_STEP_SIZE)

trainer = mz.trainers.Trainer(
    model=model,
    optimizer=optimizer,
    trainloader=trainloader,
    validloader=validloader,
    validate_interval=None,
    epochs=EPOCHS,
    scheduler=scheduler,
    clip_norm=CLIP_NORM,
    save_dir=str(full_save_dir), # MODIFIED: Ensure it's a string
    patience=EPOCHS,
    key=ranking_task.metrics[0]
)
print("Trainer setup complete.")

# --- Helper function to safely get parameter values (from MatchLSTM script) ---
def safe_get_param_value(params_table, key, default_val):
    if isinstance(params_table, mz.engine.param_table.ParamTable):
        # For MatchZoo ParamTable, __getitem__ (e.g., params_table[key]) returns the actual value.
        # .get() returns the Param object.
        if key in params_table:
            return params_table[key]  # Correctly fetches the value
        else:
            return default_val
    # Fallback for general dictionary-like objects if params_table is not a ParamTable
    elif hasattr(params_table, 'get') and callable(getattr(params_table, 'get')):
        return params_table.get(key, default_val) # Standard dict.get()
    # Fallback for attribute access (less likely for this specific use case with model.params)
    elif hasattr(params_table, key):
        ret_attr = getattr(params_table, key, default_val)
        # Handle cases where getattr might still return a Param-like wrapper from a non-ParamTable object
        # This is a heuristic check.
        if default_val != ret_attr and type(ret_attr) != type(default_val) and hasattr(ret_attr, 'value'):
             # Avoid calling .value on primitive types or if .value is callable (method)
            if not isinstance(ret_attr.value, (type(None), bool, int, float, str, list, dict)) and callable(ret_attr.value):
                pass # it's a method, not a value attribute we want
            elif type(ret_attr) != type(ret_attr.value): # Check if it's indeed a wrapper
                return ret_attr.value
        return ret_attr
    return default_val

# --- Training ---
print(f"Starting ConvKNRM model training for {EPOCHS} epochs...")
try:
    trainer.run()
except Exception as e:
    print(f"CRITICAL ERROR during trainer.run(): {e}")
    import traceback
    traceback.print_exc()
    # sys.exit(1) # Optionally exit
print("Training finished or error occurred.")

# --- Save Model and Preprocessor ---
print(f"Saving model to: {MODEL_SAVE_PATH}")
torch.save(model.state_dict(), MODEL_SAVE_PATH)

print(f"Saving preprocessor to: {PREPROCESSOR_SAVE_PATH}")
preprocessor.save(PREPROCESSOR_SAVE_PATH)

# --- Create and Save Config Dictionary ---
# (Assuming safe_get_param_value is defined and works as intended)
config_to_save = {
    "model_name": "ConvKNRM",
    "model_class": model.__class__.__name__,
    "task_type": model.params['task'].__class__.__name__ if model.params['task'] else None,
    "loss_function_class": model.params['task'].loss.__class__.__name__ if model.params['task'] and hasattr(model.params['task'], 'loss') else None,
    # For RankHingeLoss, num_neg is not directly on the loss object in the same way as RankCrossEntropyLoss
    # It's often handled by the DataLoader (num_neg=1 in this script's trainset)
    # We can record the dataloader's num_neg if that's the intended information
    "dataloader_num_neg_for_loss_effect": trainset._num_neg if hasattr(trainset, '_num_neg') else None, # This will now reflect 4
    "optimizer_class": optimizer.__class__.__name__,
    "learning_rate": optimizer.param_groups[0]['lr'], # CHANGED to get lr from Adam
    "model_hyperparameters_used": {
        "filters": safe_get_param_value(model.params, 'filters', FILTERS),
        "conv_activation_func": safe_get_param_value(model.params, 'conv_activation_func', CONV_ACTIVATION_FUNC),
        "max_ngram": safe_get_param_value(model.params, 'max_ngram', MAX_NGRAM),
        "use_crossmatch": safe_get_param_value(model.params, 'use_crossmatch', USE_CROSSMATCH),
        "kernel_num": safe_get_param_value(model.params, 'kernel_num', KERNEL_NUM),
        "sigma": safe_get_param_value(model.params, 'sigma', SIGMA),
        "exact_sigma": safe_get_param_value(model.params, 'exact_sigma', EXACT_SIGMA),
        "mask_value": safe_get_param_value(model.params, 'mask_value', 0), # BasicModel default
        "embedding_input_dim_model_param": safe_get_param_value(model.params, 'embedding_input_dim', None),
        "embedding_output_dim_model_param": safe_get_param_value(model.params, 'embedding_output_dim', None)
    },
    "embedding_source_file": EMBEDDING_FILE_PATH if 'EMBEDDING_FILE_PATH' in globals() and os.path.exists(EMBEDDING_FILE_PATH) else "random_dummy_embeddings",
    "embedding_dim_used": EMBEDDING_DIMENSION,
    "batch_size": BATCH_SIZE,
    "epochs_configured": EPOCHS,
    "epochs_completed": trainer._epoch if hasattr(trainer, '_epoch') else 0,
    "patience_used": trainer._early_stopping.patience if hasattr(trainer, '_early_stopping') and hasattr(trainer._early_stopping, 'patience') else None,
    "validate_interval": trainer._validate_interval if hasattr(trainer, '_validate_interval') else None,
    "early_stopping_key": trainer._early_stopping.key if hasattr(trainer, '_early_stopping') and hasattr(trainer._early_stopping, 'key') else None,
    "device_used_for_training": str(trainer._device) if hasattr(trainer, '_device') else None,
    "start_epoch_configured": trainer._start_epoch if hasattr(trainer, '_start_epoch') else 1,
    "gradient_clip_norm": trainer._clip_norm if hasattr(trainer, '_clip_norm') else None, # CLIP_NORM is defined
    "scheduler_class": trainer._scheduler.__class__.__name__ if hasattr(trainer, '_scheduler') and trainer._scheduler is not None else None,
    "scheduler_step_size": SCHEDULER_STEP_SIZE if hasattr(trainer, '_scheduler') and isinstance(trainer._scheduler, torch.optim.lr_scheduler.StepLR) else None,
    "trainer_save_directory": str(trainer._save_dir.resolve()) if hasattr(trainer, '_save_dir') and trainer._save_dir is not None else None,
    "trainer_save_all_flag": trainer._save_all if hasattr(trainer, '_save_all') else False,
    "trainer_verbose_level": trainer._verbose if hasattr(trainer, '_verbose') else 1,
    "fixed_length_left": preprocessor.context.get('fixed_length_left'),
    "fixed_length_right": preprocessor.context.get('fixed_length_right'),
    "vocab_size_from_preprocessor_context": preprocessor.context.get('vocab_size'),
    "embedding_input_dim_from_preprocessor_context": preprocessor.context.get('embedding_input_dim'),
    "matchzoo_version": mz.__version__,
    "training_script_path": str(Path(__file__).resolve()), # Added for traceability
    "timestamp": datetime.now().isoformat() # Added for traceability
}

print(f"Saving config to: {CONFIG_SAVE_PATH}")
with open(CONFIG_SAVE_PATH, 'w') as f:
    json.dump(config_to_save, f, indent=4)

print("All artifacts (model, preprocessor, config) saved.")

print("ConvKNRM training script finished.") # Added to signify end
