import matchzoo as mz
import nltk
import pandas as pd
import numpy as np
import torch
import os
from pathlib import Path
import json # Added
import sys # Added
import shutil # Added for rmtree
import argparse # Added for command-line arguments

# --- Argument Parsing --- Added
parser = argparse.ArgumentParser(description="KNRM Training Script")
parser.add_argument("--train_file", type=str, required=True, help="Path to the training data file.")
parser.add_argument("--dev_file", type=str, required=True, help="Path to the development/validation data file.")
parser.add_argument("--test_file", type=str, required=True, help="Path to the test data file.")
args = parser.parse_args()

# --- Script Configuration ---
TRAIN_FILE_PATH = args.train_file 
DEV_FILE_PATH = args.dev_file 
TEST_FILE_PATH = args.test_file # Placeholder, will be replaced by master script

EMBEDDING_FILE_PATH = r"D:/SemanticSearch/embedding/glove.6B/glove.6B.300d.txt"
EMBEDDING_DIMENSION = 300 # Should match the GloVe file used

MODEL_NAME = "KNRM" # For constructing output directory
OUTPUT_DIR = Path(f"D:/SemanticSearch/trained_{MODEL_NAME.lower()}_model")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_SAVE_PATH = OUTPUT_DIR / "model.pt" # Standardized name
PREPROCESSOR_SAVE_PATH = OUTPUT_DIR / "preprocessor.dill" # Standardized name
CONFIG_SAVE_PATH = OUTPUT_DIR / "config.json"

# KNRM specific parameters (can be adjusted)
BATCH_SIZE = 128 # From ConvKNRM example # MODIFIED from 20 to 128 (aligning with KNRM notebook)
EPOCHS = 10     # From ConvKNRM example
KERNEL_NUM = 21 # Default KNRM
SIGMA = 0.1     # Default KNRM
EXACT_SIGMA = 0.001 # Default KNRM
# Add other KNRM specific params if needed, e.g., CLIP_NORM, SCHEDULER_STEP_SIZE if used

# --- Setup NLTK ---
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    print("NLTK 'punkt' tokenizer not found. Attempting to download...")
    # Standardized NLTK data path within the project for better portability
    nltk_data_dir = Path(os.getcwd()) / '.nltk_data' 
    nltk_data_dir.mkdir(parents=True, exist_ok=True)
    if str(nltk_data_dir) not in nltk.data.path:
        nltk.data.path.append(str(nltk_data_dir))
    try:
        nltk.download('punkt', download_dir=str(nltk_data_dir))
        print(f"'punkt' tokenizer downloaded to {nltk_data_dir} or already available there.")
    except Exception as e:
        print(f"Failed to download 'punkt': {e}. Please ensure NLTK can download data or install 'punkt' manually.")
        sys.exit(1)

# --- Task Definition ---
print("Defining ranking task for KNRM...")
# Using RankHingeLoss as it's common for ranking, similar to ConvKNRM example
ranking_task = mz.tasks.Ranking(losses=mz.losses.RankHingeLoss()) 
ranking_task.metrics = [
    mz.metrics.NormalizedDiscountedCumulativeGain(k=3),
    mz.metrics.NormalizedDiscountedCumulativeGain(k=5),
    mz.metrics.MeanAveragePrecision()
]
print(f"`ranking_task` initialized with loss: {ranking_task.losses[0]} and metrics: {ranking_task.metrics}")

# --- Helper function to load triplet data from TSV ---
def load_triplet_data_from_tsv(file_path, delimiter='\t'): # Corrected delimiter to be a tab character
    print(f"Loading triplet data from: {file_path} with delimiter '{repr(delimiter)}'") # Use repr to show actual delimiter
    pairs_data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                parts = line.strip().split(delimiter)
                if len(parts) == 3:
                    query, pos_doc, neg_doc = parts[0], parts[1], parts[2]
                    pairs_data.append({'text_left': query, 'text_right': pos_doc, 'label': 1})
                    pairs_data.append({'text_left': query, 'text_right': neg_doc, 'label': 0})
                else:
                    print(f"Skipping malformed line {i+1} (expected 3 columns, got {len(parts)}): {line.strip()}")
        print(f"Loaded {len(pairs_data)} pairs from {len(pairs_data)//2 if pairs_data else 0} triplets in {file_path}")
        return pairs_data
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}") # Changed error message
        return []
    except Exception as e:
        print(f"Error reading or processing file {file_path}: {e}") # Changed error message
        return []

print("Loading CUSTOM dataset...")
# The TRAIN_FILE_PATH, DEV_FILE_PATH, TEST_FILE_PATH will be set by the master script
source_train_data = load_triplet_data_from_tsv(TRAIN_FILE_PATH)
source_dev_data = load_triplet_data_from_tsv(DEV_FILE_PATH)
source_test_data = load_triplet_data_from_tsv(TEST_FILE_PATH)

if not source_train_data:
    print("CRITICAL: Training data is empty. Exiting.")
    sys.exit(1) 
if not source_dev_data:
    print("CRITICAL: Development/Validation data is empty. Exiting.")
    sys.exit(1) 

train_df = pd.DataFrame(source_train_data)
dev_df = pd.DataFrame(source_dev_data)
test_df = pd.DataFrame(source_test_data) if source_test_data else pd.DataFrame(columns=['text_left', 'text_right', 'label'])

if train_df.empty:
    print("CRITICAL: Training DataFrame is empty. Exiting.")
    sys.exit(1)
if dev_df.empty:
    print("CRITICAL: Development DataFrame is empty. Exiting.")
    sys.exit(1)

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

# --- Preprocessing ---
print("Preprocessing data for KNRM...")
# Using KNRM's default preprocessor
preprocessor = mz.preprocessors.BasicPreprocessor(
    truncated_mode='post',  # Added to align with notebook
    truncated_length_left=10,
    truncated_length_right=100, # MODIFIED from 40 to 100 (aligning with KNRM notebook)
    filter_low_freq=2
)
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

    print("Normalizing embedding matrix...")
    l2_norm = np.sqrt((embedding_matrix * embedding_matrix).sum(axis=1))
    epsilon = 1e-8 
    embedding_matrix = embedding_matrix / (l2_norm[:, np.newaxis] + epsilon)
    embedding_matrix = np.nan_to_num(embedding_matrix, nan=0.0)
    print(f"Normalized embedding matrix shape: {embedding_matrix.shape}")

# --- Dataset and DataLoader ---
print("Creating Datasets and DataLoaders...")
trainset = mz.dataloader.Dataset(
    data_pack=train_pack_processed,
    mode='pair',
    num_dup=5,      # Common value, can be tuned # MODIFIED from 20 to 5 (aligning with KNRM notebook)
    num_neg=1,      # For RankHingeLoss or similar
    batch_size=BATCH_SIZE,
    resample=True,
    sort=False,
    shuffle=True    
)
validset = mz.dataloader.Dataset(
    data_pack=dev_pack_processed, 
    batch_size=BATCH_SIZE,
    shuffle=False   
)

padding_callback = mz.models.KNRM.get_default_padding_callback()

trainloader = mz.dataloader.DataLoader(
    dataset=trainset,
    stage='train',
    callback=padding_callback
)
validloader = mz.dataloader.DataLoader(
    dataset=validset,
    stage='dev', 
    callback=padding_callback
)
print("Datasets and DataLoaders created.")

# --- Model Setup ---
print("Setting up KNRM model...")
model = mz.models.KNRM()
model.params['task'] = ranking_task
model.params['embedding'] = embedding_matrix
model.params['embedding_freeze'] = False # Ensure embeddings are trainable
model.params['kernel_num'] = 21
model.params['sigma'] = 0.1
model.params['exact_sigma'] = 0.001

model.build()
print(model)
print(f"Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

# --- Trainer Setup ---
print("Setting up Trainer...")
optimizer = torch.optim.Adadelta(model.parameters()) # Changed from Adam to Adadelta to match notebook
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3) # Optional: add if needed

# Determine the dataset name part from the path
dataset_name_part = os.path.basename(args.train_file)

# Define the base directory for this model type
model_save_dir_base = "trained_knrm_model"

# Construct the full save directory path
full_save_dir = os.path.join(model_save_dir_base, dataset_name_part)

# Ensure the directory exists
os.makedirs(full_save_dir, exist_ok=True)
print(f"[KNRM Script] Model will be saved in: {full_save_dir}")

trainer = mz.trainers.Trainer(
    model=model,
    optimizer=optimizer,
    trainloader=trainloader,
    validloader=validloader,
    validate_interval=None, 
    epochs=EPOCHS, 
    save_dir=full_save_dir, # MODIFIED
    clip_norm=10, 
    patience=EPOCHS, # MODIFIED from EPOCHS
    key=ranking_task.metrics[0]# MODIFIED from ranking_task.metrics[0]
)
print("Trainer setup complete.")

# --- Helper function to safely get parameter values ---
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
print(f"Starting KNRM model training for {EPOCHS} epochs...")
trainer.run()
print("Training finished.")

# --- Save Model and Preprocessor ---
print(f"Saving model to: {MODEL_SAVE_PATH}")
torch.save(model.state_dict(), MODEL_SAVE_PATH)

# --- Save Preprocessor (handling MatchZoo's directory creation) ---
TEMP_PREPROCESSOR_PATH = OUTPUT_DIR / "temp_preprocessor_save"
ACTUAL_PREPROCESSOR_FILE_INSIDE_TEMP = TEMP_PREPROCESSOR_PATH / "preprocessor.dill"

print(f"Attempting to save preprocessor via MatchZoo to a temporary location: {TEMP_PREPROCESSOR_PATH}")
preprocessor.save(TEMP_PREPROCESSOR_PATH)

if TEMP_PREPROCESSOR_PATH.is_dir() and ACTUAL_PREPROCESSOR_FILE_INSIDE_TEMP.is_file():
    print(f"MatchZoo created a directory. Moving {ACTUAL_PREPROCESSOR_FILE_INSIDE_TEMP} to {PREPROCESSOR_SAVE_PATH}")
    PREPROCESSOR_SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    if PREPROCESSOR_SAVE_PATH.is_dir():
        print(f"Removing existing directory at target preprocessor path: {PREPROCESSOR_SAVE_PATH}")
        shutil.rmtree(PREPROCESSOR_SAVE_PATH)
    elif PREPROCESSOR_SAVE_PATH.is_file():
        print(f"Removing existing file at target preprocessor path: {PREPROCESSOR_SAVE_PATH}")
        PREPROCESSOR_SAVE_PATH.unlink()
    shutil.move(str(ACTUAL_PREPROCESSOR_FILE_INSIDE_TEMP), str(PREPROCESSOR_SAVE_PATH))
    print(f"Preprocessor moved to: {PREPROCESSOR_SAVE_PATH}")
    try:
        shutil.rmtree(TEMP_PREPROCESSOR_PATH)
        print(f"Cleaned up temporary directory: {TEMP_PREPROCESSOR_PATH}")
    except OSError as e:
        print(f"Warning: Could not remove temporary directory {TEMP_PREPROCESSOR_PATH}: {e}")
elif TEMP_PREPROCESSOR_PATH.is_file():
    print(f"MatchZoo saved preprocessor directly as a file. Moving {TEMP_PREPROCESSOR_PATH} to {PREPROCESSOR_SAVE_PATH}")
    PREPROCESSOR_SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    if PREPROCESSOR_SAVE_PATH.is_dir():
        shutil.rmtree(PREPROCESSOR_SAVE_PATH)
    elif PREPROCESSOR_SAVE_PATH.is_file():
        PREPROCESSOR_SAVE_PATH.unlink()
    shutil.move(str(TEMP_PREPROCESSOR_PATH), str(PREPROCESSOR_SAVE_PATH))
    print(f"Preprocessor moved to: {PREPROCESSOR_SAVE_PATH}")
else:
    print(f"Warning: Preprocessor saving did not result in an expected file/directory at {TEMP_PREPROCESSOR_PATH}. Check MatchZoo behavior.")
    print(f"Saving preprocessor directly to intended path: {PREPROCESSOR_SAVE_PATH} as a fallback.")
    if PREPROCESSOR_SAVE_PATH.is_dir():
        shutil.rmtree(PREPROCESSOR_SAVE_PATH)
    preprocessor.save(PREPROCESSOR_SAVE_PATH) # Fallback

# --- Save Configuration ---
print(f"Saving configuration to: {CONFIG_SAVE_PATH}")
config_to_save = {
    "model_name": MODEL_NAME, # Use MODEL_NAME variable
    "model_class": model.__class__.__name__,
    "task_type": model.params['task'].__class__.__name__ if model.params['task'] else None,
    "loss_function_class": model.params['task'].loss.__class__.__name__ if model.params['task'] and hasattr(model.params['task'], 'loss') else None,
    # For RankHingeLoss, num_neg is often handled by the DataLoader
    "dataloader_num_neg_for_loss_effect": trainset._num_neg if hasattr(trainset, '_num_neg') else None,
    "optimizer_class": optimizer.__class__.__name__,
    "learning_rate": optimizer.defaults.get('lr'),
    "model_hyperparameters_used": {
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
    "gradient_clip_norm": trainer._clip_norm if hasattr(trainer, '_clip_norm') else None,
    "scheduler_class": trainer._scheduler.__class__.__name__ if hasattr(trainer, '_scheduler') and trainer._scheduler is not None else None,
    # Add scheduler_step_size if scheduler is StepLR, similar to ConvKNRM
    "scheduler_step_size": trainer._scheduler.step_size if hasattr(trainer, '_scheduler') and isinstance(trainer._scheduler, torch.optim.lr_scheduler.StepLR) else None,
    "trainer_save_directory": str(trainer._save_dir.resolve()) if hasattr(trainer, '_save_dir') and trainer._save_dir is not None else None,
    "trainer_save_all_flag": trainer._save_all if hasattr(trainer, '_save_all') else False,
    "trainer_verbose_level": trainer._verbose if hasattr(trainer, '_verbose') else 1,
    "fixed_length_left": preprocessor.context.get('fixed_length_left'),
    "fixed_length_right": preprocessor.context.get('fixed_length_right'),
    "vocab_size_from_preprocessor_context": preprocessor.context.get('vocab_size'),
    "embedding_input_dim_from_preprocessor_context": preprocessor.context.get('embedding_input_dim'),
    "matchzoo_version": mz.__version__,
    "training_script": os.path.basename(__file__),
    "training_date": pd.Timestamp.now().isoformat()
}

# Ensure all values in model_hyperparameters_used are serializable
for key, value in config_to_save.get("model_hyperparameters_used", {}).items():
    if isinstance(value, np.integer):
        config_to_save["model_hyperparameters_used"][key] = int(value)
    elif isinstance(value, np.floating):
        config_to_save["model_hyperparameters_used"][key] = float(value)
    elif isinstance(value, np.ndarray):
        config_to_save["model_hyperparameters_used"][key] = value.tolist()
    elif not isinstance(value, (str, int, float, bool, list, dict, type(None))):
        config_to_save["model_hyperparameters_used"][key] = str(value)

with open(CONFIG_SAVE_PATH, 'w') as f:
    json.dump(config_to_save, f, indent=4)
print(f"Configuration saved to {CONFIG_SAVE_PATH}")

print(f"{MODEL_NAME} training script finished.") # Added to signify end
print(f"Model saved at: {MODEL_SAVE_PATH}")
print(f"Preprocessor saved at: {PREPROCESSOR_SAVE_PATH}")
