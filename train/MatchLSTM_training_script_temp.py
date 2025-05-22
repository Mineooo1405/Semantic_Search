import matchzoo as mz
import nltk
import pandas as pd
import numpy as np
import torch
import os
from pathlib import Path
# from transform_data import transform_to_matchzoo_format
import json # ADDED for saving config
import dill # ADDED for saving preprocessor explicitly

# --- Helper function to safely get parameter values ---
def safe_get_param_value(params_table, key, default_val):
    if key in params_table: # Check if key exists
        val = params_table[key] # Retrieve item, should be the value due to ParamTable.__getitem__
        # Defensive check: if somehow a Param object itself is returned by __getitem__
        if isinstance(val, mz.engine.param.Param):
            # This case should ideally not happen if ParamTable.__getitem__ works as documented (returns .value)
            # print(f"DEBUG: Param object was returned by __getitem__ for {key}, accessing .value")
            return val.value
        return val # Assumed to be the actual value
    # print(f"DEBUG: Key \'{key}\' not found in model.params. Using default: {default_val}")
    return default_val

# --- Setup NLTK ---
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    print("NLTK 'punkt' tokenizer not found. Attempting to download...")
    nltk_data_dir = Path.home() / '.matchzoo' / 'nltk_data'
    nltk_data_dir.mkdir(parents=True, exist_ok=True)
    if str(nltk_data_dir) not in nltk.data.path:
        nltk.data.path.append(str(nltk_data_dir))
    nltk.download('punkt', download_dir=str(nltk_data_dir))
    print(f"'punkt' tokenizer downloaded to {nltk_data_dir} or already available there.")

# --- Task Definition ---
print("Defining ranking task for MatchLSTM...")
ranking_task = mz.tasks.Ranking(losses=mz.losses.RankCrossEntropyLoss(num_neg=10))
ranking_task.metrics = [
    mz.metrics.NormalizedDiscountedCumulativeGain(k=3),
    mz.metrics.NormalizedDiscountedCumulativeGain(k=5),
    mz.metrics.MeanAveragePrecision()
]
print(f"`ranking_task` initialized with loss: {ranking_task.losses[0]} and metrics: {ranking_task.metrics}") 

# --- Helper function to load triplet data from TSV ---
def load_triplet_data_from_tsv(file_path):
    print(f"Loading triplet data from: {file_path}")
    data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) == 3:
                    data.append(parts)
                else:
                    print(f"Skipping malformed line (expected 3 columns, got {len(parts)}): {line.strip()}")
        print(f"Loaded {len(data)} triplets.")
        return data
    except FileNotFoundError:
        print(f"ERROR: File not found: {file_path}")
        return []
    except Exception as e:
        print(f"ERROR: Could not read file {file_path}: {e}")
        return []

# --- Load CUSTOM Dataset ---
print("Loading CUSTOM dataset.")
TRAIN_FILE_PATH = r"D:/SemanticSearch/TrainingData_MatchZoo_BEIR/msmarco_semantic-grouping/train_2/msmarco_semantic-grouping_train_train_mz.tsv"
DEV_FILE_PATH = r"D:/SemanticSearch/TrainingData_MatchZoo_BEIR/msmarco_semantic-grouping/train_2/msmarco_semantic-grouping_train_dev_mz.tsv"
TEST_FILE_PATH = r"D:/SemanticSearch/TrainingData_MatchZoo_BEIR/msmarco_semantic-grouping/train_2/msmarco_semantic-grouping_train_dev_mz.tsv" # Using dev for test

source_train_data = load_triplet_data_from_tsv(TRAIN_FILE_PATH)
source_dev_data = load_triplet_data_from_tsv(DEV_FILE_PATH)
source_test_data = load_triplet_data_from_tsv(TEST_FILE_PATH)

# transformed_train_data = transform_to_matchzoo_format(source_train_data)
# transformed_dev_data = transform_to_matchzoo_format(source_dev_data)
# transformed_test_data = transform_to_matchzoo_format(source_test_data)

train_df = pd.DataFrame(source_train_data, columns=['text_left', 'text_right', 'label'])
dev_df = pd.DataFrame(source_dev_data, columns=['text_left', 'text_right', 'label'])
test_df = pd.DataFrame(source_test_data, columns=['text_left', 'text_right', 'label'])

if not train_df.empty:
    train_pack_raw = mz.pack(train_df)
    train_pack_raw.task = ranking_task
    print(f"Train DataPack created with {len(train_pack_raw)} entries.")
else:
    print("Training data is empty. Exiting.")
    exit()

if not dev_df.empty:
    dev_pack_raw = mz.pack(dev_df)
    dev_pack_raw.task = ranking_task
    print(f"Dev (Validation) DataPack created with {len(dev_pack_raw)} entries.")
else:
    print("Dev (Validation) data is empty. Exiting as it's needed for MatchLSTM validation.")
    exit()

if not test_df.empty:
    test_pack_raw = mz.pack(test_df)
    test_pack_raw.task = ranking_task
    print(f"Test DataPack created with {len(test_pack_raw)} entries.")
else:
    print("Test data is empty. Using Dev data as fallback for test_pack_raw.")
    if 'dev_pack_raw' in locals() and dev_pack_raw: 
        test_pack_raw = dev_pack_raw
    else:
        print("Critical error: No data for test_pack_raw. Exiting.")
        exit()

print("CUSTOM dataset loaded and transformed.")

# --- Preprocessing ---
print("Preprocessing data for MatchLSTM...")
preprocessor = mz.models.MatchLSTM.get_default_preprocessor()

train_pack_processed = preprocessor.fit_transform(train_pack_raw)
dev_pack_processed = preprocessor.transform(dev_pack_raw)
test_pack_processed = preprocessor.transform(test_pack_raw)
print("Data preprocessed.")

# --- Embedding Setup ---
print("Setting up YOUR CUSTOM embeddings...")
YOUR_EMBEDDING_FILE_PATH = r"D:\SemanticSearch\embedding\glove.6B\glove.6B.100d.txt"
YOUR_EMBEDDING_DIMENSION = 100

custom_embedding = None
if not os.path.exists(YOUR_EMBEDDING_FILE_PATH):
    print(f"WARNING: Embedding file not found ('{YOUR_EMBEDDING_FILE_PATH}').")
    print(f"Using DUMMY random embeddings with dimension {YOUR_EMBEDDING_DIMENSION}.")
    term_index_for_dummy = preprocessor.context['vocab_unit'].state['term_index']
    if not term_index_for_dummy:
        raise ValueError("Preprocessor has not been fit, cannot create dummy embeddings.")
    max_idx = max(term_index_for_dummy.values()) if term_index_for_dummy else 0
    dummy_embedding_matrix = np.random.rand(max_idx + 1, YOUR_EMBEDDING_DIMENSION)
    custom_embedding = mz.embedding.Embedding(
        weights=dummy_embedding_matrix,
        term_index=term_index_for_dummy
    )
    print(f"Dummy embedding created. Underlying matrix shape: {dummy_embedding_matrix.shape}")
else:
    try:
        custom_embedding = mz.embedding.load_from_file(YOUR_EMBEDDING_FILE_PATH, mode='glove')
        print(f"Successfully loaded embeddings from: {YOUR_EMBEDDING_FILE_PATH}")
    except Exception as e:
        print(f"Error loading custom embeddings from {YOUR_EMBEDDING_FILE_PATH}: {e}")
        print("Exiting as embeddings are crucial.")
        exit()

term_index = preprocessor.context['vocab_unit'].state['term_index']
embedding_matrix = custom_embedding.build_matrix(term_index)
print(f"Original embedding matrix for model shape: {embedding_matrix.shape}")

if embedding_matrix.ndim == 2 and embedding_matrix.shape[0] > 0:
    actual_embedding_dim = embedding_matrix.shape[1]
    if actual_embedding_dim != YOUR_EMBEDDING_DIMENSION:
        print(f"WARNING: Configured YOUR_EMBEDDING_DIMENSION was {YOUR_EMBEDDING_DIMENSION}, "
              f"but loaded matrix has dimension {actual_embedding_dim}.")
        print(f"Using dimension from loaded file: {actual_embedding_dim}.")
        YOUR_EMBEDDING_DIMENSION = actual_embedding_dim
    else:
        print(f"Embedding dimension ({actual_embedding_dim}) matches configured YOUR_EMBEDDING_DIMENSION.")
else:
    print(f"ERROR: Embedding matrix could not be built properly or is empty. Shape: {embedding_matrix.shape}")
    exit()

print("Normalizing embedding matrix...")
l2_norm = np.sqrt(np.sum(embedding_matrix * embedding_matrix, axis=1, keepdims=True))
l2_norm[l2_norm == 0] = 1e-9
embedding_matrix = embedding_matrix / l2_norm
print("Embeddings processed and normalized.")
print(f"Final embedding matrix for model shape: {embedding_matrix.shape}")

BATCH_SIZE = 20 # Defined BATCH_SIZE
print("Creating MatchZoo Datasets for MatchLSTM...")
trainset = mz.dataloader.Dataset(
    data_pack=train_pack_processed,
    mode='pair',
    num_dup=5,
    num_neg=10,
    batch_size=BATCH_SIZE, # Used BATCH_SIZE
    resample=True,
    sort=False,
    shuffle=True
)
validset = mz.dataloader.Dataset(
    data_pack=dev_pack_processed,
    batch_size=BATCH_SIZE, # Used BATCH_SIZE
    resample=False,
    sort=False,
    shuffle=False
)
print("MatchZoo Datasets created.")

# --- DataLoader Setup ---
print("Creating MatchZoo DataLoaders for MatchLSTM...")
padding_callback = mz.models.MatchLSTM.get_default_padding_callback()

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
print("MatchZoo DataLoaders created.")

# --- Model Setup ---
print("Setting up MatchLSTM model...")
model = mz.models.MatchLSTM()

model.params['task'] = ranking_task
model.params['mask_value'] = 0
model.params['embedding'] = embedding_matrix

model.build()
print("MatchLSTM Model built.")
print(model)
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'Trainable parameters: {trainable_params}')

# --- Trainer Setup ---
print("Setting up Trainer...")
optimizer = torch.optim.Adadelta(model.parameters())
NUM_TRAIN_EPOCHS = 10 # ADDED: Define configured epochs
trainer = mz.trainers.Trainer(
    model=model,
    optimizer=optimizer,
    trainloader=trainloader,
    validloader=validloader,
    validate_interval=None,
    epochs=NUM_TRAIN_EPOCHS, # CHANGED: Use the variable
    patience=NUM_TRAIN_EPOCHS # Added patience
)
print("Trainer configured.")

# --- Run Training ---
print("Starting MatchLSTM model training...")
trainer.run()
print("MatchLSTM model training finished.")

# --- Consolidate Artifact Saving ---
print("Preparing and saving model, preprocessor, and config...")
MODEL_SAVE_DIR = "trained_matchlstm_model"
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

MODEL_SAVE_PATH = os.path.join(MODEL_SAVE_DIR, "model.pt")
PREPROCESSOR_SAVE_PATH = os.path.join(MODEL_SAVE_DIR, "preprocessor.dill")
CONFIG_SAVE_PATH = os.path.join(MODEL_SAVE_DIR, "config.json")

# 1. Create Config Dictionary
# This uses the `preprocessor` object that was fitted by the initial `fit_transform(train_pack_raw)`
config_to_save = {
    "model_name": "MatchLSTM",
    "model_class": model.__class__.__name__,
    "task_type": model.params['task'].__class__.__name__ if model.params['task'] else None,
    "loss_function_class": model.params['task'].loss.__class__.__name__ if model.params['task'] and hasattr(model.params['task'], 'loss') else None,
    "loss_num_neg": safe_get_param_value(model.params['task'].loss, '_num_neg', None) if model.params['task'] and hasattr(model.params['task'], 'loss') else None, # num_neg is an attribute not a Param
    "optimizer_class": optimizer.__class__.__name__,
    "learning_rate": optimizer.defaults.get('lr'), # optimizers are standard PyTorch, .defaults.get should be fine
    "model_hyperparameters_used": {
        "lstm_units": safe_get_param_value(model.params, 'lstm_units', 100),  # Default for MatchLSTM is 100
        "mask_value": safe_get_param_value(model.params, 'mask_value', 0),    # Default for BasicModel is 0
        "embedding_input_dim_model_param": safe_get_param_value(model.params, 'embedding_input_dim', None), # Should be set by embedding
        "embedding_output_dim_model_param": safe_get_param_value(model.params, 'embedding_output_dim', None) # Should be set by embedding
    },
    "embedding_source_file": YOUR_EMBEDDING_FILE_PATH if 'YOUR_EMBEDDING_FILE_PATH' in globals() and os.path.exists(YOUR_EMBEDDING_FILE_PATH) else "random_dummy_embeddings",
    "embedding_dim_used": YOUR_EMBEDDING_DIMENSION,
    "batch_size": BATCH_SIZE,
    "epochs_configured": NUM_TRAIN_EPOCHS,
    "epochs_completed": trainer._epoch if hasattr(trainer, '_epoch') else 0,
    "patience_used": trainer._early_stopping.patience if hasattr(trainer, '_early_stopping') and hasattr(trainer._early_stopping, 'patience') else None,
    "validate_interval": trainer._validate_interval if hasattr(trainer, '_validate_interval') else None,
    "early_stopping_key": trainer._early_stopping.key if hasattr(trainer, '_early_stopping') and hasattr(trainer._early_stopping, 'key') else None,
    "device_used_for_training": str(trainer._device) if hasattr(trainer, '_device') else None,
    "start_epoch_configured": trainer._start_epoch if hasattr(trainer, '_start_epoch') else 1,
    "gradient_clip_norm": trainer._clip_norm if hasattr(trainer, '_clip_norm') else None,
    "scheduler_class": trainer._scheduler.__class__.__name__ if hasattr(trainer, '_scheduler') and trainer._scheduler is not None else None,
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

# 2. Save Config
with open(CONFIG_SAVE_PATH, 'w', encoding='utf-8') as f:
    json.dump(config_to_save, f, indent=4)
print(f"Config saved to {CONFIG_SAVE_PATH}")

# 3. Save Preprocessor (using the preprocessor from the initial fit_transform)
print("DEBUG: Checking preprocessor context BEFORE saving preprocessor.dill:")
print(f"DEBUG: preprocessor.context keys: {list(preprocessor.context.keys())}")
term_index_found = False
if 'vocab_unit' in preprocessor.context and hasattr(preprocessor.context['vocab_unit'], 'state') and 'term_index' in preprocessor.context['vocab_unit'].state:
    term_index = preprocessor.context['vocab_unit'].state['term_index']
    if term_index:
        term_index_sample = list(term_index.items())
        print(f"DEBUG: 'term_index' is PRESENT in preprocessor.context['vocab_unit'].state. Vocab size: {len(term_index_sample)}. Sample: {term_index_sample[:5]}")
        term_index_found = True
    else:
        print("DEBUG: 'term_index' is EMPTY in preprocessor.context['vocab_unit'].state.")
elif 'term_index' in preprocessor.context: # Fallback check if structure is different
    term_index = preprocessor.context['term_index']
    if term_index:
        term_index_sample = list(term_index.items())
        print(f"DEBUG: 'term_index' is PRESENT directly in preprocessor.context. Vocab size: {len(term_index_sample)}. Sample: {term_index_sample[:5]}")
        term_index_found = True
    else:
        print("DEBUG: 'term_index' is EMPTY directly in preprocessor.context.")

if term_index_found:
    print(f"DEBUG: vocab_size from preprocessor.context: {preprocessor.context.get('vocab_size')}")
    print(f"DEBUG: embedding_input_dim from preprocessor.context: {preprocessor.context.get('embedding_input_dim')}")
    dill.dump(preprocessor, open(PREPROCESSOR_SAVE_PATH, 'wb'))
    print(f"Preprocessor saved to {PREPROCESSOR_SAVE_PATH}")
else:
    print("DEBUG: CRITICAL - 'term_index' could NOT be found or was empty in preprocessor.context (checked both 'vocab_unit.state.term_index' and 'term_index').")
    print(f"DEBUG: Preprocessor NOT saved to {PREPROCESSOR_SAVE_PATH}. This indicates an issue with the initial preprocessor.fit_transform or context propagation.")

# 4. Save Model State
torch.save(model.state_dict(), MODEL_SAVE_PATH)
print(f"Model saved to {MODEL_SAVE_PATH}")

print("Script finished.")

