import torch
import numpy as np
import pandas as pd
import matchzoo as mz
import nltk
import os
import sys # ADDED
from pathlib import Path
import json # ADDED for saving config
# from Tool.transform_data import transform_to_matchzoo_format # Moved down
from datetime import datetime # ADDED for saving config

# --- Add project root to sys.path ---
# This ensures that modules in the project root (like 'Tool') can be found.
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# --- End of sys.path modification ---

from Tool.transform_data import transform_to_matchzoo_format # MOVED HERE

print(f"MatchZoo version: {mz.__version__}")
print(f"PyTorch version: {torch.__version__}")
print(f"NumPy version: {np.__version__}")
print(f"Pandas version: {pd.__version__}")

EMBEDDING_FILE_PATH = r"D:/SemanticSearch/embedding/glove.6B/glove.6B.100d.txt"
EMBEDDING_DIMENSION = 100 

OUTPUT_DIR = Path("D:/SemanticSearch/trained_drmm_model")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_SAVE_PATH = OUTPUT_DIR / "drmm_model.pt"
# PREPROCESSOR_SAVE_PATH = OUTPUT_DIR / "drmm_preprocessor.dill" # Path for the preprocessor file
CONFIG_SAVE_PATH = OUTPUT_DIR / "config.json" # ADDED: Path for the config file

HIST_BIN_SIZE = 30
BATCH_SIZE = 20
EPOCHS = 10

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
        exit(1)

# --- Helper function to safely get parameter values (copied from MatchLSTM script) ---
def safe_get_param_value(params_table, key, default_val):
    if key in params_table: # Check if key exists
        val = params_table[key] # Retrieve item, should be the value due to ParamTable.__getitem__
        # Defensive check: if somehow a Param object itself is returned by __getitem__
        if isinstance(val, mz.engine.param.Param):
            return val.value
        return val # Assumed to be the actual value
    return default_val

print("Defining ranking task for DRMM...")
ranking_task = mz.tasks.Ranking(losses=mz.losses.RankCrossEntropyLoss(num_neg=10))
ranking_task.metrics = [
    mz.metrics.NormalizedDiscountedCumulativeGain(k=3),
    mz.metrics.NormalizedDiscountedCumulativeGain(k=5),
    mz.metrics.MeanAveragePrecision()
]
print(f"`ranking_task` initialized with loss: {ranking_task.losses[0]} and metrics: {ranking_task.metrics}")

def load_triplet_data_from_tsv(file_path, delimiter='\t'):
    print(f"Loading triplet data from: {file_path} with delimiter '{delimiter}'")
    data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                parts = line.strip().split(delimiter)
                if len(parts) == 3:
                    data.append(parts)
                else:
                    print(f"Skipping malformed line #{i+1} (expected 3 columns, got {len(parts)}): {line.strip()}")
        print(f"Loaded {len(data)} triplets from {file_path}.")
        if not data:
            print(f"WARNING: No data loaded from {file_path}. Check the file format and delimiter.")
        return data
    except FileNotFoundError:
        print(f"ERROR: File not found: {file_path}")
        return []
    except Exception as e:
        print(f"ERROR: Could not read file {file_path}: {e}")
        return []
TRAIN_FILE_PATH = r"D:/SemanticSearch/TrainingData_MatchZoo_BEIR/msmarco_semantic-grouping/train_2/msmarco_semantic-grouping_train_train_mz.tsv"
DEV_FILE_PATH = r"D:/SemanticSearch/TrainingData_MatchZoo_BEIR/msmarco_semantic-grouping/train_2/msmarco_semantic-grouping_train_dev_mz.tsv"
TEST_FILE_PATH = r"D:/SemanticSearch/TrainingData_MatchZoo_BEIR/msmarco_semantic-grouping/train_2/msmarco_semantic-grouping_train_dev_mz.tsv"

source_train_data = load_triplet_data_from_tsv(TRAIN_FILE_PATH)
source_test_data = load_triplet_data_from_tsv(TRAIN_FILE_PATH)
source_dev_data = load_triplet_data_from_tsv(DEV_FILE_PATH)

print("Loading CUSTOM dataset...")

if not source_train_data:
    print("CRITICAL: Training data is empty. Exiting.")
    exit(1)
if not source_dev_data:
    print("CRITICAL: Development/Validation data is empty. Exiting.")
    exit(1)

source_train_data = load_triplet_data_from_tsv(TRAIN_FILE_PATH)
source_test_data = load_triplet_data_from_tsv(TRAIN_FILE_PATH)
source_dev_data = load_triplet_data_from_tsv(DEV_FILE_PATH)

train_df = pd.DataFrame(source_train_data, columns=['text_left', 'text_right', 'label'])
test_df = pd.DataFrame(source_test_data, columns=['text_left', 'text_right', 'label'])
dev_df = pd.DataFrame(source_dev_data, columns=['text_left', 'text_right', 'label'])

train_pack_raw = mz.pack(train_df, task=ranking_task)
dev_pack_raw = mz.pack(dev_df, task=ranking_task)
test_pack_raw = mz.pack(test_df, task=ranking_task)

print(f"Train DataPack created with {len(train_pack_raw)} entries.")
print(f"Dev (Validation) DataPack created with {len(dev_pack_raw)} entries.")
print(f"Test DataPack created with {len(test_pack_raw)} entries.")

print("Preprocessing data for DRMM...")
preprocessor = mz.models.DRMM.get_default_preprocessor()
train_pack_processed = preprocessor.fit_transform(train_pack_raw)
dev_pack_processed = preprocessor.transform(dev_pack_raw)
test_pack_processed = preprocessor.transform(test_pack_raw)
print("Data preprocessed.")
print(f"Preprocessor context (vocab size, etc.): {preprocessor.context}")

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
    epsilon = 1e-8 # To prevent division by zero for zero vectors
    embedding_matrix = embedding_matrix / (l2_norm[:, np.newaxis] + epsilon)
    embedding_matrix = np.nan_to_num(embedding_matrix, nan=0.0)
    print(f"Normalized embedding matrix shape: {embedding_matrix.shape}")

print("Setting up callbacks...")
histgram_callback = mz.dataloader.callbacks.Histogram(
    embedding_matrix, bin_size=HIST_BIN_SIZE, hist_mode='LCH'
)
padding_callback = mz.models.DRMM.get_default_padding_callback()
print("Callbacks created.")

print("Creating Datasets and DataLoaders...")
trainset = mz.dataloader.Dataset(
    data_pack=train_pack_processed,
    mode='pair',
    num_dup=5,
    num_neg=10,
    batch_size=BATCH_SIZE,
    resample=True,
    sort=False,
    callbacks=[histgram_callback]
)
validset = mz.dataloader.Dataset(
    data_pack=dev_pack_processed,
    batch_size=BATCH_SIZE,
    resample=False,
    sort=False,
    callbacks=[histgram_callback]
)

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

print("Setting up DRMM model...")
model = mz.models.DRMM()

model.params['task'] = ranking_task
model.params['embedding'] = embedding_matrix
model.params['mask_value'] = 0
model.params['hist_bin_size'] = HIST_BIN_SIZE
model.params['mlp_num_layers'] = 1
model.params['mlp_num_units'] = 10
model.params['mlp_num_fan_out'] = 1
model.params['mlp_activation_func'] = 'tanh'

model.build()
print(model)
print(f"Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

print("Setting up Trainer...")
optimizer = torch.optim.Adadelta(model.parameters())

trainer = mz.trainers.Trainer(
    model=model,
    optimizer=optimizer,
    trainloader=trainloader,
    validloader=validloader,
    validate_interval=None, # Validates at the end of each epoch
    epochs=EPOCHS,
    patience=EPOCHS # MODIFIED: Set patience to EPOCHS to run all epochs
)
print("Trainer setup complete.")

print(f"Starting DRMM model training for {EPOCHS} epochs...")
trainer.run()
print("Training finished.")

# --- Consolidate Artifact Saving ---
print("Preparing and saving model, preprocessor, and config...")

# 1. Create Config Dictionary
config_to_save = {
    "model_name": "DRMM",
    "model_class": model.__class__.__name__,
    "task_type": ranking_task.__class__.__name__, # Used ranking_task directly
    "loss_function_class": ranking_task.losses[0].__class__.__name__, # MODIFIED: Access the first loss in the list
    "loss_num_neg": ranking_task.losses[0]._num_neg if hasattr(ranking_task.losses[0], '_num_neg') else None, # MODIFIED: Directly access attribute
    "optimizer_class": optimizer.__class__.__name__,
    "optimizer_params": optimizer.defaults,
    "model_hyperparameters_used": {
        "mask_value": safe_get_param_value(model.params, 'mask_value', 0),
        "hist_bin_size": safe_get_param_value(model.params, 'hist_bin_size', HIST_BIN_SIZE),
        "mlp_num_layers": safe_get_param_value(model.params, 'mlp_num_layers', 1),
        "mlp_num_units": safe_get_param_value(model.params, 'mlp_num_units', 10),
        "mlp_num_fan_out": safe_get_param_value(model.params, 'mlp_num_fan_out', 1),
        "mlp_activation_func": safe_get_param_value(model.params, 'mlp_activation_func', 'tanh'),
        # embedding_matrix is directly passed, so its dimensions are key
        "embedding_input_dim_from_matrix_shape": embedding_matrix.shape[0] if embedding_matrix is not None else None,
        "embedding_output_dim_from_matrix_shape": embedding_matrix.shape[1] if embedding_matrix is not None else None,
    },
    "embedding_source_file": EMBEDDING_FILE_PATH if 'EMBEDDING_FILE_PATH' in globals() and EMBEDDING_FILE_PATH and os.path.exists(EMBEDDING_FILE_PATH) else "random_dummy_embeddings_or_not_specified",
    "embedding_dim_configured": EMBEDDING_DIMENSION,
    "batch_size": BATCH_SIZE,
    "epochs_configured": EPOCHS,
    "epochs_completed": trainer._epoch if hasattr(trainer, '_epoch') else 0,
    "patience_used": trainer._early_stopping.patience if hasattr(trainer, '_early_stopping') and hasattr(trainer._early_stopping, 'patience') else None,
    "validate_interval": trainer._validate_interval if hasattr(trainer, '_validate_interval') else None,
    "early_stopping_key": trainer._early_stopping.key if hasattr(trainer, '_early_stopping') and hasattr(trainer._early_stopping, 'key') else None,
    "device_used_for_training": str(trainer._device) if hasattr(trainer, '_device') else None,
    "trainer_verbose_level": trainer._verbose if hasattr(trainer, '_verbose') else 1,
    "preprocessor_context": {
        "fixed_length_left": preprocessor.context.get('fixed_length_left'),
        "fixed_length_right": preprocessor.context.get('fixed_length_right'),
        "vocab_size": preprocessor.context.get('vocab_size'),
        "embedding_input_dim": preprocessor.context.get('embedding_input_dim'), # Typically vocab_size + 1
        # "vocab_path": str(Path(preprocessor.context.get('vocab_unit')._state.get('term_index_path')).resolve()) if preprocessor.context.get('vocab_unit') and preprocessor.context.get('vocab_unit')._state.get('term_index_path') else None,
        "vocab_path": None, # Initialize to None
    },
    "matchzoo_version": mz.__version__,
    "pytorch_version": torch.__version__,
    "numpy_version": np.__version__,
    "pandas_version": pd.__version__,
    "training_script": os.path.basename(__file__),
    "training_date": pd.Timestamp.now().isoformat(),
    "output_directory": str(OUTPUT_DIR.resolve())
}

# Get vocab_path safely for Python 3.7
vocab_unit_object = preprocessor.context.get('vocab_unit')
if vocab_unit_object and hasattr(vocab_unit_object, 'state') and isinstance(vocab_unit_object.state, dict):
    term_index_path_value = vocab_unit_object.state.get('term_index_path')
    if term_index_path_value:
        config_to_save["preprocessor_context"]["vocab_path"] = str(Path(term_index_path_value).resolve())

# Ensure all values in model_hyperparameters_used are serializable (e.g. numpy types to python types)
for key, value in config_to_save.get("model_hyperparameters_used", {}).items():
    if hasattr(value, 'item'): # For numpy types like np.int32
        config_to_save["model_hyperparameters_used"][key] = value.item()

# 2. Save Config
with open(CONFIG_SAVE_PATH, 'w', encoding='utf-8') as f:
    json.dump(config_to_save, f, indent=4)
print(f"Config saved to {CONFIG_SAVE_PATH}")

# 3. Save Model
print(f"Saving model to: {MODEL_SAVE_PATH}")
torch.save(model.state_dict(), MODEL_SAVE_PATH)

# 4. Save Preprocessor
# preprocessor.save() expects a directory and saves "preprocessor.dill" within it.
preprocessor_file_in_output_dir = OUTPUT_DIR / "preprocessor.dill"
print(f"Saving preprocessor to directory: {OUTPUT_DIR} (will be saved as {preprocessor_file_in_output_dir})")
preprocessor.save(OUTPUT_DIR) # Pass the directory

print("\nDRMM training script finished successfully.")
print(f"Model saved at: {MODEL_SAVE_PATH}")
print(f"Preprocessor saved at: {preprocessor_file_in_output_dir}") # Updated path
print(f"Config saved at: {CONFIG_SAVE_PATH}")
