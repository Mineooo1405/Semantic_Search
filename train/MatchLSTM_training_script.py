import matchzoo as mz
import nltk
import pandas as pd
import numpy as np
import torch
import os
from pathlib import Path
from transform_data import transform_to_matchzoo_format

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
train_file_path = r"D:/SemanticSearch/TrainingData_MatchZoo_BEIR/msmarco_semantic-grouping/train_2/msmarco_semantic-grouping_train_triplets.tsv"
dev_file_path = r"D:/SemanticSearch/TrainingData_MatchZoo_BEIR/msmarco_semantic-grouping/train_2/dev.tsv"
test_file_path = r"D:/SemanticSearch/TrainingData_MatchZoo_BEIR/msmarco_semantic-grouping/train_2/dev.tsv" # Using dev for test

source_train_data = load_triplet_data_from_tsv(train_file_path)
source_dev_data = load_triplet_data_from_tsv(dev_file_path)
source_test_data = load_triplet_data_from_tsv(test_file_path)

transformed_train_data = transform_to_matchzoo_format(source_train_data)
transformed_dev_data = transform_to_matchzoo_format(source_dev_data)
transformed_test_data = transform_to_matchzoo_format(source_test_data)

train_df = pd.DataFrame(transformed_train_data, columns=['text_left', 'text_right', 'label'])
dev_df = pd.DataFrame(transformed_dev_data, columns=['text_left', 'text_right', 'label'])
test_df = pd.DataFrame(transformed_test_data, columns=['text_left', 'text_right', 'label'])

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

trainer = mz.trainers.Trainer(
    model=model,
    optimizer=optimizer,
    trainloader=trainloader,
    validloader=validloader,
    validate_interval=None,
    epochs=10,
    patience=5 # Added patience
)
print("Trainer configured.")

# --- Run Training ---
print("Starting MatchLSTM model training...")
trainer.run()
print("MatchLSTM model training finished.")

# --- Save Model and Preprocessor ---
print("Saving model and preprocessor...")
MODEL_SAVE_DIR = "trained_matchlstm_model" # Defined MODEL_SAVE_DIR
MODEL_SAVE_PATH = os.path.join(MODEL_SAVE_DIR, "model.pt") # Defined MODEL_SAVE_PATH
PREPROCESSOR_SAVE_PATH = os.path.join(MODEL_SAVE_DIR, "preprocessor.dill") # Defined PREPROCESSOR_SAVE_PATH

os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

torch.save(model.state_dict(), MODEL_SAVE_PATH) # Used defined path
print(f"Model saved to {MODEL_SAVE_PATH}")

preprocessor.save(PREPROCESSOR_SAVE_PATH) # Used defined path
print(f"Preprocessor saved to {PREPROCESSOR_SAVE_PATH}")

print("Script finished.")

