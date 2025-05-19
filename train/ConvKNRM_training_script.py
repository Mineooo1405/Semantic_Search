import torch
import numpy as np
import pandas as pd
import matchzoo as mz
import nltk
import os
from pathlib import Path
from transform_data import transform_to_matchzoo_format

print(f"MatchZoo version: {mz.__version__}")
print(f"PyTorch version: {torch.__version__}")
print(f"NumPy version: {np.__version__}")
print(f"Pandas version: {pd.__version__}")

# --- Script Configuration ---
TRAIN_FILE_PATH = r"D:/SemanticSearch/TrainingData_MatchZoo_BEIR/msmarco_semantic-grouping/train_2/msmarco_semantic-grouping_train_triplets.tsv"
DEV_FILE_PATH = r"D:/SemanticSearch/TrainingData_MatchZoo_BEIR/msmarco_semantic-grouping/train_2/dev.tsv"
TEST_FILE_PATH = r"D:/SemanticSearch/TrainingData_MatchZoo_BEIR/msmarco_semantic-grouping/train_2/dev.tsv" # Using dev for test

EMBEDDING_FILE_PATH = r"D:/SemanticSearch/embedding/glove.6B/glove.6B.100d.txt"
EMBEDDING_DIMENSION = 100 # Should match the GloVe file used

OUTPUT_DIR = Path("D:/SemanticSearch/trained_convknrm_model")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_SAVE_PATH = OUTPUT_DIR / "convknrm_model.pt"
PREPROCESSOR_SAVE_PATH = OUTPUT_DIR / "convknrm_preprocessor.dill"

# ConvKNRM specific parameters (from conv_knrm.ipynb)
BATCH_SIZE = 20
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
        exit(1)

# --- Task Definition (from init.ipynb, used by conv_knrm.ipynb) ---
print("Defining ranking task for ConvKNRM...")
ranking_task = mz.tasks.Ranking(losses=mz.losses.RankHingeLoss())
ranking_task.metrics = [
    mz.metrics.NormalizedDiscountedCumulativeGain(k=3),
    mz.metrics.NormalizedDiscountedCumulativeGain(k=5),
    mz.metrics.MeanAveragePrecision()
]
print(f"`ranking_task` initialized with loss: {ranking_task.losses[0]} and metrics: {ranking_task.metrics}")

# --- Helper function to load triplet data from TSV ---
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

# --- Load CUSTOM Dataset ---
print("Loading CUSTOM dataset...")
source_train_data = load_triplet_data_from_tsv(TRAIN_FILE_PATH)
source_dev_data = load_triplet_data_from_tsv(DEV_FILE_PATH)
source_test_data = load_triplet_data_from_tsv(TEST_FILE_PATH)

if not source_train_data:
    print("CRITICAL: Training data is empty. Exiting.")
    exit(1)
if not source_dev_data:
    print("CRITICAL: Development/Validation data is empty. Exiting.")
    exit(1)

transformed_train_data = transform_to_matchzoo_format(source_train_data)
transformed_dev_data = transform_to_matchzoo_format(source_dev_data)
transformed_test_data = transform_to_matchzoo_format(source_test_data)

train_df = pd.DataFrame(transformed_train_data, columns=['text_left', 'text_right', 'label'])
dev_df = pd.DataFrame(transformed_dev_data, columns=['text_left', 'text_right', 'label'])
test_df = pd.DataFrame(transformed_test_data, columns=['text_left', 'text_right', 'label'])

train_pack_raw = mz.pack(train_df, task=ranking_task)
dev_pack_raw = mz.pack(dev_df, task=ranking_task)
test_pack_raw = mz.pack(test_df, task=ranking_task)

print(f"Train DataPack created with {len(train_pack_raw)} entries.")
print(f"Dev (Validation) DataPack created with {len(dev_pack_raw)} entries.")
print(f"Test DataPack created with {len(test_pack_raw)} entries.")

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
    num_neg=1,      # From conv_knrm.ipynb trainset (RankHingeLoss implies 1 neg)
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
model.params['filters'] = FILTERS
model.params['conv_activation_func'] = CONV_ACTIVATION_FUNC
model.params['max_ngram'] = MAX_NGRAM
model.params['use_crossmatch'] = USE_CROSSMATCH
model.params['kernel_num'] = KERNEL_NUM
model.params['sigma'] = SIGMA
model.params['exact_sigma'] = EXACT_SIGMA

model.build()
print(model)
print(f"Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

# --- Trainer Setup (from conv_knrm.ipynb) ---
print("Setting up Trainer...")
optimizer = torch.optim.Adadelta(model.parameters()) # As per conv_knrm.ipynb
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=SCHEDULER_STEP_SIZE) # As per conv_knrm.ipynb

trainer = mz.trainers.Trainer(
    model=model,
    optimizer=optimizer,
    trainloader=trainloader,
    validloader=validloader,
    validate_interval=None, # Validates at the end of each epoch
    epochs=EPOCHS,
    scheduler=scheduler,
    clip_norm=CLIP_NORM # As per conv_knrm.ipynb
)
print("Trainer setup complete.")

# --- Training ---
print(f"Starting ConvKNRM model training for {EPOCHS} epochs...")
trainer.run()
print("Training finished.")

# --- Save Model and Preprocessor ---
print(f"Saving model to: {MODEL_SAVE_PATH}")
torch.save(model.state_dict(), MODEL_SAVE_PATH)
print(f"Saving preprocessor to: {PREPROCESSOR_SAVE_PATH}")
preprocessor.save(PREPROCESSOR_SAVE_PATH)

print("ConvKNRM training script finished successfully.")
print(f"Model saved at: {MODEL_SAVE_PATH}")
print(f"Preprocessor saved at: {PREPROCESSOR_SAVE_PATH}")
