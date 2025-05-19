import matchzoo as mz
import nltk
import pandas as pd
import numpy as np
import torch
import os
from pathlib import Path
from transform_data import transform_to_matchzoo_format

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

print("Defining ranking task...")
ranking_task = mz.tasks.Ranking(losses=mz.losses.RankCrossEntropyLoss(num_neg=1))

def load_triplet_data_from_tsv(file_path):
    """Loads data from a TSV file (query, positive_doc, negative_doc)."""
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

print("Loading CUSTOM dataset.")
train_file_path = r"D:/SemanticSearch/TrainingData_MatchZoo_BEIR/msmarco_semantic-grouping/train_2/msmarco_semantic-grouping_train_triplets.tsv"
test_file_path = r"D:/SemanticSearch/TrainingData_MatchZoo_BEIR/msmarco_semantic-grouping/train_2/dev.tsv"
dev_file_path = r"D:/SemanticSearch/TrainingData_MatchZoo_BEIR/msmarco_semantic-grouping/train_2/dev.tsv"

source_train_data = load_triplet_data_from_tsv(train_file_path)
source_test_data = load_triplet_data_from_tsv(test_file_path)
source_dev_data = load_triplet_data_from_tsv(dev_file_path)

transformed_train_data = transform_to_matchzoo_format(source_train_data)
transformed_test_data = transform_to_matchzoo_format(source_test_data)
transformed_dev_data = transform_to_matchzoo_format(source_dev_data)

train_df = pd.DataFrame(transformed_train_data, columns=['text_left', 'text_right', 'label'])
test_df = pd.DataFrame(transformed_test_data, columns=['text_left', 'text_right', 'label'])
dev_df = pd.DataFrame(transformed_dev_data, columns=['text_left', 'text_right', 'label'])

if not train_df.empty:
    train_pack_raw = mz.pack(train_df) 
    train_pack_raw.task = ranking_task 
    print(f"Train DataPack created with {len(train_pack_raw)} entries.")
else:
    print("Training data is empty. Cannot create DataPack. Exiting.")
    exit()

if not test_df.empty:
    test_pack_raw = mz.pack(test_df) 
    test_pack_raw.task = ranking_task 
    print(f"Test DataPack created with {len(test_pack_raw)} entries.")
else:
    print("Test data is empty. Cannot create DataPack. Exiting.")
    exit()

if not dev_df.empty:
    dev_pack_raw = mz.pack(dev_df) 
    dev_pack_raw.task = ranking_task 
    print(f"Dev DataPack created with {len(dev_pack_raw)} entries.")
else:
    print("Dev data is empty. Using test data for dev pack as a fallback or exiting if critical.")
    if test_pack_raw:
        dev_pack_raw = test_pack_raw 
        print("WARNING: Dev data was empty, using test data as dev_pack_raw.")
    else:
        print("Dev data is empty and no test data to fallback. Exiting.")
        exit()


print("CUSTOM dataset loaded and transformed.")

print("Preprocessing data...")
preprocessor = mz.preprocessors.BasicPreprocessor(
    truncated_length_left=10,
    truncated_length_right=40,
    filter_low_freq=2 
)
train_pack_processed = preprocessor.fit_transform(train_pack_raw)

if dev_pack_raw: 
    dev_pack_processed = preprocessor.transform(dev_pack_raw)
else:
    dev_pack_processed = None 
    print("Warning: dev_pack_raw was not available for processing.")
test_pack_processed = preprocessor.transform(test_pack_raw)
print("Data preprocessed.")


print("Setting up CUSTOM embeddings...")

YOUR_EMBEDDING_FILE_PATH = r"D:\SemanticSearch\embedding\glove.6B\glove.6B.100d.txt" 
YOUR_EMBEDDING_DIMENSION = 100 


custom_embedding = None
if not os.path.exists(YOUR_EMBEDDING_FILE_PATH): 
    print(f"WARNING: Embedding file not found ('{YOUR_EMBEDDING_FILE_PATH}').")
    print(f"Using DUMMY random embeddings with dimension {YOUR_EMBEDDING_DIMENSION}.")
    term_index_for_dummy = preprocessor.context['vocab_unit'].state['term_index']
    vocab_size = len(term_index_for_dummy)

    if not term_index_for_dummy:
        raise ValueError("Preprocessor has not been fit, cannot create dummy embeddings without vocabulary.")

    max_idx = 0
    if term_index_for_dummy: 
        max_idx = max(term_index_for_dummy.values())

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
        print("Please ensure the path is correct and the embedding file is in a supported format (e.g., GloVe or Word2Vec text).")
        print("Script will exit as embeddings are crucial.")
        exit() 

term_index = preprocessor.context['vocab_unit'].state['term_index']
embedding_matrix = custom_embedding.build_matrix(term_index)

print(f"Final embedding matrix for model shape: {embedding_matrix.shape}")

if embedding_matrix.ndim == 2 and embedding_matrix.shape[0] > 0 : 
    actual_embedding_dim = embedding_matrix.shape[1]
    if actual_embedding_dim != YOUR_EMBEDDING_DIMENSION:
        print(f"WARNING: Configured YOUR_EMBEDDING_DIMENSION was {YOUR_EMBEDDING_DIMENSION}, "
              f"but the loaded embedding matrix has dimension {actual_embedding_dim}.")
        print(f"The model will use the dimension from the loaded file: {actual_embedding_dim}.")
        YOUR_EMBEDDING_DIMENSION = actual_embedding_dim 
    else:
        print(f"Embedding dimension ({actual_embedding_dim}) matches configured YOUR_EMBEDDING_DIMENSION.")
elif custom_embedding is not None: 
    print(f"WARNING: Embedding matrix could not be built properly. Shape is {embedding_matrix.shape}.")
    print(f"Proceeding with configured YOUR_EMBEDDING_DIMENSION ({YOUR_EMBEDDING_DIMENSION}) for model, but this may lead to errors.")

else: 
    print("ERROR: custom_embedding object is None before building matrix. This should not happen.")
    exit()



print("Normalizing embedding matrix...")
l2_norm = np.sqrt((embedding_matrix * embedding_matrix).sum(axis=1))
embedding_matrix = embedding_matrix / l2_norm[:, np.newaxis]
print("Embeddings processed.")

print("Creating MatchZoo Datasets...")
trainset = mz.dataloader.Dataset(
    data_pack=train_pack_processed,
    mode='pair',
    num_dup=5,
    num_neg=1,
    batch_size=20,  
    resample=True,  
    sort=False      
)

validset = mz.dataloader.Dataset(
    data_pack=test_pack_processed, 
    batch_size=20,  
    resample=False, 
    sort=False      
)
print("MatchZoo Datasets created.")

print("Creating MatchZoo DataLoaders...")
padding_callback = mz.models.KNRM.get_default_padding_callback()

trainloader = mz.dataloader.DataLoader(
    dataset=trainset,
    # batch_size=20, 
    stage='train',
    # resample=True, 
    # sort=False,   
    callback=padding_callback
)

validloader = mz.dataloader.DataLoader(
    dataset=validset,
    # batch_size=20, 
    stage='dev', 
    callback=padding_callback
)
print("MatchZoo DataLoaders created.")

print("Setting up KNRM model...")
model = mz.models.KNRM()

model.params['task'] = ranking_task
model.params['embedding'] = embedding_matrix
model.params['kernel_num'] = 21
model.params['sigma'] = 0.1
model.params['exact_sigma'] = 0.001

model.build()
print("KNRM Model built.")
print(model) 
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'Trainable parameters: {trainable_params}')

print("Setting up Trainer...")
optimizer = torch.optim.Adadelta(model.parameters()) 

trainer = mz.trainers.Trainer(
    model=model,
    optimizer=optimizer,
    trainloader=trainloader,
    validloader=validloader,
    validate_interval=None, 
    epochs=10 
)
print("Trainer configured.")

print("Starting KNRM model training...")
trainer.run()
print("KNRM model training finished.")

print("Saving model and preprocessor...")
model_save_dir = "trained_knrm_model" 
os.makedirs(model_save_dir, exist_ok=True)

model_save_path = os.path.join(model_save_dir, "model.pt")
torch.save(model.state_dict(), model_save_path)
print(f"Model saved to {model_save_path}")

preprocessor_save_path = os.path.join(model_save_dir, "preprocessor.dill")
preprocessor.save(preprocessor_save_path)
print(f"Preprocessor saved to {preprocessor_save_path}")

print("Script finished.")
