import matchzoo as mz
import os
import pandas as pd
import sys
import dill
from datetime import datetime
import numpy as np
import keras
import shutil
import torch
from torch.utils.data import DataLoader, Dataset
import json

def load_glove_embeddings(path, term_index, embedding_dim):
    embeddings_index = {}
    print(f"Reading GloVe file from: {path}")
    try:
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                values = line.split()
                if len(values) < embedding_dim + 1:
                    continue
                word = values[0]
                try:
                    coefs = np.asarray(values[1:], dtype='float32')
                    if len(coefs) == embedding_dim:
                        embeddings_index[word] = coefs
                except ValueError:
                    continue
    except FileNotFoundError:
        print(f"Error: GloVe file not found at {path}")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading GloVe file: {e}")
        sys.exit(1)

    num_tokens = len(term_index) + 1
    embedding_matrix = np.zeros((num_tokens, embedding_dim))
    hits = 0
    misses = 0
    for word, i in term_index.items():
        if i >= num_tokens:
             continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
            hits += 1
        else:
            misses += 1
    print(f"Built embedding matrix: {hits} words found, {misses} words missing.")
    return embedding_matrix

def select_model():
    available_models = {
        "1": "KNRM",
        "2": "MatchLSTM",
    }
    print("\n--- Select Model for Training ---")
    for key, name in available_models.items():
        print(f"{key}. {name}")

    while True:
        choice = input(f"Enter the number of the model you want to train (e.g., 1 for KNRM): ").strip()
        if choice in available_models:
            selected_model_name = available_models[choice]
            print(f"You selected: {selected_model_name}")
            return selected_model_name
        else:
            print("Invalid choice. Please select a number from the list.")

def get_validated_train_data_path():
    default_train_path = "D:/SemanticSearch/TrainingData_MatchZoo_BEIR/msmarco_semantic-grouping/train_2/msmarco_semantic-grouping_train_triplets.tsv"
    allowed_chunk_keywords = {
        "semantic-grouping": "grouping",
        "semantic-splitter": "splitter",
        "text-splitter": "text"
    }
    while True:
        user_path = input(f"Enter the full path to your training triplets file (e.g., .tsv)\n(default: {default_train_path}): ").strip()
        if not user_path:
            user_path = default_train_path
        
        user_path = os.path.normpath(user_path)

        if not os.path.exists(user_path) or not os.path.isfile(user_path):
            print(f"Error: File not found or not a file at '{user_path}'. Please try again.")
            continue
        try:
            file_parent_dir = os.path.dirname(user_path)
            chunk_type_dir = os.path.dirname(file_parent_dir)
            method_indicator_from_path = os.path.basename(chunk_type_dir).lower()
            
            current_chunk_type = None
            for keyword, type_name in allowed_chunk_keywords.items():
                if keyword in method_indicator_from_path:
                    current_chunk_type = type_name
                    break
            if current_chunk_type:
                print(f"Detected chunk type: {current_chunk_type} from path component '{method_indicator_from_path}'.")
                return user_path, current_chunk_type
            else:
                print(f"Error: Could not determine a valid chunk type from path component '{method_indicator_from_path}' (derived from '{user_path}').")
                print(f"Ensure '{method_indicator_from_path}' contains keywords like 'semantic-grouping', 'semantic-splitter', or 'text-splitter'.")
        except Exception as e_path_parse: 
            print(f"Error parsing path '{user_path}' to determine chunk type: {e_path_parse}")
        
        retry = input("Do you want to try entering the path again? (yes/no): ").strip().lower()
        if retry != 'yes':
            print("Exiting due to invalid training data path.")
            sys.exit(1)

TRAIN_DATA_PATH, chunk_type_for_folder = get_validated_train_data_path()
GLOVE_PATH = "D:/SemanticSearch/embedding/glove.6B/glove.6B.100d.txt" 
EMBEDDING_DIM = 100
MODEL_NAME_TO_RUN = select_model()
current_date_str = datetime.now().strftime("%Y%m%d")
base_trained_models_dir = "D:/SemanticSearch/TrainedModels"
MODEL_OUTPUT_DIR = os.path.join(base_trained_models_dir, f"model_{MODEL_NAME_TO_RUN}_{chunk_type_for_folder}_{current_date_str}")
BATCH_SIZE = 512
EPOCHS = 5
LEARNING_RATE = 1e-3
HINGE_MARGIN = 1.0
FIXED_LENGTH_LEFT = 30
FIXED_LENGTH_RIGHT = 100

os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)
if not os.path.exists(GLOVE_PATH):
    print(f"Error: GloVe file not found at {GLOVE_PATH}")
    sys.exit(1)

print("Loading training data and preparing DataPack for MatchZoo 2.2...")
queries = {}
docs = {}
relations_list = []
query_counter = 0
doc_counter = 0
processed_triplets = 0

with open(TRAIN_DATA_PATH, 'r', encoding='utf-8') as f:
    for i, line in enumerate(f):
        parts = line.strip().split('\t')
        if len(parts) == 3:
            query_text, positive_doc_text, negative_doc_text = parts[0], parts[1], parts[2]

            if query_text not in queries:
                queries[query_text] = f"q_{query_counter}"
                query_counter += 1
            query_id = queries[query_text]

            if positive_doc_text not in docs:
                docs[positive_doc_text] = f"d_{doc_counter}"
                doc_counter += 1
            positive_doc_id = docs[positive_doc_text]

            if negative_doc_text not in docs:
                docs[negative_doc_text] = f"d_{doc_counter}"
                doc_counter += 1
            negative_doc_id = docs[negative_doc_text]

            relations_list.append({'id_left': query_id, 'id_right': positive_doc_id, 'label': 1})
            relations_list.append({'id_left': query_id, 'id_right': negative_doc_id, 'label': 0})
            processed_triplets += 1

print(f"Loaded {processed_triplets} triplets.")
print(f"Unique queries: {len(queries)}, Unique docs: {len(docs)}, Relations: {len(relations_list)}")

relation_df = pd.DataFrame(relations_list)
left_df = pd.DataFrame([{'id_left': q_id, 'text_left': q_text} for q_text, q_id in queries.items()])
right_df = pd.DataFrame([{'id_right': d_id, 'text_right': d_text} for d_text, d_id in docs.items()])

left_df = left_df[['id_left', 'text_left']]
right_df = right_df[['id_right', 'text_right']]
relation_df = relation_df[['id_left', 'id_right', 'label']]

print("Creating raw DataPack...")
unique_ids_in_relation_left = set(relation_df['id_left'].unique())
unique_ids_in_left_df = set(left_df['id_left'].unique())

if not unique_ids_in_relation_left.issubset(unique_ids_in_left_df):
    print("ERROR: Mismatch in id_left!")
    missing_ids = unique_ids_in_relation_left - unique_ids_in_left_df
    print(f"IDs in relation_df but not in left_df: {missing_ids}")
    sys.exit("Exiting due to id_left inconsistency.")

try:
    train_pack_raw = mz.DataPack(
        relation=relation_df,
        left=left_df,
        right=right_df
    )
    print(f"Raw DataPack created. Relation shape: {train_pack_raw.relation.shape}, Left shape: {train_pack_raw.left.shape}, Right shape: {train_pack_raw.right.shape}")
except Exception as e:
    print(f"Error creating raw DataPack: {e}")
    sys.exit(1)

print("Defining Ranking task...")
ranking_task = mz.tasks.Ranking()

print("Initializing BasicPreprocessor...")
preprocessor = mz.preprocessors.BasicPreprocessor()
preprocessor.fixed_length_left = FIXED_LENGTH_LEFT
preprocessor.fixed_length_right = FIXED_LENGTH_RIGHT
print(f"Set fixed_length_left={FIXED_LENGTH_LEFT}, fixed_length_right={FIXED_LENGTH_RIGHT}")

print("Fitting preprocessor...")
preprocessor.fit(train_pack_raw, verbose=0)

print("Loading GloVe embeddings...")
try:
    term_index = preprocessor.context['vocab_unit'].state['term_index']
except KeyError as e:
     print(f"Error accessing term_index from preprocessor: {e}")
     sys.exit(1)
embedding_matrix = load_glove_embeddings(GLOVE_PATH, term_index, EMBEDDING_DIM)

print("Transforming data...")
train_pack_processed_internal = preprocessor.transform(train_pack_raw, verbose=0)

processed_left_df = train_pack_processed_internal.left
processed_right_df = train_pack_processed_internal.right
processed_relation_df = train_pack_processed_internal.relation

if 'id_left' in processed_left_df.columns:
    processed_left_df = processed_left_df.set_index('id_left', drop=True)
elif processed_left_df.index.name != 'id_left':
    processed_left_df.index.name = 'id_left'

if 'id_right' in processed_right_df.columns:
    processed_right_df = processed_right_df.set_index('id_right', drop=True)
elif processed_right_df.index.name != 'id_right':
    processed_right_df.index.name = 'id_right'

valid_left_ids = set(processed_left_df.index)
valid_right_ids = set(processed_right_df.index)

original_relation_count = len(processed_relation_df)
processed_relation_df = processed_relation_df[
    processed_relation_df['id_left'].isin(valid_left_ids) &
    processed_relation_df['id_right'].isin(valid_right_ids)
]
filtered_relation_count = len(processed_relation_df)
print(f"Relation filtering: Original count: {original_relation_count}, Filtered count: {filtered_relation_count}")
if original_relation_count > filtered_relation_count:
    print(f"INFO: Removed {original_relation_count - filtered_relation_count} relations due to missing IDs.")

if filtered_relation_count == 0:
    print("ERROR: No relations left after filtering. Cannot proceed.")
    sys.exit("Exiting due to empty relations after filtering.")

train_pack_for_generator = mz.DataPack(
    relation=processed_relation_df,
    left=processed_left_df,
    right=processed_right_df
)

preprocessor_save_path = os.path.join(MODEL_OUTPUT_DIR, 'preprocessor.dill')
try:
    os.makedirs(os.path.dirname(preprocessor_save_path), exist_ok=True)
    with open(preprocessor_save_path, 'wb') as f:
        dill.dump(preprocessor, f)
    print(f"Preprocessor saved to {preprocessor_save_path}")
except Exception as e:
    print(f"Error saving preprocessor: {e}")

print(f"Initializing model: {MODEL_NAME_TO_RUN}...")
if MODEL_NAME_TO_RUN == "KNRM":
    model = mz.models.KNRM()
elif MODEL_NAME_TO_RUN == "MatchLSTM":
    model = mz.models.MatchLSTM()
    model.params['hidden_size'] = 200
    model.params['dropout'] = 0.2
    model.params['lstm_layer'] = 1
    model.params['drop_lstm'] = False
    model.params['concat_lstm'] = True
    model.params['rnn_type'] = 'lstm'
else:
    print(f"Error: Model {MODEL_NAME_TO_RUN} is not currently supported or defined in the script.")
    sys.exit(1)

model.params['task'] = ranking_task

loss_function = mz.losses.RankHingeLoss(margin=HINGE_MARGIN)
evaluation_metrics = [mz.metrics.MeanAveragePrecision(), 
                        mz.metrics.NormalizedDiscountedCumulativeGain(k=3),
                        mz.metrics.NormalizedDiscountedCumulativeGain(k=5),
                        mz.metrics.MeanReciprocalRank()]

if 'loss' not in model.params:
    model.params.add(mz.Param(name='loss', value=loss_function, desc="Loss function for training"))
else:
    model.params['loss'] = loss_function

if 'metrics' not in model.params:
    model.params.add(mz.Param(name='metrics', value=evaluation_metrics, desc="Metrics for evaluation"))
else:
    model.params['metrics'] = evaluation_metrics

if MODEL_NAME_TO_RUN == "DSSM":
    if 'vocab_size' not in model.params:
        model.params.add(mz.Param(name='vocab_size', value=embedding_matrix.shape[0], desc="Vocabulary size for DSSM MLPs"))
    else:
        model.params['vocab_size'] = embedding_matrix.shape[0]

embedding_params_to_add = {
    'embedding_input_dim': embedding_matrix.shape[0],
    'embedding_output_dim': embedding_matrix.shape[1],
    'embedding_trainable': False
}

for param_name, param_value in embedding_params_to_add.items():
    if param_name not in model.params:
        model.params.add(mz.Param(name=param_name, value=param_value, desc=f"{param_name} for embedding layer"))
    else:
        model.params[param_name] = param_value

print("Guessing and filling missing params...")
model.guess_and_fill_missing_params()
try:
    embedding_matrix = torch.from_numpy(embedding_matrix).float()
    print(f"Embedding matrix dtype: {embedding_matrix.dtype}")
    
    model.build()
    print(f"Model dtype before conversion: {next(model.parameters()).dtype}")
    
    model = model.float()
    print(f"Model dtype after conversion: {next(model.parameters()).dtype}")
    
    for param in model.parameters():
        param.data = param.data.float()
    print(f"Model parameters dtype after loop: {next(model.parameters()).dtype}")

    if 'embedding_layer_name' in model.params:
        embedding_layer_name = model.params['embedding_layer_name'] 
    else:
        embedding_layer_name = 'embedding' 
    
    found_embedding_layer = False
    try:
        model.embedding.weight.data = embedding_matrix
        print(f"Successfully set weights for embedding layer: {embedding_layer_name}")
        print(f"Embedding layer dtype: {model.embedding.weight.dtype}")
        found_embedding_layer = True
    except Exception as e_set_weights:
        print(f"Error setting weights for layer {embedding_layer_name}: {e_set_weights}")
    
    if not found_embedding_layer:
        print("Warning: Could not automatically find and set weights for an embedding layer.")

except Exception as e:
    print(f"Error building model or setting embedding weights: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

class MatchZooDataset(Dataset):
    def __init__(self, data_pack, max_len_left=30, max_len_right=100):
        self.data_pack = data_pack
        self.relations = data_pack.relation.reset_index(drop=True)
        self.left = data_pack.left
        self.right = data_pack.right
        self.max_len_left = max_len_left
        self.max_len_right = max_len_right

    def __len__(self):
        return len(self.relations)

    def pad_sequence(self, seq, max_len):
        if len(seq) > max_len:
            return seq[:max_len]
        return seq + [0] * (max_len - len(seq))

    def __getitem__(self, idx):
        relation = self.relations.iloc[idx]
        left_id = relation['id_left']
        right_id = relation['id_right']
        label = relation['label']

        left_text = self.left.loc[left_id]['text_left']
        right_text = self.right.loc[right_id]['text_right']

        left_text = self.pad_sequence(left_text, self.max_len_left)
        right_text = self.pad_sequence(right_text, self.max_len_right)

        return {
            'text_left': torch.tensor(left_text, dtype=torch.long),
            'text_right': torch.tensor(right_text, dtype=torch.long),
            'label': torch.tensor(label, dtype=torch.float)
        }

print("Creating data generator...")
dataset = MatchZooDataset(train_pack_for_generator, 
                         max_len_left=FIXED_LENGTH_LEFT,
                         max_len_right=FIXED_LENGTH_RIGHT)
train_generator = DataLoader(
    dataset=dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)
if len(train_generator) == 0:
    print("Error: Data generator created 0 batches.")
    sys.exit(1)

print("Setting up optimizer and loss function...")
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = torch.nn.BCEWithLogitsLoss()

print("Starting training...")
model.train()
for epoch in range(EPOCHS):
    total_loss = 0
    for batch_idx, batch in enumerate(train_generator):
        optimizer.zero_grad()
        
        text_left = batch['text_left'].to(torch.float32)
        text_right = batch['text_right'].to(torch.float32)
        labels = batch['label'].to(torch.float32)
        
        model_input = {
            'text_left': text_left,
            'text_right': text_right
        }
        
        outputs = model(model_input)
        
        if outputs.dim() == 1:
            outputs = outputs.unsqueeze(1)
        if labels.dim() == 1:
            labels = labels.unsqueeze(1)
            
        outputs = outputs.to(torch.float32)
            
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        if batch_idx % 100 == 0:
            print(f"Epoch {epoch+1}/{EPOCHS}, Batch {batch_idx}/{len(train_generator)}, Current Loss: {loss.item():.4f}")
    
    avg_loss = total_loss / len(train_generator)
    print(f"Epoch {epoch+1}/{EPOCHS}, Average Loss: {avg_loss:.4f}")

print("Training finished.")

print(f"Saving MatchZoo model to: {MODEL_OUTPUT_DIR}")
try:
    # Save PyTorch model state dict (only save weights)
    model_save_path = os.path.join(MODEL_OUTPUT_DIR, 'model.pt')
    torch.save(model.state_dict(), model_save_path)
    print(f"PyTorch model weights saved successfully to {model_save_path}")

    # Save Keras weights for compatibility with evaluate_matchzoo.py
    weights_save_path = os.path.join(MODEL_OUTPUT_DIR, 'weights.h5')
    try:
        # Try to access backend directly
        if hasattr(model, 'backend'):
            model.backend.save_weights(weights_save_path)
            print(f"Keras weights saved successfully to {weights_save_path}")
        else:
            # If no backend attribute, try to save weights using PyTorch format
            torch.save(model.state_dict(), weights_save_path)
            print(f"Model weights saved in PyTorch format to {weights_save_path}")
    except Exception as e:
        print(f"Warning: Could not save weights in Keras format: {e}")
        print("Attempting to save weights in PyTorch format...")
        try:
            torch.save(model.state_dict(), weights_save_path)
            print(f"Model weights saved in PyTorch format to {weights_save_path}")
        except Exception as e2:
            print(f"Error saving weights in any format: {e2}")

    # Save preprocessor
    preprocessor_save_path = os.path.join(MODEL_OUTPUT_DIR, 'preprocessor.dill')
    with open(preprocessor_save_path, 'wb') as f:
        dill.dump(preprocessor, f)
    print(f"Preprocessor saved to {preprocessor_save_path}")

    # Save model config
    config_save_path = os.path.join(MODEL_OUTPUT_DIR, 'config.json')
    config = {
        'model_name': MODEL_NAME_TO_RUN,
        'batch_size': BATCH_SIZE,
        'epochs': EPOCHS,
        'learning_rate': LEARNING_RATE,
        'fixed_length_left': FIXED_LENGTH_LEFT,
        'fixed_length_right': FIXED_LENGTH_RIGHT,
        'embedding_dim': EMBEDDING_DIM,
        'model_params': {
            'hidden_size': model.params['hidden_size'] if 'hidden_size' in model.params else None,
            'dropout': model.params['dropout'] if 'dropout' in model.params else None,
            'lstm_layer': model.params['lstm_layer'] if 'lstm_layer' in model.params else None,
            'drop_lstm': model.params['drop_lstm'] if 'drop_lstm' in model.params else None,
            'concat_lstm': model.params['concat_lstm'] if 'concat_lstm' in model.params else None,
            'rnn_type': model.params['rnn_type'] if 'rnn_type' in model.params else None
        }
    }
    with open(config_save_path, 'w') as f:
        json.dump(config, f, indent=4)
    print(f"Model config saved to {config_save_path}")

except Exception as e:
    print(f"Error saving model artifacts: {e}")
    import traceback
    traceback.print_exc()

print(f"Final model artifacts should be in: {MODEL_OUTPUT_DIR}")
print(f"Contents of {MODEL_OUTPUT_DIR}: {os.listdir(MODEL_OUTPUT_DIR) if os.path.exists(MODEL_OUTPUT_DIR) else 'Directory not found'}")

print("Script finished.")
