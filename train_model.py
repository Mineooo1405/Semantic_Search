import matchzoo as mz
import torch
import os
import pandas as pd
import sys
import dill # Import dill for saving preprocessor
import math
from datetime import datetime
import numpy as np # Needed for embedding matrix
from matchzoo import data_generator  # Import the entire dataloader module
from keras.layers import Layer # Corrected import path for Keras 2.x
# The pytorch-specific import will be handled through try/except blocks later
import keras
# Cũng kiểm tra module gốc

def load_glove_embeddings(path, term_index, embedding_dim):
    """
    Loads GloVe embeddings from a file and creates an embedding matrix.
    """
    embeddings_index = {}
    print(f"Reading GloVe file from: {path}")
    try:
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                values = line.split()
                # Ensure the line can be split correctly
                if len(values) < embedding_dim + 1:
                    # print(f"Skipping malformed line: {line.strip()}") # Optional: Debugging
                    continue
                word = values[0]
                try:
                    coefs = np.asarray(values[1:], dtype='float32')
                    # Check dimension consistency
                    if len(coefs) == embedding_dim:
                        embeddings_index[word] = coefs
                    # else: # Optional: Debugging
                        # print(f"Skipping word '{word}' with unexpected dimension {len(coefs)}")
                except ValueError:
                    # print(f"Skipping word '{word}' due to non-float value") # Optional: Debugging
                    continue
    except FileNotFoundError:
        print(f"Error: GloVe file not found at {path}")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading GloVe file: {e}")
        sys.exit(1)

    print(f'Found {len(embeddings_index)} word vectors in GloVe file.')

    # Prepare embedding matrix
    # Index 0 is usually reserved for padding/OOV in MatchZoo's VocabUnit
    num_tokens = len(term_index) + 1 # Add 1 for OOV token (index 0)
    embedding_matrix = np.zeros((num_tokens, embedding_dim))
    hits = 0
    misses = 0

    # term_index maps words to indices > 0
    for word, i in term_index.items():
        if i >= num_tokens: # Sanity check
             print(f"Warning: Index {i} for word '{word}' is out of bounds for matrix size {num_tokens}")
             continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # Words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
            hits += 1
        else:
            misses += 1
            # embedding_matrix[i] remains zeros

    print(f"Built embedding matrix: {hits} words found, {misses} words missing.")
    return embedding_matrix

# --- 1. Constants ---
TRAIN_DATA_PATH = "D:/SemanticSearch/TrainingData_MatchZoo_BEIR/msmarco_semantic-grouping/msmarco_semantic-grouping_train_triplets.tsv"
# Specify path to your GloVe file
GLOVE_PATH = "D:/SemanticSearch/embedding/glove.6B/glove.6B.100d.txt" # <<<--- !!! UPDATE THIS PATH !!!
EMBEDDING_DIM = 100 # Should match the GloVe file dimension
# MODEL_NAME_TO_RUN = "MatchLSTM" # Not available in MZ 2.1.0
MODEL_NAME_TO_RUN = "KNRM"      # Use KNRM instead
# or
# MODEL_NAME_TO_RUN = "MVLSTM"    # Use MVLSTM instead
MODEL_OUTPUT_DIR = f"D:/SemanticSearch/TrainedModels/{MODEL_NAME_TO_RUN}_mz210"
BATCH_SIZE = 128 # Can often be larger for non-BERT models
EPOCHS = 5 # Adjust as needed
LEARNING_RATE = 1e-3 # Common starting point for these models
HINGE_MARGIN = 1.0
# Fixed lengths for preprocessor (adjust based on your data analysis)
FIXED_LENGTH_LEFT = 30
FIXED_LENGTH_RIGHT = 100

os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)

# --- Check GloVe Path ---
if not os.path.exists(GLOVE_PATH):
    print(f"Error: GloVe file not found at {GLOVE_PATH}")
    print("Please download GloVe (e.g., glove.6B.100d.txt) and update GLOVE_PATH in the script.")
    sys.exit(1)

# --- 2. Load and Prepare Data for Ranking Task ---
# MatchLSTM/KNRM with RankHingeLoss typically expect pairs with labels (1 for pos, 0 for neg)
print("Loading training data and creating pairs...")
ranking_data_list = []
# limit = 1000 # Uncomment for quick testing
processed_triplets = 0
with open(TRAIN_DATA_PATH, 'r', encoding='utf-8') as f:
    for i, line in enumerate(f):
        # if i >= limit: break # Uncomment for quick testing
        parts = line.strip().split('\t')
        if len(parts) == 3:
            anchor, positive, negative = parts[0], parts[1], parts[2]
            # Add positive pair with label 1
            ranking_data_list.append({
                'text_left': anchor,
                'text_right': positive,
                'label': 1,
                'id_left': f'q_{i}',
                'id_right': f'p_{i}'
            })
            # Add negative pair with label 0
            ranking_data_list.append({
                'text_left': anchor,
                'text_right': negative,
                'label': 0,
                'id_left': f'q_{i}',
                'id_right': f'n_{i}'
            })
            processed_triplets += 1
        # else: print(f"Skipping invalid line {i+1}") # Debug

print(f"Loaded {processed_triplets} triplets, created {len(ranking_data_list)} pairs.")
train_df = pd.DataFrame(ranking_data_list)
# Pack the DataFrame into MatchZoo's DataPack format
# Ensure column names match what PairGenerator expects (text_left, text_right, label)
train_pack = mz.pack(train_df)

# --- 3. Define Task ---
# Define the Ranking task with RankHingeLoss
print("Defining Ranking task with RankHingeLoss...")
try:
    # Try initializing Ranking by passing the loss instance with the 'loss' keyword
    ranking_task = mz.tasks.Ranking(
        loss=mz.losses.RankHingeLoss(margin=HINGE_MARGIN)
        # Optionally assign metrics here if needed
        # metrics=[mz.metrics.MeanAveragePrecision(), mz.metrics.NormalizedDiscountedCumulativeGain(k=3)]
    )
    print("Initialized Ranking task with RankHingeLoss.")

except TypeError as e:
    # Handle case where 'loss' keyword is also incorrect
    print(f"TypeError during Ranking task initialization: {e}")
    print("Could not initialize Ranking task with loss. Trying assignment...")
    try:
        # Fallback: Try assignment again (though it failed before)
        ranking_task = mz.tasks.Ranking()
        ranking_task.loss = mz.losses.RankHingeLoss(margin=HINGE_MARGIN)
        print("Assigned RankHingeLoss to task after initialization.")
    except Exception as assign_e:
        print(f"Failed to assign loss after initialization: {assign_e}")
        print("Please check MatchZoo 2.1.0 documentation for correct task/loss definition.")
        sys.exit(1)
except AttributeError as e:
    # Handle case where Ranking or RankHingeLoss is not found
    print(f"AttributeError defining task or loss: {e}")
    print("Please check your MatchZoo version (expected 2.1.0) and its installation.")
    sys.exit(1)
except Exception as e:
    print(f"An unexpected error occurred during task definition: {e}")
    sys.exit(1)


# Assign the task to the datapack (optional for some generators, but good practice)
train_pack.task = ranking_task

# --- 4. Preprocessing ---
# Use BasicPreprocessor for non-BERT models
print("Initializing BasicPreprocessor...")
# Initialize without fixed_length arguments
preprocessor = mz.preprocessors.BasicPreprocessor(
    # remove_stop_words=True # Optional
)
# Set fixed lengths after initialization
preprocessor.fixed_length_left = FIXED_LENGTH_LEFT
preprocessor.fixed_length_right = FIXED_LENGTH_RIGHT
print(f"Set fixed_length_left={FIXED_LENGTH_LEFT}, fixed_length_right={FIXED_LENGTH_RIGHT}")


print("Fitting preprocessor...")
# Fit on the data to build vocabulary
preprocessor.fit(train_pack, verbose=1)

print("Loading GloVe embeddings manually...")
# Get the term_index dictionary from the fitted preprocessor
# Ensure the path to the vocab_unit and state is correct for MZ 2.1.0
try:
    term_index = preprocessor.context['vocab_unit'].state['term_index']
except KeyError as e:
     print(f"Error accessing term_index in preprocessor context: {e}")
     print("Preprocessor context structure might have changed. Please inspect preprocessor.context.")
     sys.exit(1)
except Exception as e:
     print(f"An unexpected error occurred accessing term_index: {e}")
     sys.exit(1)

# Call the manual loading function
embedding_matrix = load_glove_embeddings(GLOVE_PATH, term_index, EMBEDDING_DIM)
print(f"Embedding matrix shape: {embedding_matrix.shape}")

print("Transforming data...")
# Transform the data using the fitted preprocessor
try:
    # Specify the text fields to process
    train_processed = preprocessor.transform(train_pack, verbose=1)
except Exception as e:
    print(f"Error during preprocessing transform: {e}")
    sys.exit(1)

# Save preprocessor using dill
preprocessor_save_path = os.path.join(MODEL_OUTPUT_DIR, 'preprocessor.dill')
print(f"Saving preprocessor to {preprocessor_save_path}...")
try:
    with open(preprocessor_save_path, 'wb') as f:
        dill.dump(preprocessor, f)
    print(f"Preprocessor saved successfully.")
except Exception as e:
    print(f"Error saving preprocessor: {e}")

# --- 5. Initialize Model ---
print(f"Initializing model: {MODEL_NAME_TO_RUN}...")

try:
    if MODEL_NAME_TO_RUN == "MatchLSTM":
        # Import directly from matchzoo.models
        model = mz.models.MatchLSTM()
        print("Using MatchLSTM from matchzoo.models")
    elif MODEL_NAME_TO_RUN == "KNRM":
        # Import directly from matchzoo.models
        model = mz.models.KNRM()
        print("Using KNRM from matchzoo.models")
        # KNRM specific params (example, check docs)
        # model.params['kernel_num'] = 11
        # model.params['sigma'] = 0.1
        # model.params['exact_sigma'] = 0.001
    else:
        print(f"Error: Unknown model name '{MODEL_NAME_TO_RUN}'")
        sys.exit(1)

except AttributeError:
    # This error now means the model is truly missing from matchzoo.models
    print(f"Error: Could not find {MODEL_NAME_TO_RUN} in matchzoo.models.")
    print("Please ensure MatchZoo 2.1.0 is correctly installed and contains the model.")
    sys.exit(1)
except Exception as e:
    print(f"An unexpected error occurred during model initialization: {e}")
    sys.exit(1)


# Set common model parameters
model.params['task'] = ranking_task
# model.params['embedding'] = embedding_matrix # <<< REMOVE OR COMMENT OUT THIS LINE
model.params['embedding_input_dim'] = embedding_matrix.shape[0] # Vocab size
model.params['embedding_output_dim'] = embedding_matrix.shape[1] # Embedding dimension
model.params['embedding_trainable'] = False # <<< ADD THIS LINE to use pre-trained weights without fine-tuning

model.guess_and_fill_missing_params() # Let this handle parameter setup
print("Building model...")
try:
    model.build() # Model build should infer shapes

    # --- Potential Step: Manually set weights AFTER build ---
    # If build() doesn't automatically use the matrix, you might need this:
    # try:
    #     # Find the embedding layer (name might vary)
    #     embedding_layer = model._backend.get_layer(index=1) # Or find by name='embedding'
    #     if isinstance(embedding_layer, keras.layers.Embedding):
    #          print(f"Setting weights for embedding layer: {embedding_layer.name}")
    #          embedding_layer.set_weights([embedding_matrix])
    #     else:
    #          print("Warning: Could not find Keras Embedding layer to set weights.")
    # except Exception as set_weight_e:
    #     print(f"Warning: Failed to manually set embedding weights: {set_weight_e}")
    # ---------------------------------------------------------

except Exception as e:
    print(f"Error building model: {e}")
    sys.exit(1)

print("Model Structure:")
print(model) # Print model summary

# --- 6. Create Data Generator ---
print("Creating data generator...")
try:
    # Use the main DataGenerator class with mode='pair' for ranking tasks
    train_generator = mz.data_generator.DataGenerator( # Use DataGenerator class
        data_pack=train_processed,       # Pass the processed DataPack
        mode='pair',                  # Specify pair mode for ranking
        num_dup=1,                    # Default value, adjust if needed
        num_neg=1,                    # Default value, adjust if needed
        batch_size=BATCH_SIZE,
        shuffle=True
        # Removed stage='train' as it might not be a param for DataGenerator.__init__
    )
    print(f"Generator created with batch size {BATCH_SIZE} using mz.data_generator.DataGenerator(mode='pair').")

except AttributeError:
    # This error would now indicate a problem finding DataGenerator itself
    print("Error: Could not find DataGenerator in mz.data_generator.")
    print("Please check your MatchZoo version (expected 2.1.0) and its installation.")
    sys.exit(1)
except Exception as e:
    print(f"Error creating data generator: {e}")
    sys.exit(1)

# Check the number of batches
print(f"Number of batches in generator: {len(train_generator)}")
if len(train_generator) == 0:
    print("Error: Data generator created 0 batches. Check data processing and batch size.")
    sys.exit(1)

# --- 7. Compile and Train Model using Keras API ---
print("Compiling model with Keras optimizer...")
try:
    # Use a Keras optimizer (Adam is common)
    optimizer = keras.optimizers.Adam(learning_rate=LEARNING_RATE)

    # Compile the underlying Keras model directly via model._backend
    # The MatchZoo model wrapper might have its own compile method with a different signature.
    if hasattr(model, '_backend') and hasattr(model._backend, 'compile'):
        print("Compiling the underlying Keras model (model._backend)...")
        # Call compile on the Keras model, passing the loss function from the task
        model._backend.compile(optimizer=optimizer, loss=ranking_task.loss)
        print("Underlying Keras model compiled successfully.")
    else:
        print("Error: Could not find a Keras backend model or its compile method.")
        sys.exit(1)

except AttributeError as e:
     print(f"Error accessing model._backend or task.loss: {e}")
     print("Model structure or task definition might be different than expected.")
     sys.exit(1)
except Exception as e:
    print(f"Error compiling model: {e}")
    sys.exit(1)

print("Starting training using model.fit()...")
try:
    # Define Keras callbacks if needed (e.g., ModelCheckpoint)
    checkpoint_path = os.path.join(MODEL_OUTPUT_DIR, 'keras_model_epoch_{epoch:02d}.h5')
    model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        save_weights_only=True, # Save entire model (weights are saved within the .h5 file)
        monitor='loss', # Monitor training loss
        mode='min',
        save_best_only=False) # Save at each epoch end

    # Call fit on the underlying Keras model directly, as train_generator is a Keras Sequence
    if hasattr(model, '_backend') and hasattr(model._backend, 'fit'):
        print("Calling fit on the underlying Keras model (model._backend)...")
        history = model._backend.fit( # <<< CALL FIT ON THE BACKEND MODEL
            train_generator,
            epochs=EPOCHS,
            callbacks=[model_checkpoint_callback], # Add callbacks here
            verbose=1
        )
        print("Training finished.")
        print(f"Keras model checkpoints saved in: {MODEL_OUTPUT_DIR}")
    else:
        print("Error: Could not find a Keras backend model or its fit method.")
        sys.exit(1)

except Exception as e:
    print(f"An error occurred during training: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ...sau khi train xong...
model.save(MODEL_OUTPUT_DIR)
print(f"MatchZoo model saved to {MODEL_OUTPUT_DIR}")

exit() # Exit after training for now


