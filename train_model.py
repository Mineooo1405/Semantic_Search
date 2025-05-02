import matchzoo as mz
import torch
import os
import pandas as pd

TRAIN_DATA_PATH = "D:/SemanticSearch/TrainingData_MatchZoo_BEIR_Semantic/msmarco_semantic_train_triplets.tsv"
MODEL_OUTPUT_DIR = "D:/SemanticSearch/TrainedModels/my_semantic_bert_model" 
PRETRAINED_MODEL_NAME = "bert-base-uncased"
BATCH_SIZE = 16 
EPOCHS = 3
LEARNING_RATE = 2e-5
TRIPLET_MARGIN = 1.0 

os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)

print("Loading training data...")
train_data_list = []
# limit = 1000 # Uncomment for quick testing
with open(TRAIN_DATA_PATH, 'r', encoding='utf-8') as f:
    for i, line in enumerate(f):
        # if i >= limit: break # Uncomment for quick testing
        parts = line.strip().split('\t')
        if len(parts) == 3:
            # Use standard MatchZoo names 'text_left' (anchor), 'text_right' (positive)
            # and add 'text_neg' for the negative sample
            train_data_list.append({
                 'text_left': parts[0],    # Anchor (Query)
                 'text_right': parts[1],   # Positive Chunk
                 'text_neg': parts[2],     # Negative Chunk
                 'id_left': f'q_{i}',      # Dummy ID
                 'id_right': f'p_{i}',     # Dummy ID
                 # 'id_neg': f'n_{i}'      # ID for negative might not be strictly needed by generator/model
             })
        # else: print(f"Skipping invalid line {i+1}") # Debug

print(f"Loaded {len(train_data_list)} triplets.")
# Ensure the DataFrame has the columns expected by the preprocessor and generator
train_pack = mz.DataPack(data=pd.DataFrame(train_data_list),
                         task=mz.tasks.Ranking()) # Specify task type for DataPack

# --- 3. Tiền xử lý ---
# Use BertPreprocessor as before
# Note: Check MatchZoo docs if a specific preprocessor for Bi-Encoders exists
preprocessor = mz.preprocessors.BertPreprocessor(model_name=PRETRAINED_MODEL_NAME)

print("Preprocessing data...")
# Explicitly tell the preprocessor which text fields to process
# Ensure these field names match the keys used in train_data_list and the generator/model expectations
train_processed = preprocessor.fit_transform(train_pack,
                                             text_fields=['text_left', 'text_right', 'text_neg'],
                                             verbose=1)
# Save preprocessor
preprocessor.save(os.path.join(MODEL_OUTPUT_DIR, 'preprocessor.dill'))
print("Preprocessor saved.")

# --- 4. Định nghĩa Task và Model ---
# Define Ranking task with TripletLoss
# Note: Metrics like MAP/NDCG are for evaluation, not direct training loss here
ranking_task = mz.tasks.Ranking(
    losses=mz.losses.TripletLoss(margin=TRIPLET_MARGIN)
    # metrics=[mz.metrics.MeanAveragePrecision()] # Can add metrics for potential evaluation later
)

# Initialize the BERT model for the ranking task
# Assuming mz.models.Bert can function as a Bi-Encoder when given a Ranking task
# It might internally create two shared-weight BERT encoders. Verify in MatchZoo docs.
model = mz.models.Bert(
    task=ranking_task,
    pretrained_model_name=PRETRAINED_MODEL_NAME
)

# Set input shapes and fill parameters (similar to DSSM example)
# The preprocessor context should contain the necessary shape information after fit_transform
model.params['input_shapes'] = preprocessor.context['input_shapes']
model.params['task'] = ranking_task # Ensure task is set
model.guess_and_fill_missing_params()
model.build()
# model.compile() # Usually not needed with mz.trainers.Trainer

print(model) # Print model summary

# --- 5. Huấn luyện ---
# Use TripletDataGenerator (check if this specific name exists, or if a generic one adapts)
# Assuming a generator that yields batches of (anchor, positive, negative) processed data
# Let's stick with TripletGenerator if it's the intended one for TripletLoss
# Ensure the 'inputs' passed match the structure expected by the generator
train_generator = mz.dataloader.TripletGenerator(
    inputs=train_processed, # Pass the processed DataPack
    batch_size=BATCH_SIZE,
    shuffle=True,
    # Ensure column names match what the generator expects, e.g., 'text_left', 'text_right', 'text_neg'
    # Check MatchZoo docs for TripletGenerator specific arguments if needed
)

# Configure Trainer (PyTorch style)
trainer = mz.trainers.Trainer(
    model=model,
    optimizer=torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE),
    trainloader=train_generator,
    epochs=EPOCHS,
    # validation_loader=val_generator, # Add validation later if needed
    # callbacks=[evaluate], # Add evaluation callback later if needed
    save_dir=MODEL_OUTPUT_DIR,
    save_all=True, # Save best model (if validation exists) and last checkpoint
    verbose=1
)

print("Starting training...")
trainer.run()

print("Training finished.")
print(f"Model and preprocessor saved in: {MODEL_OUTPUT_DIR}")

# print("Loading the best model...")
# model.load_state_dict(torch.load(os.path.join(MODEL_OUTPUT_DIR, 'model.pt')))