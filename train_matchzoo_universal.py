\
import argparse
import os
import sys
import json
import random
import traceback
import time
from datetime import datetime
import numpy as np
import pandas as pd
import torch
import dill
import matchzoo as mz
from sklearn.preprocessing import normalize
from typing import Optional, Tuple
# MODIFIED: Removed import for read_embedding_dim

# ---------- Utility Functions ---------- #
def log(msg):
    """Prints a message with a timestamp."""
    print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] {msg}')

def set_seeds(seed_value):
    """Sets random seeds for reproducibility."""
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)

def load_data_pack_from_tsv(path: str, task): 
    """Loads data from a TSV file and converts it to a MatchZoo DataPack."""
    if not os.path.exists(path):
        log(f"Warning: File not found at {path}. Returning None.")
        return None
    try:
        # Expecting columns: query, positive_doc, negative_doc without header
        df = pd.read_csv(path, sep='\\t', header=None, names=["query", "positive_doc", "negative_doc"], engine='python')
    except Exception as e:
        log(f"Error reading TSV file {path}: {e}")
        return None

    records = []
    for idx, row in df.iterrows():
        query_text = str(row["query"])
        pos_doc_text = str(row["positive_doc"])
        neg_doc_text = str(row["negative_doc"])

        query_id = f"Q{idx:06d}"
        # Create unique doc IDs for this query's positive and negative instance
        # This helps if the same document text appears for different queries
        pos_doc_id = f"D{idx:06d}_P"
        neg_doc_id = f"D{idx:06d}_N"

        records.append([query_id, query_text, pos_doc_id, pos_doc_text, 1])  # Positive pair
        records.append([query_id, query_text, neg_doc_id, neg_doc_text, 0])  # Negative pair

    if not records:
        log(f"Warning: No records generated from {path}. Returning None.")
        return None

    df_records = pd.DataFrame(records, columns=["id_left", "text_left", "id_right", "text_right", "label"])
    return mz.pack(df_records, task=task)

def get_embedding_dim_from_file(embed_path: str, embed_format: str) -> int:
    """Reads an embedding file to determine its dimension."""
    log(f"Attempting to determine embedding dimension from file: {embed_path} (format: {embed_format})")
    try:
        with open(embed_path, 'r', encoding='utf-8') as f:
            first_line = f.readline().strip()
            if not first_line:
                raise ValueError("Embedding file is empty or first line is blank.")
            
            parts = first_line.split()
            # The first part is the word, the rest are vector components
            dimension = len(parts) - 1
            if dimension <= 0:
                raise ValueError(f"Could not determine a valid dimension from line: '{first_line}'. Dimension found: {dimension}")
            log(f"Determined embedding dimension: {dimension} from first line.")
            return dimension
    except Exception as e:
        log(f"❌ Error reading embedding dimension from {embed_path}: {e}")
        log(f"Falling back to a default dimension of 100. THIS MIGHT BE INCORRECT.")
        return 100 # Fallback, though this is risky if it doesn't match the actual file

def is_valid_dev_pack(data_pack: Optional[mz.DataPack]) -> bool:
    """Checks if a dev DataPack is valid for ranking (each query has pos and neg examples)."""
    if data_pack is None or data_pack.relation is None or data_pack.relation.empty:
        return False
    
    # Check if 'label' column exists and is numeric
    if 'label' not in data_pack.relation.columns:
        log("Warning: 'label' column missing in dev_pack.relation.")
        return False
    if not pd.api.types.is_numeric_dtype(data_pack.relation['label']):
        log("Warning: 'label' column in dev_pack.relation is not numeric.")
        # Attempt to convert, assuming it might be string "0" or "1"
        try:
            data_pack.relation['label'] = pd.to_numeric(data_pack.relation['label'])
        except ValueError:
            log("Error: Could not convert 'label' column to numeric. Dev pack invalid.")
            return False

    try:
        grouped = data_pack.relation.groupby('id_left')['label'].apply(lambda x: set(x.unique()))
        if grouped.empty:
             log("Warning: Dev pack is empty after grouping by id_left.")
             return False
        for labels in grouped:
            if 0 not in labels or 1 not in labels:
                log("Warning: A query in dev_pack lacks either positive or negative examples.")
                return False
        return True
    except Exception as e:
        log(f"Error during dev_pack validation: {e}")
        return False


def load_and_normalize_embedding(embed_path: Optional[str], embed_format: Optional[str],
                                 term_index: dict, default_embed_dim: int) -> Tuple[Optional[np.ndarray], int]:
    """Loads embeddings from file, L2 normalizes them, or returns info for random embeddings."""
    embedding_matrix_to_return = None
    final_dimension = default_embed_dim

    if embed_path and os.path.exists(embed_path):
        log(f"Attempting to load and build custom embedding from: {embed_path} (format: {embed_format}) for term_index size {len(term_index)}")
        try:
            # 1. Determine the actual embedding dimension from the pre-trained file
            actual_embed_dim_from_file = get_embedding_dim_from_file(embed_path, embed_format) # MODIFIED: Use local helper
            log(f"Detected embedding dimension {actual_embed_dim_from_file} from file {embed_path}.")
            final_dimension = actual_embed_dim_from_file

            # 2. Create an Embedding module instance
            # padding_idx=0 assumes that 0 in term_index is the padding token.
            # Ensure preprocessor.context['vocab_unit'].state['padding_idx'] aligns if it exists.
            embedding_module = mz.embedding.Embedding(
                output_dim=actual_embed_dim_from_file,
                padding_idx=0 # Standard padding index
            )

            # 3. Build the embedding matrix using the dataset's term_index and the pre-trained file
            # This populates embedding_module.weight with a matrix of shape (len(term_index), actual_embed_dim_from_file)
            embedding_module.build(
                term_index=term_index,
                embed_file=embed_path,
                embed_mode=embed_format
            )
            
            loaded_embedding_matrix_np = embedding_module.weight.data.cpu().numpy()
            log(f"Successfully built custom embedding matrix of shape {loaded_embedding_matrix_np.shape}.")

            # Optional: L2 Normalization (example)
            # norms = np.linalg.norm(loaded_embedding_matrix_np, axis=1, keepdims=True)
            # norms[norms == 0] = 1 # Avoid division by zero for zero vectors
            # embedding_matrix_to_return = loaded_embedding_matrix_np / norms
            # log("Applied L2 normalization to custom embeddings.")
            embedding_matrix_to_return = loaded_embedding_matrix_np # Using raw embeddings for now

        except Exception as e:
            log(f"❌ Error loading or building custom embedding matrix from {embed_path}: {e}")
            log(traceback.format_exc())
            log(f"Falling back to random embeddings with dim {default_embed_dim}.")
            # Fall-through to random initialization by keeping embedding_matrix_to_return = None
            # and final_dimension = default_embed_dim (already set)
            embedding_matrix_to_return = None # Explicitly set to None on error
            final_dimension = default_embed_dim # Revert to default_embed_dim on error
    
    if embedding_matrix_to_return is None:
        log(f"Using random embeddings with dimension {default_embed_dim}.")
        # Ensure vocab_size for random embeddings is len(term_index)
        # term_index usually starts from 1, 0 is padding. Max index = vocab_size - 1.
        # So, if len(term_index) is N, indices are 0 to N-1.
        vocab_size_for_random = len(term_index)
        embedding_matrix_to_return = np.random.uniform(-0.2, 0.2, (vocab_size_for_random, default_embed_dim)).astype(np.float32)
        final_dimension = default_embed_dim # Ensure this is set if we fell back

    return embedding_matrix_to_return, final_dimension

# ---------- Model Definitions and Hyperparameters ---------- #
MODEL_CLASSES = [
    mz.models.KNRM, mz.models.MatchLSTM, mz.models.ArcI,
    mz.models.MatchPyramid, mz.models.DRMM, mz.models.ConvKNRM
]
MODEL_NAMES = [
    "KNRM", "MatchLSTM", "ArcI",
    "MatchPyramid", "DRMM", "ConvKNRM"
]

# Optional: Define model-specific hyperparameters
MODEL_SPECIFIC_HYPERPARAMS = {
    'KNRM': {'kernel_num': 21, 'sigma': 0.1, 'exact_sigma': 0.001},
    'MatchPyramid': {'kernel_count': [16, 32], 'kernel_size': [[3, 3], [3, 3]], 'dpool_size': [3, 10]},
    'DRMM': {'top_k': 20, 'mlp_num_layers': 1, 'mlp_num_units': 5, 'mlp_num_fan_out': 1, 'mlp_activation_func': 'relu', 'hist_bins': 30}, # MODIFIED: Added hist_bins
    'ConvKNRM': {'filters': 128, 'conv_activation_func': 'tanh', 'max_ngram': 3} # Example, adjust as needed
    # Add other models if specific hyperparams are desired beyond defaults
}

# ---------- Main Training Script ---------- #
def main():
    parser = argparse.ArgumentParser(description="Universal MatchZoo-py Model Trainer")
    parser.add_argument('--train_path', type=str, required=True, help="Path to training TSV file (query, positive_doc, negative_doc)")
    parser.add_argument('--dev_path', type=str, default=None, help="Path to development TSV file (optional)")
    parser.add_argument('--embed_path', type=str, default=None, help="Path to custom pre-trained embedding file (optional)")
    parser.add_argument('--embed_format', type=str, default='glove', choices=['glove', 'word2vec', 'fasttext'],
                        help="Format of the custom embedding file (required if --embed_path is set)")
    parser.add_argument('--embed_dim_random', type=int, default=100, help="Dimension for random embeddings if --embed_path is not used")
    parser.add_argument('--output_dir', type=str, default="./matchzoo_trained_models", help="Root directory to save trained models and artifacts")
    parser.add_argument('--start_model', type=str, choices=MODEL_NAMES, default=None, help="Model name to start training from (skips previous models if --model_names_to_run is not used).")
    parser.add_argument('--model_names_to_run', type=str, nargs='+', choices=MODEL_NAMES, default=MODEL_NAMES, help="List of model names to train (e.g., DRMM ConvKNRM). If not specified, all models will be trained or those from --start_model onwards.")
    parser.add_argument('--epochs', type=int, default=10, help="Number of training epochs")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size for training and validation")
    parser.add_argument('--lr', type=float, default=1e-3, help="Learning rate for Adam optimizer")
    parser.add_argument('--patience', type=int, default=3, help="Patience for early stopping (if dev set is used)")
    parser.add_argument('--seed', type=int, default=2025, help="Random seed for reproducibility")
    parser.add_argument('--max_len_left', type=int, default=30, help="Max sequence length for left text (queries)")
    parser.add_argument('--max_len_right', type=int, default=100, help="Max sequence length for right text (documents)")
    parser.add_argument('--early_stopping_metric', type=str, default='map', choices=['map', 'ndcg@3', 'ndcg@5'], help="Metric for early stopping (alias like 'map' or 'ndcg@3')")

    args = parser.parse_args()

    set_seeds(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log(f"Using device: {device}")

    # 1. Define Ranking Task and Metrics
    log("Defining ranking task and metrics...")
    ranking_task = mz.tasks.Ranking(losses=mz.losses.RankCrossEntropyLoss(num_neg=1)) # num_neg=1 for pairwise
    metrics_to_use = [
        mz.metrics.MeanAveragePrecision(),      # Default threshold is 0.0, name() -> 'mean_average_precision(0.0)'
        mz.metrics.NormalizedDiscountedCumulativeGain(k=3), # Default threshold is 0.0, name() -> 'normalized_discounted_cumulative_gain@3(0.0)'
        mz.metrics.NormalizedDiscountedCumulativeGain(k=5)  # Default threshold is 0.0, name() -> 'normalized_discounted_cumulative_gain@5(0.0)'
    ]
    ranking_task.metrics = metrics_to_use
    
    # Determine the actual metric key for the Trainer's early stopping.
    # This key must match the keys in the 'result' dictionary from validation,
    # which are generated by metric.name().
    user_alias_for_early_stopping = args.early_stopping_metric
    actual_metric_key_for_trainer = None

    if user_alias_for_early_stopping == 'map':
        metric_instance = next((m for m in metrics_to_use if isinstance(m, mz.metrics.MeanAveragePrecision)), None)
        if metric_instance:
            actual_metric_key_for_trainer = metric_instance.name()
    elif user_alias_for_early_stopping.startswith('ndcg@'):
        try:
            k_val = int(user_alias_for_early_stopping.split('@')[1])
            metric_instance = next((m for m in metrics_to_use if isinstance(m, mz.metrics.NormalizedDiscountedCumulativeGain) and getattr(m, '_k', None) == k_val), None)
            if metric_instance:
                actual_metric_key_for_trainer = metric_instance.name()
        except ValueError:
            log(f"❌ Invalid format for ndcg alias: {user_alias_for_early_stopping}")
            sys.exit(1)

    if not actual_metric_key_for_trainer:
        valid_choices_from_metrics = [m.name() for m in metrics_to_use] # These are the real keys
        log(f"❌ Could not map early stopping alias '{user_alias_for_early_stopping}' to a valid metric name used by the Trainer. "
            f"Please ensure '{user_alias_for_early_stopping}' corresponds to one of the configured metrics. "
            f"Available actual metric keys from metrics_to_use: {valid_choices_from_metrics}. "
            f"User choices for --early_stopping_metric are: {parser.get_default('early_stopping_metric')}") # Show choices
        # Try to find the choices for the argument to give better guidance
        for action in parser._actions:
            if action.dest == 'early_stopping_metric':
                log(f"Valid aliases for --early_stopping_metric: {action.choices}")
                break
        sys.exit(1)
    
    log(f"✅ Early stopping will monitor Trainer metric key: '{actual_metric_key_for_trainer}' (derived from alias '{user_alias_for_early_stopping}')")

    # 2. Load Data
    log(f"Loading training data from: {args.train_path}")
    train_pack = load_data_pack_from_tsv(args.train_path, task=ranking_task)
    if train_pack is None or train_pack.left.empty: # Check if train_pack is empty
        log("Error: Training data could not be loaded or is empty. Exiting.")
        sys.exit(1)

    dev_pack = None
    if args.dev_path:
        log(f"Loading development data from: {args.dev_path}")
        dev_pack = load_data_pack_from_tsv(args.dev_path, task=ranking_task)

    # 3. Preprocessing
    log("Initializing preprocessor...")
    # Using ArcI's default preprocessor as a common base, with configurable truncation
    preprocessor = mz.preprocessors.BasicPreprocessor( # ArcI.get_default_preprocessor() is often BasicPreprocessor based
        truncated_length_left=args.max_len_left,
        truncated_length_right=args.max_len_right,
        filter_low_freq=2 # Common practice
    )
    # Alternative: preprocessor = mz.models.ArcI.get_default_preprocessor(truncated_length_left=args.max_len_left, truncated_length_right=args.max_len_right)


    log("Fitting preprocessor on training data...")
    # Fit on train_pack.relation, then transform train_pack and dev_pack
    # MatchZoo preprocessor API: fit(data_pack), transform(data_pack)
    train_processed = preprocessor.fit_transform(train_pack, verbose=0)
    
    dev_processed = None
    use_dev_set_for_validation = False
    if dev_pack and not dev_pack.left.empty:
        log("Transforming development data...")
        dev_processed = preprocessor.transform(dev_pack, verbose=0)
        if is_valid_dev_pack(dev_processed):
            log("Development data is valid for ranking evaluation and early stopping.")
            use_dev_set_for_validation = True
        else:
            log("Warning: Development data is NOT valid for ranking (e.g., missing pos/neg pairs for some queries). Early stopping will be disabled.")
            dev_processed = None # Do not use invalid dev set
    else:
        log("No development data provided or it's empty. Early stopping will be disabled.")

    # 4. Load or Initialize Embeddings
    if 'vocab_unit' in preprocessor.context and hasattr(preprocessor.context['vocab_unit'], 'state'):
        term_index = preprocessor.context['vocab_unit'].state['term_index']
    else:
        log("❌ Could not extract 'term_index' from preprocessor context. Ensure your preprocessor is fitted correctly.")
        sys.exit(1)

    vocab_size = len(term_index)
    log(f"Vocabulary size: {vocab_size}")

    embedding_matrix, final_embed_dim = load_and_normalize_embedding(
        args.embed_path, args.embed_format, term_index, args.embed_dim_random
    )
    log(f"Final embedding dimension to be used by models: {final_embed_dim}")


    # 5. Training Loop for Each Model
    models_to_process = []
    if args.model_names_to_run and args.model_names_to_run != MODEL_NAMES: # User specified a list
        models_to_process = [name for name in args.model_names_to_run if name in MODEL_NAMES]
        log(f"Explicitly training models: {models_to_process}")
    else: # Default: use start_model or all
        start_idx = MODEL_NAMES.index(args.start_model) if args.start_model and args.start_model in MODEL_NAMES else 0
        models_to_process = MODEL_NAMES[start_idx:]
        log(f"Training models from index {start_idx}: {models_to_process}")

    for model_name in models_to_process:
        model_cls_idx = MODEL_NAMES.index(model_name) # Get the original index for MODEL_CLASSES
        model_cls = MODEL_CLASSES[model_cls_idx]
        current_model_output_dir = os.path.join(args.output_dir, model_name)
        os.makedirs(current_model_output_dir, exist_ok=True)

        success_flag_path = os.path.join(current_model_output_dir, "_SUCCESS")
        if os.path.exists(success_flag_path):
            log(f"Model {model_name} already trained (found _SUCCESS flag). Skipping.")
            continue

        log(f"===== Training Model: {model_name} =====")

        # 1. Configure Padding Callback (Initial)
        log(f"Configuring padding callback for {model_name}...")
        padding_callback = None
        current_cb_type_name = "None"
        try:
            # Get the default padding callback instance from the model class
            padding_callback = model_cls.get_default_padding_callback()
            current_cb_type_name = type(padding_callback).__name__
            log(f"Retrieved default padding callback: {current_cb_type_name}")

            # Update fixed lengths to match args, if the callback has these attributes
            if hasattr(padding_callback, '_fixed_length_left'):
                padding_callback._fixed_length_left = args.max_len_left
            if hasattr(padding_callback, '_fixed_length_right'):
                padding_callback._fixed_length_right = args.max_len_right
            log(f"Ensured padding callback lengths are L:{args.max_len_left}, R:{args.max_len_right}")

            # Configure term_index for specific callbacks that use it (e.g., for n-grams)
            # MODIFIED: Safely check types for isinstance to avoid AttributeError if a type doesn't exist
            types_needing_term_index_setup = []
            if hasattr(mz.dataloader.callbacks, 'DRMMPadding'):
                types_needing_term_index_setup.append(mz.dataloader.callbacks.DRMMPadding)
            if hasattr(mz.dataloader.callbacks, 'ConvKNRMPadding'): # Only add if it exists
                types_needing_term_index_setup.append(mz.dataloader.callbacks.ConvKNRMPadding)
            
            if types_needing_term_index_setup and isinstance(padding_callback, tuple(types_needing_term_index_setup)):
                if hasattr(padding_callback, 'set_term_index') and callable(getattr(padding_callback, 'set_term_index')):
                    padding_callback.set_term_index(term_index)
                    log(f"Called set_term_index on {current_cb_type_name}.")
                elif hasattr(padding_callback, '_term_index'): # Fallback for direct attribute access
                    padding_callback._term_index = term_index
                    log(f"Set _term_index on {current_cb_type_name}.")
                else:
                    log(f"Warning: {current_cb_type_name} does not have set_term_index or _term_index for term_index configuration.")
            
        except Exception as e_cb_config:
            log(f"Warning: Error during padding callback configuration for {model_name}: {e_cb_config}. This might happen if a specific callback (e.g., ConvKNRMPadding) is not found. Falling back to BasicPadding.")
            padding_callback = mz.dataloader.callbacks.BasicPadding(
                fixed_length_left=args.max_len_left,
                fixed_length_right=args.max_len_right
            )
            current_cb_type_name = type(padding_callback).__name__
            log(f"Using BasicPadding for {model_name}.")
        
        log(f"Initial padding callback: {current_cb_type_name}")

        # 2. Initialize Model and Set Initial Parameters
        model = model_cls()
        model.params['task'] = ranking_task
        
        if model_name == "ArcI":
            log(f"Explicitly setting left_length and right_length for ArcI from args: L={args.max_len_left}, R={args.max_len_right}")
            # ArcI specific parameters that need to align with preprocessor's truncation
            if 'left_length' in model.params:
                model.params['left_length'] = args.max_len_left
            if 'right_length' in model.params:
                model.params['right_length'] = args.max_len_right

        if embedding_matrix is not None:
            model.params['embedding'] = embedding_matrix
            model.params['embedding_input_dim'] = vocab_size 
            model.params['embedding_output_dim'] = final_embed_dim
            model.params['embedding_freeze'] = False # Or make this a CLI arg
        else: # Random embeddings
            model.params['embedding_input_dim'] = vocab_size
            model.params['embedding_output_dim'] = final_embed_dim # This is args.embed_dim_random
            model.params['embedding_freeze'] = False
        
        # 3. Apply Model-Specific Hyperparameters
        if model_name in MODEL_SPECIFIC_HYPERPARAMS:
            log(f"Applying specific hyperparameters for {model_name}: {MODEL_SPECIFIC_HYPERPARAMS[model_name]}")
            model.params.update(MODEL_SPECIFIC_HYPERPARAMS[model_name])

        # 4. Further Configure Padding Callback (e.g., DRMM hist_size from model.params)
        # This is done before guess_and_fill, as hist_bins is a primary param for DRMM.
        if model_name == "DRMM":
            DRMMPadding_cls = getattr(mz.dataloader.callbacks, 'DRMMPadding', None)
            if DRMMPadding_cls and isinstance(padding_callback, DRMMPadding_cls):
                drmm_default_hist_bins = 30 
                # Use get with a default of None to avoid KeyError if 'hist_bins' isn't in params yet
                hist_bins_param = model.params.get('hist_bins', None) 
                model_hist_bins = hist_bins_param.value if hist_bins_param else drmm_default_hist_bins
                
                current_padding_hist_size = getattr(padding_callback, '_hist_size', 'N/A') # Get current for logging
                
                # Check if update is needed
                needs_update = False
                if hasattr(padding_callback, '_hist_size'):
                    if padding_callback._hist_size != model_hist_bins:
                        needs_update = True
                elif hasattr(padding_callback, 'set_hist_size'): # If only setter exists, assume update might be needed
                    needs_update = True 
                # If neither, it cannot be set, so no update needed.

                if needs_update:
                    log(f"Attempting to update DRMMPadding hist_size. Current: {current_padding_hist_size}, Target (from model.params or default): {model_hist_bins}.")
                    if hasattr(padding_callback, 'set_hist_size') and callable(getattr(padding_callback, 'set_hist_size')):
                        log(f"Calling DRMMPadding set_hist_size({model_hist_bins}).")
                        padding_callback.set_hist_size(model_hist_bins)
                    elif hasattr(padding_callback, '_hist_size'): # Fallback to direct attribute set if setter not found
                        log(f"Setting DRMMPadding _hist_size = {model_hist_bins}.")
                        padding_callback._hist_size = model_hist_bins
                    else:
                        log(f"Warning: DRMMPadding hist_size ({current_padding_hist_size}) cannot be updated to {model_hist_bins} as neither 'set_hist_size' method nor '_hist_size' attribute is available for setting.")
                else:
                    log(f"DRMMPadding hist_size ({current_padding_hist_size}) already matches target ({model_hist_bins}) or cannot be set.")
        
        # 5. Finalize Model Parameters and Build Model
        try:
            model.guess_and_fill_missing_params(verbose=0)
            log(f"Model parameters for {model_name} after guess_and_fill: {str(model.params)}") # MODIFIED: Use str()

            # If guess_and_fill changed hist_bins for DRMM, ensure padding callback reflects it.
            if model_name == "DRMM":
                DRMMPadding_cls = getattr(mz.dataloader.callbacks, 'DRMMPadding', None)
                if DRMMPadding_cls and isinstance(padding_callback, DRMMPadding_cls):
                    # Correctly get hist_bins after guess_and_fill
                    final_hist_bins_param = model.params.get('hist_bins')
                    final_model_hist_bins = final_hist_bins_param.value if final_hist_bins_param else None # Can be None if not set by guess_and_fill

                    current_padding_hist_size = getattr(padding_callback, '_hist_size', None)
                    if final_model_hist_bins is not None and current_padding_hist_size != final_model_hist_bins:
                        log(f"Re-updating DRMMPadding _hist_size from {current_padding_hist_size} to {final_model_hist_bins} (from finalized model.params).")
                        if hasattr(padding_callback, 'set_hist_size') and callable(getattr(padding_callback, 'set_hist_size')):
                            padding_callback.set_hist_size(final_model_hist_bins)
                        elif hasattr(padding_callback, '_hist_size'):
                             padding_callback._hist_size = final_model_hist_bins


            log(f"Building model {model_name}...")
            model.build()
            model.to(device)
            log(f"Model {model_name} built and moved to device.")
        except Exception as e:
            log(f"Error during guess_and_fill_missing_params or model.build() for {model_name}: {e}. Skipping model.")
            log(traceback.format_exc())
            continue
            
        # 6. Final Configuration of Padding Callback (e.g., DRMM embedding_matrix from built model)
        if model_name == "DRMM":
            DRMMPadding_cls = getattr(mz.dataloader.callbacks, 'DRMMPadding', None)
            if DRMMPadding_cls and isinstance(padding_callback, DRMMPadding_cls):
                log(f"Finalizing DRMMPadding callback for {model_name} with embedding matrix...")
                drmm_embed_for_hist = None
                # Check if a pre-trained embedding matrix was passed to the model's parameters
                if model.params.get('embedding') is not None and isinstance(model.params['embedding'], np.ndarray):
                    drmm_embed_for_hist = model.params['embedding']
                    log("Using pre-trained embedding matrix (from model.params) for DRMM histogram.")
                # Else, if the model has an internal embedding layer (initialized randomly)
                elif hasattr(model, 'embedding') and isinstance(model.embedding, mz.embedding.Embedding) and hasattr(model.embedding, 'weight'):
                    try:
                        drmm_embed_for_hist = model.embedding.weight.data.cpu().numpy()
                        log("Using DRMM's internal (randomly initialized) embedding matrix for histogram.")
                    except Exception as e_embed_extract:
                        log(f"Warning: Could not extract weights from DRMM's internal embedding layer: {e_embed_extract}")
                else:
                    log("Warning: Could not determine embedding matrix for DRMM histogram. "
                        "Model.params['embedding'] is not a numpy array or no model.embedding.weight found.")

                if drmm_embed_for_hist is not None:
                    if hasattr(padding_callback, 'set_embedding_matrix') and callable(getattr(padding_callback, 'set_embedding_matrix')):
                        log(f"Setting embedding_matrix for DRMMPadding (shape: {drmm_embed_for_hist.shape})...")
                        padding_callback.set_embedding_matrix(drmm_embed_for_hist)
                    else:
                        log("Warning: DRMMPadding instance does not have a set_embedding_matrix method.")
                else:
                    log("Warning: Embedding for DRMM histogram is None. Skipping setting it on DRMMPadding.")
        
        # 7. Create DataLoaders (NOW padding_callback is fully configured)
        log(f"Creating DataLoaders for {model_name} with padding callback: {type(padding_callback).__name__}")
        train_dataset = mz.dataloader.Dataset(
            data_pack=train_processed,
            mode='pair',
            num_dup=1,
            num_neg=1,
            batch_size=args.batch_size,
            shuffle=True
        )
        train_loader = mz.dataloader.DataLoader(
            dataset=train_dataset,
            stage='train',
            callback=padding_callback # Use the fully configured callback
        )

        valid_loader = None
        if use_dev_set_for_validation and dev_processed:
            valid_dataset = mz.dataloader.Dataset(
                data_pack=dev_processed,
                mode='point',
                batch_size=args.batch_size,
                shuffle=False
            )
            valid_loader = mz.dataloader.DataLoader(
                dataset=valid_dataset,
                stage='dev',
                callback=padding_callback # Use the same configured callback
            )
            log("Validation DataLoader created.")
        else:
            log("Validation DataLoader not created (no valid dev set or not provided).")

        # 8. Setup Trainer
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        
        trainer_kwargs = {
            'model': model,
            'optimizer': optimizer,
            'trainloader': train_loader,
            'validloader': valid_loader,
            'epochs': args.epochs,
            'verbose': 1 # Log progress bar and metrics per epoch
        }

        if use_dev_set_for_validation and valid_loader:
            trainer_kwargs['patience'] = args.patience
            trainer_kwargs['key'] = actual_metric_key_for_trainer # Use the mapped key
            log(f"Early stopping enabled. Metric: {actual_metric_key_for_trainer}, Patience: {args.patience}")
        else:
            # If no valid dev set, patience is effectively disabled by Trainer if key is None or patience=epochs
            trainer_kwargs['patience'] = args.epochs # Run for full epochs
            trainer_kwargs['key'] = None # No metric to monitor for early stopping
            log("Early stopping disabled (no valid dev set or not provided).")

        trainer = mz.trainers.Trainer(**trainer_kwargs)

        # Run Training
        try:
            log(f"Starting training for {model_name}...")
            trainer.run()
            log(f"Training finished for {model_name}.")

            # Save Artifacts
            log("Saving model artifacts...")
            # 1. Model weights
            torch.save(model.state_dict(), os.path.join(current_model_output_dir, "model.pt"))
            # 2. Preprocessor
            with open(os.path.join(current_model_output_dir, "preprocessor.dill"), "wb") as f:
                dill.dump(preprocessor, f)
            
            # Prepare model.params for JSON serialization
            params_for_json = {}
            if hasattr(model, 'params') and model.params is not None:
                # ParamTable is iterable, yielding (key, param_object)
                for key, param_value_obj in model.params:
                    # param_value_obj has a 'value' attribute
                    params_for_json[key] = param_value_obj.value 

            # 3. Config
            config_to_save = {
                "model_name": model_name,
                "params_completed": params_for_json, # Use the converted dict
                "training_args": vars(args),
                "epochs_trained": trainer.epoch_idx + 1, # epoch_idx is 0-based
                "best_metric_value": trainer.best_metric_value if use_dev_set_for_validation else None,
                "custom_embedding_path": args.embed_path if embedding_matrix is not None else "random_initialization",
                "embedding_dimension": final_embed_dim,
                "vocabulary_size": vocab_size,
                "timestamp": datetime.now().isoformat()
            }
            with open(os.path.join(current_model_output_dir, "config.json"), "w", encoding="utf-8") as f:
                # Convert numpy types to native Python types for JSON serialization
                def convert_numpy_types(obj):
                    if isinstance(obj, np.integer): return int(obj)
                    if isinstance(obj, np.floating): return float(obj)
                    if isinstance(obj, np.ndarray): return obj.tolist()
                    if isinstance(obj, (torch.Tensor)): return obj.tolist() # For any tensors in params
                    if isinstance(obj, type): return obj.__name__ # For class types in params
                    return obj

                json.dump(config_to_save, f, indent=4, default=convert_numpy_types)
            
            # 4. Success Flag
            with open(success_flag_path, "w") as f:
                f.write(datetime.now().isoformat())
            
            log(f"{model_name} training and artifact saving completed successfully. Output: {current_model_output_dir}")

        except Exception as e:
            log(f"Error during training or saving for {model_name}: {e}")
            log(traceback.format_exc())
            log(f"Skipping to next model due to error in {model_name}.")
            continue # Continue to the next model

        log(f"===== Finished Model: {model_name} =====")
        time.sleep(5) # Small pause

    log("All specified models have been processed.")

if __name__ == "__main__":
    main()