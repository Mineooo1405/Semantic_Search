# Standard Library Imports
import os
import sys
import json
import argparse
import traceback
from datetime import datetime
from collections import defaultdict

# Third-party Imports
import torch
import numpy as np
import pandas as pd
import matchzoo as mz # For mz.metrics.Evaluator, mz.DataPack, etc.
import dill
from tqdm import tqdm

# Typing
from typing import Optional, Tuple, Any, Dict, List

# Correct import for BasePreprocessor in MatchZoo-py 1.1.1
from matchzoo.engine.base_preprocessor import BasePreprocessor

# --- torch_directml setup (COMMENTED OUT) ---
# _torch_directml_device_instance = None # Stores the DirectML device object if available
# _torch_directml_module = None          # Stores the torch_directml module itself if imported
# try:
#     import torch_directml
#     if torch_directml.is_available():
#         _torch_directml_module = torch_directml
#         _torch_directml_device_instance = _torch_directml_module.device()
#         # Get device name safely
#         try:
#             # Assuming device index 0 for the name, common for single DML GPU setups
#             # dml_device_name = _torch_directml_module.get_device_name(0) # This was causing an error
#             # print(f"DirectML device detected and available: {dml_device_name}")
#             print(f"DirectML device detected and available.") # Simplified message
#         except Exception as e:
#             # Fallback if getting name fails but device() succeeded
#             print(f"DirectML device instance created, but could not retrieve name (device index 0): {e}")
#             print("Proceeding with DirectML device instance.")
#     else:
#         # This case means torch_directml was imported but is_available() is false.
#         print("torch_directml module was imported, but no DirectML device is currently available (is_available() returned False).")
# except ImportError:
#     # This case means the torch_directml package itself is not installed.
#     print("torch_directml package not found. GPU acceleration via DirectML will not be available.")
# except Exception as e:
#     # Catch any other unexpected errors during the setup.
#     print(f"An unexpected error occurred during torch_directml setup: {e}")
#     traceback.print_exc()
# --- end torch_directml setup ---

# --- NEW HELPER FUNCTIONS FOR INTERACTIVE INPUT ---
def get_interactive_input_path(prompt: str, default: Optional[str] = None, must_exist: bool = True, check_is_dir: Optional[bool] = None) -> str:
    """
    Prompts the user for a path.
    Args:
        prompt: The message to display to the user.
        default: The default path if the user enters nothing.
        must_exist: If True, the path must exist.
        check_is_dir: If True, path must be a directory. If False, path must be a file. If None, no check.
    Returns:
        The path entered by the user or the default.
    """
    while True:
        prompt_with_default = f"{prompt} (default: {default})" if default else prompt
        user_input = input(f"{prompt_with_default}: ").strip()
        
        path_to_check = user_input
        if not user_input and default:
            path_to_check = default
        
        if not path_to_check: # If still empty (no input, no default where default is required)
            if default is None: # Only print "Input is required" if there was no default to fall back on
                print("Input is required.")
                continue
            # If there was a default, and user entered nothing, path_to_check is now the default.
            # If default itself was empty and required, this loop continues.

        if must_exist:
            if not os.path.exists(path_to_check):
                print(f"Error: Path not found: {path_to_check}")
                continue
            if check_is_dir is True and not os.path.isdir(path_to_check):
                print(f"Error: Path is not a directory: {path_to_check}")
                continue
            if check_is_dir is False and not os.path.isfile(path_to_check):
                print(f"Error: Path is not a file: {path_to_check}")
                continue
        return path_to_check

def get_interactive_input_str(prompt: str, default: Optional[str] = None) -> str:
    """Prompts the user for a string input."""
    prompt_with_default = f"{prompt} (default: {default})" if default else prompt
    user_input = input(f"{prompt_with_default}: ").strip()
    if not user_input and default is not None:
        return default
    return user_input

def get_interactive_input_int(prompt: str, default: Optional[int] = None, min_val: Optional[int] = None) -> int:
    """Prompts the user for an integer input."""
    while True:
        default_str = str(default) if default is not None else None
        prompt_with_default = f"{prompt} (default: {default_str})" if default_str else prompt
        user_input_str = input(f"{prompt_with_default}: ").strip()

        if not user_input_str and default is not None:
            return default
        
        try:
            value = int(user_input_str)
            if min_val is not None and value < min_val:
                print(f"Value must be at least {min_val}.")
                continue
            return value
        except ValueError:
            print("Invalid input. Please enter an integer.")

# --- END NEW HELPER FUNCTIONS ---

# Helper function to load GloVe embeddings (can be shared or simplified if not needed for eval directly)
def load_glove_embeddings_eval(path, term_index, embedding_dim):
    embeddings_index = {}
    try:
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                values = line.split()
                word = values[0]
                try:
                    coefs = np.asarray(values[1:], dtype='float32')
                    if len(coefs) == embedding_dim:
                        embeddings_index[word] = coefs
                except ValueError:
                    pass # Skip lines that cannot be parsed correctly
    except FileNotFoundError:
        print(f"Error: GloVe file not found at {path}")
        return None
    
    num_tokens = len(term_index) + 1
    embedding_matrix = np.zeros((num_tokens, embedding_dim))
    hits = 0
    misses = 0
    for word, i in term_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
            hits += 1
        else:
            misses += 1
    print(f"Built embedding matrix for eval: {hits} words found, {misses} words missing.")
    return embedding_matrix

def load_model_and_artifacts(model_dir_path):
    """Loads model, preprocessor, and config from a given directory."""
    config_path = os.path.join(model_dir_path, 'config.json')
    preprocessor_path = os.path.join(model_dir_path, 'preprocessor.dill')
    model_weights_path = os.path.join(model_dir_path, 'model.pt')

    # Check for each file and print a specific message if missing
    missing_files = []
    if not os.path.exists(config_path):
        missing_files.append('config.json')
    if not os.path.exists(preprocessor_path):
        missing_files.append('preprocessor.dill')
    if not os.path.exists(model_weights_path):
        missing_files.append('model.pt')

    if missing_files:
        print(f"Error: Missing the following artifact file(s) in {model_dir_path}: {', '.join(missing_files)}")
        return None, None, None

    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        # Load preprocessor directly using dill
        if not os.path.isfile(preprocessor_path):
            print(f"Error: Preprocessor file not found at {preprocessor_path}")
            # Attempt to see if it's in a subdirectory (old MatchZoo behavior)
            old_style_preprocessor_dir = os.path.join(model_dir_path, "preprocessor.dill")
            old_style_preprocessor_file = os.path.join(old_style_preprocessor_dir, "preprocessor.dill")
            if os.path.isfile(old_style_preprocessor_file):
                print(f"Found preprocessor in old style directory: {old_style_preprocessor_file}")
                with open(old_style_preprocessor_file, 'rb') as f_preprocessor:
                    preprocessor = dill.load(f_preprocessor)
            else:
                print(f"Also not found in {old_style_preprocessor_file}")
                return None, None, None
        else:
            with open(preprocessor_path, 'rb') as f_preprocessor:
                preprocessor = dill.load(f_preprocessor)
        
        model_class_name = config.get('model_class')
        model_hyperparams_from_config = config.get('model_hyperparameters_used', {})

        if not hasattr(mz.models, model_class_name):
            print(f"Error: Model class {model_class_name} not found in matchzoo.models.")
            return None, None, None
        
        model_instance = getattr(mz.models, model_class_name)()
        
        # Get hyperparams from config
        model_hyperparams_from_config = config.get('model_hyperparameters_used', {})

        # Explicitly set embedding dimensions from config for model structure
        # These are the dimensions the saved model.pt checkpoint expects.
        expected_emb_in_dim = model_hyperparams_from_config.get('embedding_input_dim')
        expected_emb_out_dim = model_hyperparams_from_config.get('embedding_output_dim')

        if expected_emb_in_dim is None or expected_emb_out_dim is None:
            print("Error: 'embedding_input_dim' or 'embedding_output_dim' not found in config's model_hyperparameters_used.")
            # Attempt to fall back to other config values if they exist (less ideal but better than nothing)
            if expected_emb_in_dim is None and 'embedding_input_dim_from_preprocessor_context' in config:
                expected_emb_in_dim = config.get('embedding_input_dim_from_preprocessor_context')
                print(f"Warning: Using 'embedding_input_dim_from_preprocessor_context' from config: {expected_emb_in_dim}")
            elif expected_emb_in_dim is None and 'vocab_size' in config: # Some configs might save it as vocab_size
                 expected_emb_in_dim = config.get('vocab_size')
                 print(f"Warning: Using 'vocab_size' from config as embedding_input_dim: {expected_emb_in_dim}")


            if expected_emb_out_dim is None and 'embedding_dim_used' in config: # This is usually just embedding_dim
                expected_emb_out_dim = config.get('embedding_dim_used')
                print(f"Warning: Using 'embedding_dim_used' from config as embedding_output_dim: {expected_emb_out_dim}")
            elif expected_emb_out_dim is None and 'embedding_output_dim_from_model_params' in config:
                expected_emb_out_dim = config.get('embedding_output_dim_from_model_params')
                print(f"Warning: Using 'embedding_output_dim_from_model_params' from config: {expected_emb_out_dim}")


            if expected_emb_in_dim is None or expected_emb_out_dim is None:
                print("Critical Error: Cannot determine embedding dimensions for the model from config. Aborting load.")
                return None, None, None
        else:
            print(f"DEBUG: Using embedding_input_dim: {expected_emb_in_dim}, embedding_output_dim: {expected_emb_out_dim} from model_hyperparameters_used in config.")

        model_instance.params['embedding_input_dim'] = int(expected_emb_in_dim)
        model_instance.params['embedding_output_dim'] = int(expected_emb_out_dim)
        
        ranking_task = mz.tasks.Ranking() # Instantiate without loss argument

        # Determine and instantiate the actual loss function
        actual_loss_instance = mz.losses.RankHingeLoss() # Default loss
        loss_params = {} # Initialize loss_params

        if 'loss_function_class' in config:
            loss_class_name = config.get('loss_function_class') # Use .get() for safety
            if loss_class_name and isinstance(loss_class_name, str): # Check if it's a non-empty string
                if hasattr(mz.losses, loss_class_name):
                    # Populate loss_params from config and model_hyperparams_from_config
                    if 'loss_hinge_margin (if applicable)' in config and config['loss_hinge_margin (if applicable)'] is not None:
                        loss_params['margin'] = float(config['loss_hinge_margin (if applicable)'])
                    
                    # Check for num_neg in model_hyperparams_from_config (already present in original logic)
                    if 'loss_num_neg' in model_hyperparams_from_config and model_hyperparams_from_config['loss_num_neg'] is not None:
                         loss_params['num_neg'] = int(model_hyperparams_from_config['loss_num_neg'])
                    # Potentially other loss parameters could be extracted here if needed

                    try:
                        actual_loss_instance = getattr(mz.losses, loss_class_name)(**loss_params)
                    except TypeError as e:
                        print(f"Warning: Could not instantiate {loss_class_name} with params {loss_params}: {e}. Using default RankHingeLoss.")
                        actual_loss_instance = mz.losses.RankHingeLoss() # Fallback to default if params cause error
                else:
                    print(f"Warning: Loss class '{loss_class_name}' from config not found in mz.losses. Using default RankHingeLoss.")
            else:
                print(f"Warning: 'loss_function_class' in config is None or not a string ('{loss_class_name}'). Using default RankHingeLoss.")
        else:
            print("Warning: 'loss_function_class' not found in config. Using default RankHingeLoss.")

        # Dynamically assign the chosen loss instance to the task object
        ranking_task.loss = actual_loss_instance

        model_instance.params['task'] = ranking_task

        # Handle pre-trained embedding matrix
        if 'embedding_source_file' in config and config.get('embedding_source_file') and \
           'embedding_dim_used' in config and config.get('embedding_dim_used') is not None:
            
            glove_path_trained = config['embedding_source_file']
            # The embedding_dim_for_glove_load should be the expected_emb_out_dim from the model's structure
            embedding_dim_for_glove_load = int(expected_emb_out_dim)
            
            term_index = None
            if 'vocab_unit' in preprocessor.context and hasattr(preprocessor.context['vocab_unit'], 'state') and 'term_index' in preprocessor.context['vocab_unit'].state:
                term_index = preprocessor.context['vocab_unit'].state['term_index']
            elif 'term_index' in preprocessor.context: # Fallback for older direct structure
                term_index = preprocessor.context['term_index']

            if term_index is None:
                print(f"Error: 'term_index' not found in preprocessor.context (checked 'vocab_unit.state.term_index' and direct 'term_index'). This is required for loading pre-trained embeddings.")
                print(f"Available keys in preprocessor.context: {list(preprocessor.context.keys())}")
                if 'vocab_unit' in preprocessor.context and hasattr(preprocessor.context['vocab_unit'], 'state'):
                    print(f"Available keys in preprocessor.context['vocab_unit'].state: {list(preprocessor.context['vocab_unit'].state.keys())}")
                return None, None, None # Critical error
            
            embedding_matrix_for_eval = load_glove_embeddings_eval(glove_path_trained, term_index, embedding_dim_for_glove_load)
            if embedding_matrix_for_eval is None:
                print(f"Error: Could not load GloVe embeddings specified in config: {glove_path_trained}")
                return None, None, None
            embedding_matrix_for_eval = np.nan_to_num(embedding_matrix_for_eval, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Crucial check: Only set model.params['embedding'] if the loaded matrix dimensions
            # match what the model structure (defined by expected_emb_in_dim) expects.
            # embedding_matrix_for_eval.shape[0] is len(term_index_from_preprocessor) + 1
            # expected_emb_in_dim is len(term_index_from_training_config) + 1
            if embedding_matrix_for_eval.shape[0] == int(expected_emb_in_dim) and \
               embedding_matrix_for_eval.shape[1] == int(expected_emb_out_dim):
                print(f"DEBUG: Shape of loaded GloVe matrix {embedding_matrix_for_eval.shape} matches expected embedding dims ({expected_emb_in_dim}, {expected_emb_out_dim}). Setting model.params['embedding'].")
                model_instance.params['embedding'] = embedding_matrix_for_eval
            else:
                print(f"Warning: Shape of loaded GloVe matrix ({embedding_matrix_for_eval.shape}) " +
                      f"does not match expected embedding dimensions ({expected_emb_in_dim}, {expected_emb_out_dim}) from config. ")
                print("Not setting model.params['embedding']. Model will use its own nn.Embedding layer, " +
                      "and its weights will be populated from the checkpoint.")
                # Ensure 'embedding' param is not set if it was there by default or from a previous step
                # No removal needed here; if not set, model's build process handles it.
        
        # No need for an elif to remove 'embedding'; if not set above, it's fine.
        # If 'embedding_source_file' was not in config, the above block is skipped,
        # and model.params['embedding'] would not have been set by this logic.

        # Update model with OTHER hyperparameters from config 
        # (excluding task, loss, embedding, and embedding dimensions which are already handled)
        specific_model_params_to_set = {}
        for k, v in model_hyperparams_from_config.items():
            if k not in ['task', 'loss', 'embedding', 'embedding_input_dim', 'embedding_output_dim'] and v is not None:
                # Attempt to restore original types if they were simple (int, float, list of simple types)
                # More complex types or structures might need explicit handling or were stringified
                if isinstance(v, str):
                    try: # Attempt to parse if it looks like a list or dict string
                        parsed_v = json.loads(v.replace("'", "\"")) # Handle single quotes if any
                        specific_model_params_to_set[k] = parsed_v
                    except json.JSONDecodeError:
                        specific_model_params_to_set[k] = v # Keep as string if not parsable as JSON
                else:
                     specific_model_params_to_set[k] = v
        
        model_instance.params.update(specific_model_params_to_set)
        
        model_instance.build()

        # Ensure 'preprocessor' Param exists before assignment
        if 'preprocessor' not in model_instance.params: # This uses ParamTable.__contains__
            print("DEBUG: Manually adding 'preprocessor' Param to model_instance.params")
            # Create the Param object with the preprocessor instance as its value,
            # and a description.
            preprocessor_param = mz.engine.param.Param(
                name='preprocessor',
                value=preprocessor  # The loaded preprocessor instance
                # is_hyperparam=False # This was causing the TypeError
            )
            # Set the is_hyperparam attribute on the instance
            preprocessor_param.is_hyperparam = False # Preprocessor is not a hyperparameter for tuning
            
            # Add the created Param object to the model_instance.params
            model_instance.params.add(preprocessor_param)
        else:
            # If 'preprocessor' Param object already exists, update its value
            print("DEBUG: 'preprocessor' Param found. Updating its value with the loaded preprocessor.")
            model_instance.params['preprocessor'] = preprocessor # Updates the .value of the existing Param
            
            # Also, ensure 'is_hyperparam' is False for the existing parameter
            # in case it was saved differently or defaulted to True.
            existing_param_obj = model_instance.params.get('preprocessor') # Gets the Param object
            if existing_param_obj is not None and hasattr(existing_param_obj, 'is_hyperparam'):
                if existing_param_obj.is_hyperparam: # Only print and set if it's currently True
                    # print("DEBUG: Ensuring is_hyperparam is False for existing 'preprocessor' Param.")
                    existing_param_obj.is_hyperparam = False

        model_instance.load_state_dict(torch.load(model_weights_path, map_location=torch.device('cpu')))
        model_instance.eval() 
        
        print(f"Successfully loaded model {config.get('model_name')} from {model_dir_path}")
        return model_instance, preprocessor, config

    except Exception as e:
        print(f"Error loading model from {model_dir_path}: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None

def load_evaluation_data(data_path, preprocessor, config):
    """Loads and preprocesses evaluation data."""
    print(f"Loading evaluation data from: {data_path}")
    try:
        # Assuming TSV: query_id, doc_id, text_left (query), text_right (doc), label (relevance)
        # Or: text_left, text_right, label (and generate ids)
        # For now, let's assume a common format: text_left, text_right, label
        # And we will generate q_ids and d_ids for DataPack structure.
        eval_df = pd.read_csv(data_path, sep='\t', header=None, names=['text_left', 'text_right', 'label'], dtype={'label': int})
        
        eval_df['id_left'] = [f'eval_q_{i}' for i in range(len(eval_df))]
        eval_df['id_right'] = [f'eval_d_{i}' for i in range(len(eval_df))]
        
        print(f"Loaded {len(eval_df)} pairs from {data_path}")
        
        # Prepare DataFrames for MatchZoo DataPack
        relation_df = eval_df[['id_left', 'id_right', 'label']]
        # Ensure 'text_left' and 'text_right' are the column names for text content
        # The index should be the respective ID column
        left_data = eval_df[['id_left', 'text_left']].set_index('id_left')
        right_data = eval_df[['id_right', 'text_right']].set_index('id_right')

        eval_pack_raw = mz.DataPack(
            relation=relation_df,
            left=left_data,
            right=right_data
        )
        
        eval_callbacks = []
        model_name = config.get('model_name')
        model_hyperparams_from_config = config.get('model_hyperparameters_used', {})

        if model_name == "DRMM":
            if 'embedding_source_file' in config and config['embedding_source_file']:
                glove_path_trained = config['embedding_source_file']
                embedding_dim_trained = int(config['embedding_dim_used'])
                term_index = preprocessor.context['term_index']
                embedding_matrix_for_hist = load_glove_embeddings_eval(glove_path_trained, term_index, embedding_dim_trained)
                if embedding_matrix_for_hist is None: return None
                embedding_matrix_for_hist = np.nan_to_num(embedding_matrix_for_hist, nan=0.0, posinf=0.0, neginf=0.0)
                
                # Get DRMM histogram params from config if saved, else use defaults
                hist_bin_size = model_hyperparams_from_config.get('mlp_num_bins', model_hyperparams_from_config.get('bin_size', 30))
                hist_mode = model_hyperparams_from_config.get('hist_mode', 'LCH')
                
                histogram_callback = mz.dataloader.callbacks.Histogram(
                    embedding_matrix=embedding_matrix_for_hist,
                    bin_size=int(hist_bin_size),
                    hist_mode=str(hist_mode)
                )
                eval_callbacks.append(histogram_callback)
            else:
                print("Warning: DRMM model, but no embedding source found in config for Histogram callback.")
        else:
            padding_callback = mz.dataloader.callbacks.BasicPadding(
                fixed_length_left=30,
                fixed_length_right=100
            )
            if hasattr(mz.models, config.get('model_class')):
                model_cls_for_padding = getattr(mz.models, config.get('model_class'))
                if hasattr(model_cls_for_padding, 'get_default_padding_callback'):
                    try:
                        padding_callback = model_cls_for_padding.get_default_padding_callback(
                            fixed_length_left=30,
                            fixed_length_right=100
                        )
                    except Exception as e_pad_cb:
                        print(f"Warning: Using BasicPadding for {model_name}. Error getting default: {e_pad_cb}")
            eval_callbacks.append(padding_callback)

        eval_pack_processed = preprocessor.transform(eval_pack_raw, verbose=0)
        
        eval_dataset = mz.dataloader.Dataset(
            data_pack=eval_pack_processed,
            mode='point', 
            batch_size=config.get('batch_size', 64),
            callbacks=eval_callbacks
        )
        return eval_dataset

    except Exception as e:
        print(f"Error loading or processing evaluation data: {e}")
        import traceback
        traceback.print_exc()
        return None

def load_test_data_tsv(file_path):
    """Loads test data from a TSV file (text_left, text_right, label)."""
    try:
        # Assuming no header, as saved by Train_MatchZoo_BEIR_Multiple.py
        df = pd.read_csv(file_path, sep='\\t', header=None, names=['text_left', 'text_right', 'label'], quoting=3, engine='python') # MODIFIED: Added engine='python'
        df['label'] = pd.to_numeric(df['label'], errors='coerce').fillna(0).astype(int)
        
        # Add id_left and id_right for compatibility with predict_and_score_for_ranking
        df['id_left'] = [f'test_q_{i}' for i in range(len(df))]
        df['id_right'] = [f'test_d_{i}' for i in range(len(df))] # Though not strictly needed by current error, good for consistency
        
        print(f"Loaded and augmented {len(df)} pairs from {file_path} with id_left and id_right.")
        return df
    except Exception as e:
        print(f"Error loading TSV file {file_path}: {e}", file=sys.stderr)
        raise

def predict_and_score_for_ranking(
    model: torch.nn.Module,
    preprocessor: BasePreprocessor, # Corrected type hint
    original_test_df: pd.DataFrame,
    device: torch.device,
    batch_size: int = 128,
    desc: str = "Predicting"
) -> pd.DataFrame:
    """
    Predicts scores for ranking given a model, preprocessor, and test data (DataFrame).

    Args:
        model: The trained MatchZoo model.
        preprocessor: The preprocessor used during training.
        original_test_df: DataFrame with at least 'text_left', 'text_right', and 'id_left'.
                          It can also contain 'label' for convenience, which will be carried over.
        device: The torch device to use for prediction.
        batch_size: Batch size for prediction.
        desc: Description for the tqdm progress bar.

    Returns:
        A pandas DataFrame with original columns plus a 'score' column.
    """
    print(f"Starting prediction with batch size: {batch_size} on device: {device}")
    
    # Ensure necessary columns are present in the input DataFrame
    required_cols = ['text_left', 'text_right', 'id_left'] # 'label' is optional but good to have
    for col in required_cols:
        if col not in original_test_df.columns:
            print(f"ERROR: Missing required column '{col}' in original_test_df for prediction.", file=sys.stderr)
            return pd.DataFrame() # Return empty DataFrame on error

    # Create a MatchZoo DataPack for prediction (no task needed for just getting scores)
    # We only need text_left and text_right for the model to predict.
    # Other columns like id_left, label will be merged back.
    # predict_df_for_mz = original_test_df[['text_left', 'text_right']].copy()
    # predict_pack_raw = mz.DataPack(data=predict_df_for_mz)

    # Prepare DataFrames for MatchZoo DataPack for prediction
    # The original_test_df should have id_left, text_left, id_right, text_right, and optionally label
    relation_df_pred = original_test_df[['id_left', 'id_right']].copy()
    # Add a dummy label if it's not present, though not strictly needed for prediction's DataPack structure
    if 'label' not in original_test_df.columns:
        relation_df_pred['label'] = 0 
    else:
        relation_df_pred['label'] = original_test_df['label']

    left_df_pred = original_test_df[['id_left', 'text_left']].set_index('id_left')
    right_df_pred = original_test_df[['id_right', 'text_right']].set_index('id_right')

    predict_pack_raw = mz.DataPack(
        relation=relation_df_pred,
        left=left_df_pred,
        right=right_df_pred
    )

    print("Transforming data using preprocessor...")
    try:
        # Important: Use the *fitted* preprocessor to transform the new data
        predict_pack_processed = preprocessor.transform(predict_pack_raw, verbose=0)
    except Exception as e:
        print(f"Error during preprocessor.transform: {e}", file=sys.stderr)
        traceback.print_exc()
        return pd.DataFrame()

    print(f"Processed data for prediction: {len(predict_pack_processed)} items.")
    if len(predict_pack_processed) == 0:
        print("Warning: Preprocessing resulted in zero items for prediction.", file=sys.stderr)
        return pd.DataFrame() # Or original_test_df with empty scores

    # Create a DataLoader for prediction
    # The padding callback should be obtained from the model's preprocessor context or default for the model type
    # For simplicity, trying to get default padding callback from the model class itself.
    # This might need adjustment if the model instance or preprocessor stores specific padding info.
    padding_callback = None
    try:
        # Try to get the padding callback associated with the model class
        # This assumes the model class has a static/class method get_default_padding_callback
        if hasattr(type(model), 'get_default_padding_callback') and callable(getattr(type(model), 'get_default_padding_callback')):
            padding_callback = type(model).get_default_padding_callback()
            # If fixed lengths were stored in config and are relevant to the callback, apply them
            # This is a common pattern but might need to be more robust based on how padding was handled in training
            if preprocessor and preprocessor.context:
                fixed_length_left = preprocessor.context.get('fixed_length_left')
                fixed_length_right = preprocessor.context.get('fixed_length_right')
                if fixed_length_left is not None and hasattr(padding_callback, '_fixed_length_left'):
                    padding_callback._fixed_length_left = fixed_length_left
                if fixed_length_right is not None and hasattr(padding_callback, '_fixed_length_right'):
                    padding_callback._fixed_length_right = fixed_length_right
        else:
            print("Warning: Could not get default padding callback from model class. Using BasicPadding.", file=sys.stderr)
            # Fallback to BasicPadding if specific callback is not found or applicable
            # This requires knowing the fixed lengths, which should ideally come from the preprocessor
            fixed_length_left = preprocessor.context.get('fixed_length_left', 30) # Default fallback
            fixed_length_right = preprocessor.context.get('fixed_length_right', 100) # Default fallback
            padding_callback = mz.dataloader.callbacks.BasicPadding(fixed_length_left=fixed_length_left, fixed_length_right=fixed_length_right)

    except Exception as e_cb:
        print(f"Error getting/configuring padding callback: {e_cb}. Falling back to BasicPadding.", file=sys.stderr)
        fixed_length_left = preprocessor.context.get('fixed_length_left', 30)
        fixed_length_right = preprocessor.context.get('fixed_length_right', 100)
        padding_callback = mz.dataloader.callbacks.BasicPadding(fixed_length_left=fixed_length_left, fixed_length_right=fixed_length_right)

    if padding_callback is None:
        print("ERROR: Padding callback could not be initialized. Cannot proceed with DataLoader.", file=sys.stderr)
        return pd.DataFrame()

    print(f"Using padding callback: {type(padding_callback).__name__}")

    predict_dataset = mz.dataloader.Dataset(
        data_pack=predict_pack_processed,
        mode='point', # For prediction, usually point mode (each pair independently)
        batch_size=batch_size,
        shuffle=False
    )
    predict_loader = mz.dataloader.DataLoader(
        dataset=predict_dataset,
        stage='test', # Inform callback it's for testing/prediction
        callback=padding_callback
    )

    model.eval() # Ensure model is in evaluation mode
    all_scores = []

    print(f"Iterating through DataLoader for prediction (total batches: {len(predict_loader)})...")
    with torch.no_grad(): # Disable gradient calculations for prediction
        for batch_num, batch_x_y in enumerate(tqdm(predict_loader, desc=desc)):
            batch_x = batch_x_y[0] # Input data
            # Move data to the specified device
            # Batch_x is expected to be a dict of tensors
            try:
                batch_x_on_device = {k: v.to(device) for k, v in batch_x.items()} 
            except AttributeError as e:
                print(f"Error moving batch to device. Batch X keys: {batch_x.keys()}, Error: {e}", file=sys.stderr)
                # If a value in batch_x is not a tensor, this will fail.
                # This can happen if data is not correctly processed by dataloader/callback.
                print(f"Problematic batch_x content (first item of each key if list/tensor):", file=sys.stderr)
                for k,v in batch_x.items():
                    if isinstance(v, (list, torch.Tensor)) and len(v) > 0:
                        print(f"  {k}: type {type(v[0])}, value {v[0]}", file=sys.stderr)
                    else:
                        print(f"  {k}: type {type(v)}, value {v}", file=sys.stderr)
                # Fallback: try to predict on CPU if device transfer fails for unknown reasons
                # This is a desperate measure and indicates a deeper problem.
                if device != torch.device("cpu"):
                    print("Attempting prediction on CPU as a fallback...", file=sys.stderr)
                    try:
                        batch_x_on_device = {k: v.to(torch.device("cpu")) for k, v in batch_x.items()}
                    except Exception as cpu_e:
                        print(f"Fallback to CPU also failed: {cpu_e}", file=sys.stderr)
                        # Skip this batch or return error
                        all_scores.extend([0.0] * len(batch_x.get(list(batch_x.keys())[0], []))) # Add dummy scores
                        continue 
                else:
                    # Already on CPU and failed, so skip
                    all_scores.extend([0.0] * len(batch_x.get(list(batch_x.keys())[0], []))) # Add dummy scores
                    continue
            except Exception as e_device:
                print(f"Unexpected error moving batch to device: {e_device}", file=sys.stderr)
                all_scores.extend([0.0] * len(batch_x.get(list(batch_x.keys())[0], []))) # Add dummy scores
                continue

            try:
                # Model forward pass
                # The output shape and type depend on the model and task.
                # For ranking, it's typically a score per pair.
                predictions = model(batch_x_on_device)

                # Ensure predictions are on CPU and converted to a list of floats
                if isinstance(predictions, torch.Tensor):
                    predictions = predictions.detach().cpu().numpy()
                
                # Handle different prediction shapes
                # If predictions are (batch_size, 1), squeeze them
                if predictions.ndim == 2 and predictions.shape[1] == 1:
                    scores = predictions.squeeze(-1).tolist()
                # If predictions are (batch_size, num_classes) for classification, take the score of the positive class (e.g., class 1)
                elif predictions.ndim == 2 and predictions.shape[1] > 1:
                    # This case needs to be handled based on how the model is trained.
                    # Assuming for ranking, if output is multi-dimensional, it might be [score_for_neg, score_for_pos]
                    # or just a single score. If it's a single score, the previous case handles it.
                    # For now, let's assume if it's (batch, N), we take the last score as relevance (common in some setups)
                    # OR, if the task is Ranking and output is (batch, 1), it's already handled.
                    # This part might need refinement based on specific model output structures.
                    print(f"Warning: Model output has shape {predictions.shape}. Assuming last column is relevance score.", file=sys.stderr)
                    scores = predictions[:, -1].tolist() 
                elif predictions.ndim == 1:
                    scores = predictions.tolist()
                else:
                    print(f"Error: Unexpected prediction shape: {predictions.shape}", file=sys.stderr)
                    # Add dummy scores for this batch
                    num_in_batch = len(batch_x.get(list(batch_x.keys())[0], [])) # Get batch size from input
                    scores = [0.0] * num_in_batch
                
                all_scores.extend(scores)

            except Exception as e_pred:
                print(f"Error during model prediction or score processing for batch {batch_num}: {e_pred}", file=sys.stderr)
                traceback.print_exc()
                # Add dummy scores for this batch if prediction fails
                num_in_batch = len(batch_x.get(list(batch_x.keys())[0], [])) # Get batch size from input
                all_scores.extend([0.0] * num_in_batch)
                continue # Continue to the next batch
    
    print(f"Prediction finished. Total scores predicted: {len(all_scores)}")

    # Check if the number of scores matches the number of input items
    if len(all_scores) != len(original_test_df):
        print(f"Warning: Number of scores ({len(all_scores)}) does not match number of input items ({len(original_test_df)}).", file=sys.stderr)
        # Pad with zeros or truncate if mismatch, though this indicates a problem.
        # For now, we'll proceed, but this should be investigated.
        # If more scores than items, truncate. If fewer, pad with 0.0 (or NaN).
        if len(all_scores) > len(original_test_df):
            all_scores = all_scores[:len(original_test_df)]
        else:
            all_scores.extend([0.0] * (len(original_test_df) - len(all_scores)))

    # Add scores to the original DataFrame
    # Ensure index alignment if original_test_df had a non-default index.
    # For simplicity, assuming original_test_df has a default 0-based integer index if it was just loaded.
    output_df = original_test_df.copy()
    output_df['score'] = all_scores

    return output_df

def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained MatchZoo model.")
    parser.add_argument("--model_dir", type=str, required=False, # Changed to False for interactive mode
                        help="Directory containing the trained model artifacts (model.pt, preprocessor.dill, config.json).")
    parser.add_argument("--eval_data", type=str, required=False, # Changed to False for interactive mode
                        help="Path to the evaluation data file (TSV format: text_left, text_right, label). Label should be integer (0 or 1 for relevance).")
    parser.add_argument("--output_dir", type=str, default="./eval_results", 
                        help="Directory to save evaluation results.")
    parser.add_argument("--metrics", nargs='+', default=["ndcg@3", "ndcg@5", "map", "mrr"], 
                        help="List of metrics to calculate (e.g., ndcg@k map mrr). Default: ndcg@3 ndcg@5 map mrr")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size for prediction.")

    # Check if any arguments were passed (other than the script name itself)
    if len(sys.argv) > 1:
        # If arguments are passed, parse them as usual
        # Temporarily set required back to True for standard parsing if args are present
        parser.get_action('--model_dir').required = True
        parser.get_action('--eval_data').required = True
        args = parser.parse_args()
    else:
        # No command-line arguments, enter interactive mode
        print("No command-line arguments provided. Entering interactive mode...\n")
        
        model_dir_interactive = get_interactive_input_path(
            prompt="Enter path to the model directory (e.g., D:/path/to/trained_model_dir)",
            must_exist=True,
            check_is_dir=True
        )
        eval_data_interactive = get_interactive_input_path(
            prompt="Enter path to the evaluation data TSV file (e.g., D:/path/to/dev_data.tsv)",
            must_exist=True,
            check_is_dir=False # It's a file
        )
        output_dir_interactive = get_interactive_input_path(
            prompt="Enter directory to save evaluation results",
            default=parser.get_default("output_dir"),
            must_exist=False # Directory will be created
        )
        
        default_metrics_list = parser.get_default("metrics")
        metrics_str_interactive = get_interactive_input_str(
            prompt=f"Enter metrics (space-separated)",
            default=' '.join(default_metrics_list)
        )
        metrics_interactive = metrics_str_interactive.split()
        
        batch_size_interactive = get_interactive_input_int(
            prompt="Enter batch size for prediction",
            default=parser.get_default("batch_size"),
            min_val=1
        )

        args = argparse.Namespace(
            model_dir=model_dir_interactive,
            eval_data=eval_data_interactive,
            output_dir=output_dir_interactive,
            metrics=metrics_interactive,
            batch_size=batch_size_interactive
        )

    print(f"Starting evaluation at {datetime.now().isoformat()}")
    print(f"Model directory: {args.model_dir}")
    print(f"Evaluation data: {args.eval_data}")
    print(f"Output directory: {args.output_dir}")
    print(f"Metrics: {args.metrics}")

    os.makedirs(args.output_dir, exist_ok=True)

    model, preprocessor, config = load_model_and_artifacts(args.model_dir)
    if not model or not preprocessor or not config:
        sys.exit(1)

    # Determine device for evaluation
    # if torch_directml and torch_directml.is_available():
    #     eval_device = torch_directml.device()
    #     print(f"INFO: Using DirectML device for evaluation: {eval_device}")
    if torch.cuda.is_available():
        eval_device = torch.device("cuda")
        print(f"INFO: Using CUDA device for evaluation: {eval_device}")
    else:
        eval_device = torch.device("cpu")
        print(f"INFO: Using CPU device for evaluation: {eval_device}")
    
    model.to(eval_device)
    model.eval() # Set model to evaluation mode

    # The following lines related to load_evaluation_data, eval_dataset, and eval_loader are removed
    # as predict_and_score_for_ranking handles its own data loading and prediction loop.
    # eval_dataset = load_evaluation_data(args.eval_data, preprocessor, config)
    # if not eval_dataset:
    #     sys.exit(1)
    # eval_loader = mz.dataloader.DataLoader(dataset=eval_dataset)

    # The old Evaluator block is removed.

    # --- Main Evaluation Flow using predict_and_score_for_ranking and mz.metrics.Evaluator ---
    
    print(f"Loading test data for detailed prediction from: {args.eval_data}")
    try:
        test_df = load_test_data_tsv(args.eval_data)
    except Exception as e:
        print(f"Error loading test data from {args.eval_data}: {e}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)

    if test_df.empty:
        print(f"Test data is empty. Check file: {args.eval_data}", file=sys.stderr)
        sys.exit(1)

    print("Predicting scores for the test data...")
    try:
        df_with_scores = predict_and_score_for_ranking(
            model=model,
            preprocessor=preprocessor,
            original_test_df=test_df.copy(), # Use a copy to avoid modifying original df
            device=eval_device,
            batch_size=args.batch_size,
            desc="Predicting for evaluation"
        )
    except Exception as e:
        print(f"Error during prediction: {e}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)

    if df_with_scores.empty or 'score' not in df_with_scores.columns:
        print("Error: Prediction did not return scores or resulted in an empty dataframe.", file=sys.stderr)
        sys.exit(1)

    print("Grouping predictions by query for metric calculation...")
    all_y_true_grouped = []
    all_y_pred_grouped = []
    # Ensure 'id_left' is the correct query identifier from load_test_data_tsv
    # and is present in df_with_scores (it should be, as original_test_df is used)
    
    # Check if 'id_left' column exists
    if 'id_left' not in df_with_scores.columns:
        print(f"Error: 'id_left' column not found in the dataframe with scores. Available columns: {df_with_scores.columns.tolist()}", file=sys.stderr)
        print("This column is expected to be the query identifier from the input TSV.", file=sys.stderr)
        sys.exit(1)
    if 'label' not in df_with_scores.columns:
        print(f"Error: 'label' column not found in the dataframe with scores. Available columns: {df_with_scores.columns.tolist()}", file=sys.stderr)
        sys.exit(1)


    grouped_for_eval = defaultdict(lambda: {'y_true': [], 'y_pred': []})
    for _, row in df_with_scores.iterrows():
        query_id = row['id_left'] 
        grouped_for_eval[query_id]['y_true'].append(row['label'])
        grouped_for_eval[query_id]['y_pred'].append(row['score'])

    # Iterate based on the order of keys in grouped_for_eval (insertion order for Python 3.7+)
    # or sort keys if a specific order is required for reproducibility across Python versions if keys are not numbers.
    # query_keys = sorted(list(grouped_for_eval.keys())) # Example if sorting is needed
    query_keys = list(grouped_for_eval.keys())


    for query_id in query_keys:
        all_y_true_grouped.append(grouped_for_eval[query_id]['y_true'])
        all_y_pred_grouped.append(grouped_for_eval[query_id]['y_pred'])

    if not all_y_true_grouped:
        print("No data to evaluate after grouping. Check test file, predictions, and 'id_left' for query grouping.", file=sys.stderr)
        sys.exit(1)

    print("Calculating evaluation metrics...")
    ranking_metrics_to_calc = [] # Stores tuples of (metric_str, metric_instance)
    for m_str in args.metrics: 
        parts = m_str.split('@')
        metric_name = parts[0].lower()
        k = None
        if len(parts) > 1:
            try:
                k = int(parts[1])
            except ValueError:
                print(f"Warning: Could not parse k from {m_str}, ignoring k.")
        
        metric_class = None
        if metric_name == "map":
            metric_class = mz.metrics.MeanAveragePrecision()
        elif metric_name == "mrr":
            metric_class = mz.metrics.MeanReciprocalRank()
        elif metric_name == "ndcg":
            if k:
                metric_class = mz.metrics.NormalizedDiscountedCumulativeGain(k=k)
            else:
                print(f"Warning: NDCG requires a k value (e.g., ndcg@3). Skipping {m_str}.")
        # Add other metrics as needed
        # Example:
        # elif metric_name == "precision":
        #     if k:
        #         metric_class = mz.metrics.Precision(k=k)
        #     else:
        #         print(f"Warning: Precision requires a k value (e.g., precision@3). Skipping {m_str}.")
        else:
            print(f"Warning: Unknown metric '{metric_name}' from '{m_str}'. Skipping.")

        if metric_class:
            ranking_metrics_to_calc.append((m_str, metric_class)) # Store metric_str along with instance

    evaluation_results = {}
    if not ranking_metrics_to_calc:
        print("No metrics specified for calculation, or all specified metrics were invalid.")
    else:
        print(f"Calculating metrics manually: {[item[0] for item in ranking_metrics_to_calc]}") # Use item[0] for metric_str
        for metric_str, metric_instance in ranking_metrics_to_calc: # Unpack tuple
            metric_sum = 0.0
            num_queries_for_metric = 0
            
            for i in range(len(all_y_true_grouped)):
                y_true = np.array(all_y_true_grouped[i]) # Ensure numpy array
                y_pred = np.array(all_y_pred_grouped[i]) # Ensure numpy array
                # query_id_for_log = query_keys[i] # query_keys should be available from earlier grouping

                if y_true.size == 0 or y_pred.size == 0:
                    # print(f"Warning: Skipping query for metric {metric_str} due to empty true labels or predictions.")
                    continue
                
                y_true = y_true.astype(int)
                y_pred = y_pred.astype(float)

                if y_true.ndim == 0: y_true = np.expand_dims(y_true, axis=0)
                if y_pred.ndim == 0: y_pred = np.expand_dims(y_pred, axis=0)
                
                try:
                    metric_val = metric_instance(y_true, y_pred)
                    if isinstance(metric_val, torch.Tensor): # Ensure metric_val is a float
                        metric_val = metric_val.item()
                    metric_sum += metric_val
                    num_queries_for_metric += 1
                except Exception as e:
                    # print(f"Error calculating metric {metric_str} for a query: {e}")
                    # print(f"  y_true ({y_true.dtype}, shape {y_true.shape}): {y_true}")
                    # print(f"  y_pred ({y_pred.dtype}, shape {y_pred.shape}): {y_pred}")
                    pass # Continue to next query or metric

            if num_queries_for_metric > 0:
                evaluation_results[metric_str] = metric_sum / num_queries_for_metric # Use metric_str as key
            else:
                evaluation_results[metric_str] = 0.0 # Use metric_str as key
                # print(f"Warning: Metric {metric_str} could not be calculated for any query.")

    print("\\nEvaluation results (manual calculation):")
    # Ensure serializable_results uses evaluation_results
    serializable_results = {str(k): float(v) if isinstance(v, (np.floating, float, np.integer, int, torch.Tensor)) else str(v) 
                            for k, v in evaluation_results.items()}

    for metric_name, score in serializable_results.items():
        print(f"  {metric_name}: {score:.4f}")

    # --- Save results (optional) ---
    model_name_for_file = config.get('model_name', 'unknown_model').replace(" ", "_") if config else "unknown_model"
    eval_data_basename = os.path.splitext(os.path.basename(args.eval_data))[0]
    results_filename = f"eval_results_{model_name_for_file}_on_{eval_data_basename}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    results_filepath = os.path.join(args.output_dir, results_filename)

    try:
        with open(results_filepath, 'w') as f:
            json.dump(serializable_results, f, indent=4)
        print(f"\\nEvaluation results saved to: {results_filepath}")
    except Exception as e:
        print(f"Error saving results to JSON: {e}", file=sys.stderr)
        traceback.print_exc()

    print("\\nEvaluation script finished.")

if __name__ == "__main__":
    # Ensure all necessary top-level imports are here
    # import os
    # import json
    # import numpy as np
    # import pandas as pd
    # import torch
    # import matchzoo as mz
    # import argparse
    # import sys
    # from datetime import datetime
    # from tqdm import tqdm
    # from collections import defaultdict
    # import traceback
    
    try:
        main()
    except Exception as e:
        print(f"An error occurred in main: {e}", file=sys.stderr)
        traceback.print_exc()
