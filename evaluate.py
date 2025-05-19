import os
import json
import numpy as np
import pandas as pd
import torch
import matchzoo as mz
import argparse
import sys
from datetime import datetime

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

    if not all(os.path.exists(p) for p in [config_path, preprocessor_path, model_weights_path]):
        print(f"Error: Missing one or more artifact files in {model_dir_path}")
        return None, None, None

    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        preprocessor = mz.load_preprocessor(preprocessor_path)
        
        model_class_name = config.get('model_class')
        model_hyperparams_from_config = config.get('model_hyperparameters_used', {})

        if not hasattr(mz.models, model_class_name):
            print(f"Error: Model class {model_class_name} not found in matchzoo.models.")
            return None, None, None
        
        model_instance = getattr(mz.models, model_class_name)()
        
        ranking_task = mz.tasks.Ranking(loss=mz.losses.RankHingeLoss()) # Default
        if 'loss_function_class' in config:
            loss_class_name = config['loss_function_class']
            if hasattr(mz.losses, loss_class_name):
                loss_params = {}
                if 'loss_hinge_margin (if applicable)' in config and config['loss_hinge_margin (if applicable)'] is not None:
                    loss_params['margin'] = float(config['loss_hinge_margin (if applicable)'])
                if 'loss_num_neg' in model_hyperparams_from_config and model_hyperparams_from_config['loss_num_neg'] is not None:
                     loss_params['num_neg'] = int(model_hyperparams_from_config['loss_num_neg'])
                ranking_task = mz.tasks.Ranking(loss=getattr(mz.losses, loss_class_name)(**loss_params))

        model_instance.params['task'] = ranking_task

        if 'embedding_source_file' in config and config['embedding_source_file'] and \
           'embedding_dim_used' in config and config['embedding_dim_used']:
            glove_path_trained = config['embedding_source_file']
            embedding_dim_trained = int(config['embedding_dim_used'])
            term_index = preprocessor.context['term_index']
            embedding_matrix_for_eval = load_glove_embeddings_eval(glove_path_trained, term_index, embedding_dim_trained)
            if embedding_matrix_for_eval is None:
                print(f"Error: Could not load GloVe embeddings specified in config: {glove_path_trained}")
                return None, None, None
            embedding_matrix_for_eval = np.nan_to_num(embedding_matrix_for_eval, nan=0.0, posinf=0.0, neginf=0.0)
            model_instance.params['embedding'] = embedding_matrix_for_eval
        
        # Update model with other hyperparameters from config
        # This needs to be careful about types and what the model expects
        specific_model_params = {}
        for k, v in model_hyperparams_from_config.items():
            if k not in ['task', 'loss', 'embedding', 'embedding_input_dim', 'embedding_output_dim'] and v is not None:
                # Attempt to restore original types if they were simple (int, float, list of simple types)
                # More complex types or structures might need explicit handling or were stringified
                if isinstance(v, str):
                    try: # Attempt to parse if it looks like a list or dict string
                        parsed_v = json.loads(v.replace("'", "\"")) # Handle single quotes if any
                        specific_model_params[k] = parsed_v
                    except json.JSONDecodeError:
                        specific_model_params[k] = v # Keep as string if not parsable as JSON
                else:
                     specific_model_params[k] = v
        
        model_instance.params.update(specific_model_params)
        
        model_instance.build()
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
        
        eval_pack_raw = mz.DataPack(
            data=eval_df[['id_left', 'text_left', 'id_right', 'text_right']],
            relation=eval_df[['id_left', 'id_right', 'label']]
        )
        
        fixed_length_left = config.get('actual_fixed_length_left_from_preprocessor_context', 30)
        fixed_length_right = config.get('actual_fixed_length_right_from_preprocessor_context', 100)
        
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
                fixed_length_left=fixed_length_left,
                fixed_length_right=fixed_length_right
            )
            if hasattr(mz.models, config.get('model_class')):
                model_cls_for_padding = getattr(mz.models, config.get('model_class'))
                if hasattr(model_cls_for_padding, 'get_default_padding_callback'):
                    try:
                        padding_callback = model_cls_for_padding.get_default_padding_callback(
                            fixed_length_left=fixed_length_left,
                            fixed_length_right=fixed_length_right
                        )
                    except Exception as e_pad_cb:
                        print(f"Warning: Using BasicPadding for {model_name}. Error getting default: {e_pad_cb}")
            eval_callbacks.append(padding_callback)

        eval_pack_processed = preprocessor.transform(eval_pack_raw, 
                                                     fixed_length_left=fixed_length_left, 
                                                     fixed_length_right=fixed_length_right,
                                                     verbose=0)
        
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

def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained MatchZoo model.")
    parser.add_argument("--model_dir", type=str, required=True, 
                        help="Directory containing the trained model artifacts (model.pt, preprocessor.dill, config.json).")
    parser.add_argument("--eval_data", type=str, required=True, 
                        help="Path to the evaluation data file (TSV format: text_left, text_right, label). Label should be integer (0 or 1 for relevance).")
    parser.add_argument("--output_dir", type=str, default="./eval_results", 
                        help="Directory to save evaluation results.")
    parser.add_argument("--metrics", nargs='+', default=["ndcg@3", "ndcg@5", "map", "mrr"], 
                        help="List of metrics to calculate (e.g., ndcg@k map mrr). Default: ndcg@3 ndcg@5 map mrr")

    args = parser.parse_args()

    print(f"Starting evaluation at {datetime.now().isoformat()}")
    print(f"Model directory: {args.model_dir}")
    print(f"Evaluation data: {args.eval_data}")
    print(f"Output directory: {args.output_dir}")
    print(f"Metrics: {args.metrics}")

    os.makedirs(args.output_dir, exist_ok=True)

    model, preprocessor, config = load_model_and_artifacts(args.model_dir)
    if not model or not preprocessor or not config:
        sys.exit(1)

    eval_dataset = load_evaluation_data(args.eval_data, preprocessor, config)
    if not eval_dataset:
        sys.exit(1)
    
    # Ensure batch_size for loader is reasonable, can take from config or default
    eval_batch_size = config.get('batch_size', 64)
    eval_loader = mz.dataloader.DataLoader(dataset=eval_dataset, batch_size=eval_batch_size, shuffle=False)

    print("Starting prediction and evaluation...")
    
    # Parse metrics
    parsed_metrics = []
    for m_str in args.metrics:
        try:
            parsed_metrics.append(mz.metrics.parse(m_str))
        except Exception as e_metric_parse:
            print(f"Warning: Could not parse metric '{m_str}'. Skipping. Error: {e_metric_parse}")
    
    if not parsed_metrics:
        print("Error: No valid metrics to evaluate. Exiting.")
        sys.exit(1)

    evaluator = mz.trainers.Evaluator(model=model, 
                                      eval_loader=eval_loader, 
                                      metrics=parsed_metrics,
                                      device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
    try:
        results = evaluator.evaluate()
        print("Evaluation completed.")
        print("Results:")
        # Ensure results are serializable for JSON
        serializable_results = {str(k): float(v) for k, v in results.items()}

        for metric, value in serializable_results.items():
            print(f"  {metric}: {value:.4f}")

        model_name_for_file = config.get('model_name', 'unknown_model').replace(" ", "_")
        eval_data_basename = os.path.splitext(os.path.basename(args.eval_data))[0]
        results_filename = f"eval_results_{model_name_for_file}_on_{eval_data_basename}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        results_save_path = os.path.join(args.output_dir, results_filename)
        
        # Include config in results file for context
        output_data = {
            "evaluation_args": vars(args),
            "model_config_used_for_eval": config,
            "evaluation_results": serializable_results,
            "evaluation_timestamp": datetime.now().isoformat()
        }
        
        with open(results_save_path, 'w', encoding='utf-8') as f_res:
            json.dump(output_data, f_res, indent=4, ensure_ascii=False)
        print(f"Evaluation results and config saved to {results_save_path}")

    except Exception as e_eval:
        print(f"Error during evaluation: {e_eval}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print(f"Evaluation script finished at {datetime.now().isoformat()}")

if __name__ == "__main__":
    main()
