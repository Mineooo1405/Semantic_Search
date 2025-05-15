import matchzoo as mz
import pandas as pd
import os
import dill
import sys
import numpy as np
# from matchzoo import DataPack # Sẽ sử dụng mz.DataPack trực tiếp
from matchzoo.engine.param_table import ParamTable
from tqdm import tqdm # Added for progress bar

# --- Helper function to get and validate paths ---
def get_user_path(prompt_message: str, default_path: str, is_directory: bool = False, is_file: bool = False) -> str:
    """
    Prompts the user for a path and validates its existence.
    """
    while True:
        user_input = input(f"{prompt_message}\\n(default: {default_path}): ").strip()
        path_to_check = user_input if user_input else default_path

        if not os.path.exists(path_to_check):
            print(f"Error: Path not found: '{path_to_check}'. Please try again.")
            continue
        
        if is_directory and not os.path.isdir(path_to_check):
            print(f"Error: Path is not a directory: '{path_to_check}'. Please try again.")
            continue
        
        if is_file and not os.path.isfile(path_to_check):
            print(f"Error: Path is not a file: '{path_to_check}'. Please try again.")
            continue
            
        return path_to_check

# --- Helper functions for manual metric calculation ---

def average_precision(y_true, y_pred_scores):
    """
    Calculates Average Precision (AP) for a single query.
    """
    desc_score_indices = np.argsort(y_pred_scores)[::-1]
    y_true_sorted = np.asarray(y_true)[desc_score_indices]
    
    hits = 0
    sum_precisions = 0
    for i, relevant in enumerate(y_true_sorted):
        if relevant == 1:
            hits += 1
            precision_at_k = hits / (i + 1)
            sum_precisions += precision_at_k
            
    num_relevant_docs = np.sum(y_true)
    if num_relevant_docs == 0:
        return 0.0
    
    return sum_precisions / num_relevant_docs

def reciprocal_rank(y_true, y_pred_scores):
    """
    Calculates Reciprocal Rank (RR) for a single query.
    """
    desc_score_indices = np.argsort(y_pred_scores)[::-1]
    y_true_sorted = np.asarray(y_true)[desc_score_indices]
    
    for i, relevant in enumerate(y_true_sorted):
        if relevant == 1:
            return 1.0 / (i + 1)
    return 0.0

def dcg_at_k(r, k, method=0):
    """
    Discounted Cumulative Gain at k.
    """
    r = np.asfarray(r)[:k]
    if r.size:
        if method == 0:
            return np.sum(r / np.log2(np.arange(2, r.size + 2)))
        elif method == 1:
            return np.sum(r / np.log2(np.arange(1, r.size + 1) + 1))
        else:
            raise ValueError('method must be 0 or 1.')
    return 0.

def ndcg_at_k(y_true, y_pred_scores, k, method=0):
    """
    Normalized Discounted Cumulative Gain at k for a single query.
    """
    desc_score_indices = np.argsort(y_pred_scores)[::-1]
    y_true_sorted_by_pred = np.asarray(y_true)[desc_score_indices]
    actual_dcg = dcg_at_k(y_true_sorted_by_pred, k, method)
    y_true_ideal_sorted = np.sort(y_true)[::-1]
    ideal_dcg = dcg_at_k(y_true_ideal_sorted, k, method)
    if ideal_dcg == 0:
        return 0.
    return actual_dcg / ideal_dcg

def precision_at_k(y_true, y_pred_scores, k):
    """
    Precision at k for a single query.
    """
    desc_score_indices = np.argsort(y_pred_scores)[::-1][:k]
    y_true_top_k = np.asarray(y_true)[desc_score_indices]
    if len(y_true_top_k) == 0:
        return 0.0
    return np.sum(y_true_top_k) / len(y_true_top_k)

# --- END Helper functions for manual metric calculation ---

# --- 1. Get Paths from User ---
print("--- MatchZoo Evaluation Setup ---")
default_model_dir = r"D:\\SemanticSearch\\TrainedModels\\model_KNRM_grouping_20250507"
SAVED_MODEL_PATH = get_user_path(
    prompt_message="Enter the full path to the TRAINED MODEL DIRECTORY:",
    default_path=default_model_dir,
    is_directory=True
)
SAVED_PREPROCESSOR_PATH = os.path.join(SAVED_MODEL_PATH, "preprocessor.dill")
default_test_file = r"D:/SemanticSearch/TrainingData_MatchZoo_BEIR/msmarco_semantic-grouping/test_1/msmarco_semantic-grouping_test_triplets.tsv"
TEST_TRIPLETS_FILE = get_user_path(
    prompt_message="Enter the full path to the TEST TRIPLETS FILE (e.g., .tsv):",
    default_path=default_test_file,
    is_file=True
)
print("\\n--- Paths Configured ---")
print(f"Model Directory: {SAVED_MODEL_PATH}")
print(f"Preprocessor Path: {SAVED_PREPROCESSOR_PATH}")
print(f"Test Triplets File: {TEST_TRIPLETS_FILE}")
print("------------------------\\n")

print("--- MatchZoo Evaluation ---")

# Monkey-patch ParamTable.get for compatibility if needed
try:
    def new_paramtable_get(self, key, default=None):
        if key in self._params:
            param_obj = self._params[key]
            return param_obj 
        return default
    ParamTable.get = new_paramtable_get
    print("Applied monkey-patch to matchzoo.engine.param_table.ParamTable.get.")
except AttributeError:
    print("Could not monkey-patch ParamTable.get (AttributeError).")

# 1. Tải Preprocessor và Model
print(f"Loading preprocessor from: {SAVED_PREPROCESSOR_PATH}")
if not os.path.exists(SAVED_PREPROCESSOR_PATH) or not os.path.isfile(SAVED_PREPROCESSOR_PATH):
    print(f"Error: Preprocessor file not found or is not a file at {SAVED_PREPROCESSOR_PATH}")
    sys.exit(1)

try:
    with open(SAVED_PREPROCESSOR_PATH, 'rb') as f:
        preprocessor = dill.load(f)
    print("Preprocessor loaded successfully.")
except Exception as e:
    print(f"Error loading preprocessor: {e}")
    sys.exit(1)

print(f"Loading model from: {SAVED_MODEL_PATH}")
TASK = None 

# Manual Model Loading and Reconstruction
MODEL_NAME_TO_RUN = "KNRM" 
EMBEDDING_DIM = 100 
HINGE_MARGIN = 1.0 

model = None
try:
    print(f"Attempting to reconstruct model '{MODEL_NAME_TO_RUN}' and load weights...")
    if MODEL_NAME_TO_RUN == "KNRM":
        model = mz.models.KNRM()
    else:
        print(f"Error: Model type '{MODEL_NAME_TO_RUN}' is not explicitly handled for reconstruction.")
        sys.exit(1)

    if hasattr(preprocessor, 'context') and preprocessor.context and 'task' in preprocessor.context:
        TASK = preprocessor.context['task']
        if not isinstance(TASK, (mz.tasks.Ranking, mz.tasks.Classification)):
            print(f"Warning: Task from preprocessor context ({type(TASK)}) is not standard. Defaulting to Ranking.")
            TASK = mz.tasks.Ranking()
    else:
        TASK = mz.tasks.Ranking()
    
    model.params['task'] = TASK
    loss_function = mz.losses.RankHingeLoss(margin=HINGE_MARGIN)
    evaluation_metrics_train = [
        mz.metrics.MeanAveragePrecision(), 
        mz.metrics.NormalizedDiscountedCumulativeGain(k=3),
        mz.metrics.NormalizedDiscountedCumulativeGain(k=5),
        mz.metrics.MeanReciprocalRank()
    ] 
    model.params['loss'] = loss_function
    model.params['metrics'] = evaluation_metrics_train
    
    if (
        hasattr(preprocessor, 'context') and 
        preprocessor.context and  # Kiểm tra preprocessor.context không None trước khi truy cập sâu hơn
        'vocab_unit' in preprocessor.context and 
        hasattr(preprocessor.context['vocab_unit'], 'state') and 
        preprocessor.context['vocab_unit'].state and # Kiểm tra state không None
        'term_index' in preprocessor.context['vocab_unit'].state
    ):
        vocab_size = len(preprocessor.context['vocab_unit'].state['term_index']) + 1
    else:
        print("ERROR: Cannot determine vocab_size from loaded preprocessor.")
        sys.exit(1)

    embedding_params_to_add = {
        'embedding_input_dim': vocab_size,
        'embedding_output_dim': EMBEDDING_DIM,
        'embedding_trainable': False 
    }
    for param_name, param_value in embedding_params_to_add.items():
        model.params[param_name] = param_value
            
    model.guess_and_fill_missing_params()
    model.build()
    
    weights_path = os.path.join(SAVED_MODEL_PATH, 'weights.h5')
    if not os.path.exists(weights_path):
        print(f"Error: Weights file 'weights.h5' not found in {SAVED_MODEL_PATH}")
        sys.exit(1)
    
    model.backend.load_weights(weights_path)
    print(f"Model '{MODEL_NAME_TO_RUN}' reconstructed and weights loaded successfully.")

    if 'task' in model.params and isinstance(model.params['task'], (mz.tasks.Ranking, mz.tasks.Classification)):
        TASK = model.params['task']
    elif TASK is None:
        print("Critical Error: TASK is None after model reconstruction.")
        sys.exit(1)

except Exception as e:
    print(f"Error reconstructing model or loading weights: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 2. Chuẩn bị Dữ liệu Test
print(f"Loading test triplets from: {TEST_TRIPLETS_FILE}")
if not os.path.exists(TEST_TRIPLETS_FILE):
    print(f"Error: Test triplets file not found at {TEST_TRIPLETS_FILE}")
    sys.exit(1)

try:
    triplets_df = pd.read_csv(TEST_TRIPLETS_FILE, sep='\\t', header=None, names=['query', 'doc_pos', 'doc_neg'], quoting=3, encoding='utf-8', skiprows=[0])
    print(f"Loaded {len(triplets_df)} triplets.")
    
    if triplets_df.shape[1] != 3:
        print(f"ERROR: Expected 3 columns, got {triplets_df.shape[1]}. Please check TSV format.")
        sys.exit(1)
    if triplets_df.isnull().values.any():
        print("WARNING: NaN values found in the initial loaded test data.")

    print("Preparing DataPack for test data...")
    unique_queries_text = triplets_df['query'].astype(str).unique()
    query_text_to_main_id = {text: f"qmain_{i}" for i, text in enumerate(unique_queries_text)}
    
    queries_dict = {} 
    docs_dict = {}    
    relations_list = [] 
    doc_counter = 0

    for i, row in tqdm(triplets_df.iterrows(), total=len(triplets_df), desc="Processing test triplets"):
        query_text = str(row['query'])
        doc_pos_text = str(row['doc_pos'])
        doc_neg_text = str(row['doc_neg'])
        query_main_id = query_text_to_main_id[query_text]
        
        if query_main_id not in queries_dict:
             queries_dict[query_main_id] = query_text
        
        doc_pos_processing_id = f"doc_{doc_counter}"; docs_dict[doc_pos_processing_id] = doc_pos_text; doc_counter += 1
        doc_neg_processing_id = f"doc_{doc_counter}"; docs_dict[doc_neg_processing_id] = doc_neg_text; doc_counter += 1
        
        relations_list.append({'id_left': query_main_id, 'id_right': doc_pos_processing_id, 'label': 1})
        relations_list.append({'id_left': query_main_id, 'id_right': doc_neg_processing_id, 'label': 0})

    if not queries_dict or not docs_dict or not relations_list:
        print("Error: Failed to create non-empty queries, docs, or relations from triplets. Exiting.")
        sys.exit(1)

    left_df = pd.DataFrame(list(queries_dict.items()), columns=['id_left', 'text_left'])
    right_df = pd.DataFrame(list(docs_dict.items()), columns=['id_right', 'text_right'])
    relation_df = pd.DataFrame(relations_list)
    
    left_df['text_left'] = left_df['text_left'].astype(str)
    right_df['text_right'] = right_df['text_right'].astype(str)
    relation_df['label'] = relation_df['label'].astype(int)

    test_data_pack_raw = mz.DataPack(relation=relation_df, left=left_df, right=right_df)
    print("Applying preprocessor to test data...")
    test_pack_transformed_by_preprocessor = preprocessor.transform(test_data_pack_raw, verbose=0)
    
    processed_test_left_df = test_pack_transformed_by_preprocessor.left
    processed_test_right_df = test_pack_transformed_by_preprocessor.right
    processed_test_relation_df = test_pack_transformed_by_preprocessor.relation

    if 'id_left' in processed_test_left_df.columns:
        processed_test_left_df = processed_test_left_df.set_index('id_left', drop=True)
    elif processed_test_left_df.index.name != 'id_left':
        processed_test_left_df.index.name = 'id_left'

    if 'id_right' in processed_test_right_df.columns:
        processed_test_right_df = processed_test_right_df.set_index('id_right', drop=True)
    elif processed_test_right_df.index.name != 'id_right':
        processed_test_right_df.index.name = 'id_right'
    
    valid_test_left_ids = set(processed_test_left_df.index)
    valid_test_right_ids = set(processed_test_right_df.index)
    original_test_relation_count = len(processed_test_relation_df)
    processed_test_relation_df = processed_test_relation_df[
        processed_test_relation_df['id_left'].isin(valid_test_left_ids) &
        processed_test_relation_df['id_right'].isin(valid_test_right_ids)
    ]
    filtered_test_relation_count = len(processed_test_relation_df)
    if original_test_relation_count > filtered_test_relation_count:
        print(f"INFO: Test relations filtered: {original_test_relation_count} -> {filtered_test_relation_count}")
    if filtered_test_relation_count == 0 and original_test_relation_count > 0:
        print("ERROR: All test relations were filtered out.")
        sys.exit(1)

    test_pack_for_generator = mz.DataPack(
        relation=processed_test_relation_df,
        left=processed_test_left_df,
        right=processed_test_right_df
    )
    print("Test DataPack prepared for generator.")

except Exception as e:
    print(f"Error during data preparation or preprocessing: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 3. Tạo DataGenerator và Thực hiện Prediction
print("Creating DataGenerator and predicting...")
try:
    EVAL_BATCH_SIZE = 32 
    if not isinstance(test_pack_for_generator, mz.DataPack):
        print(f"Warning: test_pack_for_generator is not a mz.DataPack instance (type: {type(test_pack_for_generator)}).")

    predict_generator = mz.DataGenerator(
        data_pack=test_pack_for_generator, 
        mode='point',
        batch_size=EVAL_BATCH_SIZE,
        shuffle=False
    )
    
    if hasattr(model.backend, 'predict_generator'):
        predictions = model.backend.predict_generator(predict_generator, steps=len(predict_generator), verbose=0)
    else:
        print("model.backend.predict_generator() not found. Using model.backend.predict().")
        all_preds = []
        for i in tqdm(range(len(predict_generator)), desc="Predicting batch by batch (fallback)"):
            batch_x, _ = predict_generator[i] 
            batch_pred = model.backend.predict(batch_x) 
            all_preds.append(batch_pred)
        if all_preds:
            predictions = np.concatenate(all_preds, axis=0)
        else:
            predictions = np.array([])
            print("Warning: No predictions generated from fallback method.")

    print(f"Predictions obtained. Shape: {predictions.shape}")
    num_relations_for_eval = len(test_pack_for_generator.relation)
    if predictions.shape[0] != num_relations_for_eval:
        print(f"CRITICAL WARNING: Number of predictions ({predictions.shape[0]}) does not match relations ({num_relations_for_eval}). Results will be incorrect.")

except Exception as e:
    print(f"Error during DataGenerator creation or prediction: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 4. Đánh giá
print("Evaluating predictions...")
try:
    if not hasattr(test_pack_for_generator, 'relation') or not isinstance(test_pack_for_generator.relation, pd.DataFrame) or test_pack_for_generator.relation.empty:
        print("Error: test_pack_for_generator.relation is not available, not a DataFrame, or is empty. Cannot proceed with evaluation.")
        sys.exit(1)
        
    relation_df_processed = test_pack_for_generator.relation 
    true_labels_flat = relation_df_processed['label'].tolist() 
    query_ids_flat_for_grouping = relation_df_processed['id_left'].tolist()

    if predictions.ndim > 1 and predictions.shape[1] == 1:
        predicted_scores_flat = predictions.flatten()
    elif predictions.ndim == 1:
        predicted_scores_flat = predictions
    else: 
        print(f"Predictions have shape {predictions.shape}. Assuming column 1 contains positive/relevance scores if multi-column.")
        if predictions.shape[1] > 1 :
            predicted_scores_flat = predictions[:, 1] 
        else:
             print("Error: Prediction shape ambiguous for extracting scores.")
             sys.exit(1)
             
    len_true = len(true_labels_flat)
    len_query_ids = len(query_ids_flat_for_grouping)
    len_preds = len(predicted_scores_flat)
    if not (len_true == len_query_ids == len_preds):
        print(f"ERROR: Mismatch in lengths before grouping: true_labels={len_true}, query_ids={len_query_ids}, pred_scores={len_preds}")
        sys.exit(1)
    if np.isnan(predicted_scores_flat).any() or np.isinf(predicted_scores_flat).any():
        print("WARNING: NaN or Inf found in predicted_scores_flat before grouping!")

    if not isinstance(TASK, (mz.tasks.Ranking, mz.tasks.Classification)):
        print(f"Error: TASK type {type(TASK)} is not recognized for evaluation. Exiting.")
        sys.exit(1)

    if isinstance(TASK, mz.tasks.Ranking):
        eval_data = pd.DataFrame({
            'query_id': query_ids_flat_for_grouping, 
            'true_label': true_labels_flat,
            'pred_score': predicted_scores_flat
        })
        y_true_grouped_for_mz = []
        y_pred_grouped_for_mz = []
        group_lengths_consistent = True
        inconsistent_groups = []

        grouped_eval_data = eval_data.groupby('query_id')
        if len(grouped_eval_data) == 0 and len(eval_data) > 0:
             print("Error: No groups formed by groupby('query_id'). Check 'query_id' column.")
             sys.exit(1)

        for q_id, group in grouped_eval_data:
            current_true_labels = np.array(group['true_label'].tolist(), dtype=np.int32)
            current_pred_scores = np.array(group['pred_score'].tolist(), dtype=np.float32)
            if len(current_true_labels) != len(current_pred_scores):
                print(f"ERROR: Mismatch in length for group {q_id}: len(true)={len(current_true_labels)}, len(pred)={len(current_pred_scores)}")
                group_lengths_consistent = False; inconsistent_groups.append(q_id)
            y_true_grouped_for_mz.append(current_true_labels)
            y_pred_grouped_for_mz.append(current_pred_scores)
        
        if not group_lengths_consistent:
             print(f"ERROR: Length mismatches found within groups: {inconsistent_groups}. Stopping.")
             sys.exit(1)

        print(f"Data prepared for ranking metrics: {len(y_true_grouped_for_mz)} query groups.")
        
        print("Calculating metrics manually for each query group...")
        all_aps = []
        all_rrs = []
        all_ndcgs_at_3 = []
        all_ndcgs_at_5 = []
        all_precisions_at_1 = []

        if not y_true_grouped_for_mz:
            print("WARNING: No data grouped for ranking metrics. Skipping manual calculation.")
        else:
            for i in tqdm(range(len(y_true_grouped_for_mz)), desc="Calculating metrics per query"):
                y_true_single_query = y_true_grouped_for_mz[i]
                y_pred_single_query = y_pred_grouped_for_mz[i]
                if len(y_true_single_query) == 0: continue
                all_aps.append(average_precision(y_true_single_query, y_pred_single_query))
                all_rrs.append(reciprocal_rank(y_true_single_query, y_pred_single_query))
                all_ndcgs_at_3.append(ndcg_at_k(y_true_single_query, y_pred_single_query, k=3))
                all_ndcgs_at_5.append(ndcg_at_k(y_true_single_query, y_pred_single_query, k=5))
                all_precisions_at_1.append(precision_at_k(y_true_single_query, y_pred_single_query, k=1))

        manual_results = {}
        if all_aps: manual_results["MAP"] = np.mean(all_aps)
        if all_rrs: manual_results["MRR"] = np.mean(all_rrs)
        if all_ndcgs_at_3: manual_results["NDCG@3"] = np.mean(all_ndcgs_at_3)
        if all_ndcgs_at_5: manual_results["NDCG@5"] = np.mean(all_ndcgs_at_5)
        if all_precisions_at_1: manual_results["P@1"] = np.mean(all_precisions_at_1)
        
        print("\\n--- Manually Calculated Ranking Metrics Results ---")
        for metric_name, metric_val in manual_results.items():
            print(f"{metric_name}: {metric_val:.4f}")
        print("---------------------------------------------")

    elif isinstance(TASK, mz.tasks.Classification):
        print("Evaluating classification task...")
        try:
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
            sklearn_available = True
        except ImportError:
            print("sklearn not installed. Install it using: pip install scikit-learn")
            sklearn_available = False
            
        predicted_classes = (predicted_scores_flat > 0.5).astype(int)
        print("\\n--- Classification Evaluation Results ---")
        if not sklearn_available:
            print("Skipping detailed metrics - sklearn.metrics not available")
        else:
            try:
                acc = accuracy_score(true_labels_flat, predicted_classes)
                print(f"Accuracy: {acc:.4f}")
            except Exception as e: 
                print(f"Could not calculate Accuracy: {e}")
            try:
                prec = precision_score(true_labels_flat, predicted_classes, average='binary', zero_division=0)
                print(f"Precision (binary): {prec:.4f}")
            except Exception as e: 
                print(f"Could not calculate Precision: {e}")
            try:
                rec = recall_score(true_labels_flat, predicted_classes, average='binary', zero_division=0)
                print(f"Recall (binary): {rec:.4f}")
            except Exception as e: 
                print(f"Could not calculate Recall: {e}")
            try:
                f1 = f1_score(true_labels_flat, predicted_classes, average='binary', zero_division=0)
                print(f"F1-score (binary): {f1:.4f}")
            except Exception as e: 
                print(f"Could not calculate F1-score: {e}")
        print("---------------------------------------")
    else:
        print(f"Error: Unsupported task type for evaluation: {type(TASK)}")

except Exception as e:
    print(f"Error during evaluation: {e}")
    print("Traceback for evaluation error:")
    import traceback
    traceback.print_exc()

print("\\nEvaluation script finished.")