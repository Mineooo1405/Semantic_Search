import pandas as pd
import nltk
import os
import re
from tqdm.auto import tqdm
from beir import util
from beir.datasets.data_loader import GenericDataLoader
from typing import List, Tuple, Dict, Callable, Optional, Union
from concurrent.futures import ProcessPoolExecutor, as_completed
import torch
import torch_directml
import atexit
import json
import functools
import random
import traceback
from sklearn.model_selection import train_test_split
import datetime # ADDED for logging timestamps

from Semantic_Grouping import semantic_chunk_passage_from_grouping_logic
from Semantic_Splitter import chunk_passage_semantic_splitter
from Text_Splitter import chunk_passage_text_splitter    
# Sửa đổi import nếu OIE.py đã tích hợp caching vào extract_triples
from Tool.OIE import extract_relations_from_paragraph, close_oie_clients # MODIFIED OIE import

# Global variable for the main process DirectML device
MAIN_PROCESS_DML_DEVICE = None
# Global variable for worker-specific device, will be set by init_worker_directml
DML_WORKER_DEVICE = None

try:
    if torch_directml.is_available() and hasattr(torch_directml, 'device') and callable(torch_directml.device):
        MAIN_PROCESS_DML_DEVICE = torch_directml.device()
        print(f"Main process: DirectML device enabled: {MAIN_PROCESS_DML_DEVICE}")
    else:
        print("Main process: DirectML not available or torch_directml.device() not found. Falling back to CPU or other PyTorch default.")
        if torch.cuda.is_available():
            MAIN_PROCESS_DML_DEVICE = torch.device('cuda')
            print(f"Main process: Using CUDA device: {MAIN_PROCESS_DML_DEVICE}")
        else:
            MAIN_PROCESS_DML_DEVICE = torch.device('cpu')
            print(f"Main process: Using CPU device: {MAIN_PROCESS_DML_DEVICE}")
except Exception as e:
    print(f"Main process: Error initializing DirectML or fallback device: {e}. Defaulting to CPU.")
    MAIN_PROCESS_DML_DEVICE = torch.device('cpu')


# Đăng ký hàm để được gọi khi script thoát
atexit.register(close_oie_clients) # <<< THÊM DÒNG NÀY

# --- UI Helper Functions ---
# (get_input, get_int_input, get_float_input, get_auto_or_float_input, select_chunking_method, get_common_params giữ nguyên)
def get_input(prompt: str, default: Optional[str] = None, required: bool = False) -> str:
    while True:
        if default is not None:
            response = input(f"{prompt} (default: {default}): ").strip()
            if not response: return default
        else:
            response = input(f"{prompt}: ").strip()
        if response: return response
        elif required: print("Input is required. Please try again.")
        else: return ""

def get_int_input(prompt: str, default: Optional[int] = None, required: bool = False, min_val: Optional[int] = None, max_val: Optional[int] = None) -> Optional[int]:
    default_str = str(default) if default is not None else None
    while True:
        response_str = get_input(prompt, default_str, required)
        if not response_str and not required: return None
        if not response_str and required and default is not None: response_str = str(default)
        try:
            value = int(response_str)
            if min_val is not None and value < min_val: print(f"Value must be at least {min_val}. Please try again."); continue
            if max_val is not None and value > max_val: print(f"Value must be at most {max_val}. Please try again."); continue
            return value
        except ValueError: print("Invalid input. Please enter an integer.")
        except TypeError:
             if not required: return None
             print("Input error. Please try again.")

def get_float_input(prompt: str, default: Optional[float] = None, required: bool = False, min_val: Optional[float] = None, max_val: Optional[float] = None) -> Optional[float]:
    default_str = str(default) if default is not None else None
    while True:
        response_str = get_input(prompt, default_str, required)
        if not response_str and not required: return None
        if not response_str and required and default is not None: response_str = str(default)
        try:
            value = float(response_str)
            if min_val is not None and value < min_val: print(f"Value must be at least {min_val}. Please try again."); continue
            if max_val is not None and value > max_val: print(f"Value must be at most {max_val}. Please try again."); continue
            return value
        except ValueError: print("Invalid input. Please enter a number (e.g., 0.85).")
        except TypeError:
             if not required: return None
             print("Input error. Please try again.")

def get_auto_or_float_input(prompt: str, default: Union[str, float] = 'auto', required: bool = True) -> Union[str, float]:
    default_str = str(default)
    while True:
        response_str = get_input(prompt, default_str, required)
        if response_str.lower() == 'auto': return 'auto'
        try:
            value = float(response_str)
            if 0.0 <= value <= 1.0: return value
            else: print("Float value should typically be between 0.0 and 1.0 for thresholds.")
        except ValueError: print("Invalid input. Please enter 'auto' or a number (e.g., 0.8).")
        except TypeError:
             if not required: return 'auto'
             print("Input error. Please try again.")

def select_chunking_method() -> str:
    print("\n--- Select Chunking Method ---")
    methods = ['semantic_grouping', 'semantic_splitter', 'text_splitter']
    for i, method in enumerate(methods): print(f"{i+1}. {method}")
    while True:
        try:
            choice = int(input("Enter the number of the desired method: "))
            if 1 <= choice <= len(methods): return methods[choice-1]
            else: print("Invalid choice. Please enter a number from the list.")
        except ValueError: print("Invalid input. Please enter a number.")

def get_common_params() -> Dict:
    print("\n--- Common Configuration ---")
    params = {}
    params['dataset_name'] = get_input("BEIR Dataset Name", default="msmarco", required=True)
    params['data_dir'] = get_input("Directory for BEIR data", default="./Data/msmarco", required=True)
    params['output_dir'] = get_input("Base Output Directory", default="D:/SemanticSearch/TrainingData_MatchZoo_BEIR", required=True)
    params['embedding_model'] = get_input("Embedding Model Name (for semantic methods)", default="thenlper/gte-large", required=True)
    max_triplets = get_int_input("Max Triplets (0 for unlimited)", default=20000, required=True, min_val=0) # Đổi default thành 20000 như yêu cầu
    params['max_triplets'] = max_triplets if max_triplets > 0 else None
    max_docs = get_int_input("Max Documents to Process (0 or empty for all)", default=0, required=False, min_val=0)
    params['max_docs'] = max_docs if max_docs is not None and max_docs > 0 else None
    params['include_oie'] = get_input("Include OIE features in chunks? (yes/no)", default="no").lower() == 'yes'
    params['random_seed'] = get_int_input("Random Seed for triplet sampling (e.g., 42)", default=42, required=True) # <-- THÊM RANDOM SEED
    return params

# --- MODIFIED get_semantic_grouping_params ---
def get_semantic_grouping_params(embedding_model: str) -> Dict:
    """Gets parameters specific to Semantic Grouping."""
    print("\n--- Semantic Grouping Parameters ---")
    params = {'model_name': embedding_model} # Include model name
    params['initial_threshold'] = get_auto_or_float_input("Initial Threshold ('auto' or float 0-1)", default='auto')
    params['decay_factor'] = get_float_input("Decay Factor (e.g., 0.85)", default=0.85, required=True, min_val=0.0, max_val=1.0)
    params['min_threshold'] = get_auto_or_float_input("Minimum Threshold ('auto' or float 0-1)", default='auto')
    if params['initial_threshold'] == 'auto' or params['min_threshold'] == 'auto':
        print("--- Auto Threshold Percentiles (used if threshold is 'auto') ---")
        initial_p = get_int_input("Initial Percentile (e.g., 85)", default=85, required=True, min_val=1, max_val=99)
        min_p = get_int_input("Minimum Percentile (e.g., 25)", default=25, required=True, min_val=1, max_val=99)
        params['auto_percentiles'] = (initial_p, min_p)
    else:
        params['auto_percentiles'] = (85, 25) # Default if not used

    # <-- ADDED BATCH SIZE PROMPT -->
    params['embedding_batch_size'] = get_int_input(
        "Embedding Batch Size (smaller reduces RAM, e.g., 16, 32)",
        default=16,
        required=True,
        min_val=1
    )
    return params

def get_semantic_splitter_params(embedding_model: str) -> Dict:
    print("\n--- Semantic Splitter Parameters ---")
    params = {'model_name': embedding_model}
    params['initial_threshold'] = get_float_input("Initial Threshold (e.g., 0.6)", default=0.6, required=True, min_val=0.0, max_val=1.0)
    params['decay_factor'] = get_float_input("Decay Factor (e.g., 0.95)", default=0.95, required=True, min_val=0.0, max_val=1.0)
    params['min_threshold'] = get_float_input("Minimum Threshold (e.g., 0.35)", default=0.35, required=True, min_val=0.0, max_val=1.0)
    params['min_chunk_len'] = get_int_input("Min Sentences per Chunk", default=2, required=True, min_val=1)
    params['max_chunk_len'] = get_int_input("Max Sentences per Chunk", default=8, required=True, min_val=1)
    params['window_size'] = get_int_input("Trend Analysis Window Size", default=3, required=True, min_val=1)
    params['embedding_batch_size'] = get_int_input(
        "Embedding Batch Size (e.g., 32, 64, 128, ảnh hưởng VRAM)",
        default=32, # Có thể tăng giá trị này cho GPU mạnh
        required=True,
        min_val=1
    )
    return params

def get_text_splitter_params() -> Dict:
    print("\n--- Text Splitter Parameters ---")
    params = {}
    params['chunk_size'] = get_int_input("Chunk Size (characters)", default=1000, required=True, min_val=50)
    params['chunk_overlap'] = get_int_input("Chunk Overlap (characters)", default=200, required=True, min_val=0)
    return params

def get_split_type() -> str:
    """Prompts the user to select the data split type (train or test)."""
    print("\n--- Select Data Split ---")
    while True:
        response = input("Process data for 'train' or 'test'? ").strip().lower()
        if response in ['train', 'test']:
            return response
        else:
            print("Invalid input. Please enter 'train' or 'test'.")

# --- Worker Function for Parallel Chunking ---
# Note: Please ensure any duplicate init_worker functions are removed.
# The ProcessPoolExecutor should be initialized with initializer=init_worker_directml

def init_worker_directml():
    """Initializer for each worker process to set up DirectML device."""
    global DML_WORKER_DEVICE # This global is specific to each worker process
    try:
        # Ensure torch and torch_directml are accessible in the worker
        import torch
        import torch_directml
        if torch_directml.is_available() and hasattr(torch_directml, 'device') and callable(torch_directml.device):
            DML_WORKER_DEVICE = torch_directml.device()
            # print(f"Worker {os.getpid()} initialized. Using DirectML device: {DML_WORKER_DEVICE}")
        else:
            # print(f"Worker {os.getpid()}: DirectML not available or torch_directml.device() not found. Trying CPU/CUDA.")
            if torch.cuda.is_available():
                DML_WORKER_DEVICE = torch.device('cuda')
                # print(f"Worker {os.getpid()}: Using CUDA device: {DML_WORKER_DEVICE}")
            else:
                DML_WORKER_DEVICE = torch.device('cpu')
                # print(f"Worker {os.getpid()}: Using CPU device: {DML_WORKER_DEVICE}")
    except Exception as e:
        print(f"Worker {os.getpid()} error during DirectML/device setup: {e}. Defaulting to CPU.")
        DML_WORKER_DEVICE = torch.device('cpu')

def chunk_document_worker(
    doc_item: Tuple[str, Dict],
    chunking_func: Callable,
    params: Dict
) -> Optional[Tuple[str, List[Tuple[str, str, Optional[str]]]]]: # Ensure return type matches usage
    doc_id, doc_data = doc_item
    try:
        # DML_WORKER_DEVICE is set by init_worker_directml for this worker process.
        # Add this device to the params passed to the actual chunking function.
        # Chunking functions (e.g., semantic_grouping, semantic_splitter) must be
        # updated to accept 'device' in **kwargs and pass it to their embedding calls.
        worker_params = params.copy()
        worker_params['device'] = DML_WORKER_DEVICE # DML_WORKER_DEVICE is global *within this worker*

        passage_text = doc_data.get("text", "")
        title = doc_data.get("title", "")
        full_text_to_chunk = f"{title}. {passage_text}" if title and passage_text else passage_text or title

        if not full_text_to_chunk.strip():
            # print(f"Skipping document {doc_id} due to empty content after combining title and text.") # Optional: log skips
            return (doc_id, []) # Return doc_id and empty list if no content

        # --- ADDED DETAILED LOGGING ---
        current_time_start = datetime.datetime.now().isoformat()
        oie_active = params.get('include_oie', False)
        print(f"[{current_time_start}] Worker {os.getpid()} STARTING chunking for doc_id: {doc_id} using {chunking_func.__name__} (OIE: {oie_active})", flush=True)
        
        raw_doc_chunks_result = chunking_func(doc_id, full_text_to_chunk, **worker_params)
        
        current_time_done = datetime.datetime.now().isoformat()
        num_chunks_generated = len(raw_doc_chunks_result) if isinstance(raw_doc_chunks_result, list) else 'N/A (or error)'
        print(f"[{current_time_done}] Worker {os.getpid()} COMPLETED chunking for doc_id: {doc_id}. Chunks: {num_chunks_generated}", flush=True)
        # --- END DETAILED LOGGING ---

        final_doc_chunks: List[Tuple[str, str, Optional[str]]] = []
        if raw_doc_chunks_result and isinstance(raw_doc_chunks_result, list):
            # Assuming chunking_func already returns the correct flat list format
            # List[Tuple[str, str, Optional[str]]]
            final_doc_chunks = raw_doc_chunks_result
        # else:
            # print(f"Warning: Unexpected or empty result from {chunking_func.__name__} for {doc_id}")


        if not final_doc_chunks:
            # print(f"No chunks generated for document {doc_id} by {chunking_func.__name__}")
            return doc_id, []

        return doc_id, final_doc_chunks

    except Exception as e:
        print(f"Error processing document {doc_id} in worker {os.getpid()}: {e}")
        traceback.print_exc()
        return doc_id, [] # Return empty list on error

# --- Helper Functions ---
def clean_text(text):
    text = str(text)
    text = text.replace('\n', ' ').replace('\r', ' ')
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def find_next_run_dir(base_output_dir: str, split_prefix: str) -> str:
    """
    Finds the next available directory name like 'train_1', 'train_2', etc.
    within the base_output_dir.
    """
    os.makedirs(base_output_dir, exist_ok=True) # Ensure base directory exists
    counter = 1
    while True:
        run_dir = os.path.join(base_output_dir, f"{split_prefix}_{counter}")
        if not os.path.exists(run_dir):
            print(f"Next available run directory: {run_dir}")
            return run_dir
        counter += 1

# --- OIE Helper Functions ---
# (Removed format_oie_triples_to_string and get_chunk_text_with_oie as OIE is now handled by individual chunking scripts)
# --- End OIE Helper Functions ---

# --- ĐẶT TOÀN BỘ LOGIC CHÍNH VÀO ĐÂY ---
if __name__ == "__main__":

    # --- Get Configuration from User ---
    common_config = get_common_params()
    selected_method = select_chunking_method()
    split_type = get_split_type()

    # --- Assign config values to variables ---
    MAX_DOCS_TO_PROCESS = common_config.get('max_docs')
    RANDOM_SEED = common_config['random_seed'] # This will be used for train_test_split
    MAX_TRIPLETS_TO_GENERATE = common_config.get('max_triplets')

    method_params = {}
    if selected_method == 'semantic_grouping':
        method_params = get_semantic_grouping_params(common_config['embedding_model'])
    elif selected_method == 'semantic_splitter':
        method_params = get_semantic_splitter_params(common_config['embedding_model'])
    elif selected_method == 'text_splitter':
        method_params = get_text_splitter_params()
    
    method_params['include_oie'] = common_config.get('include_oie', False)
    if selected_method != 'text_splitter':
        method_params['device'] = MAIN_PROCESS_DML_DEVICE

    # --- Select Chunking Function ---
    selected_chunking_func: Callable
    if selected_method == 'semantic_grouping':
        selected_chunking_func = semantic_chunk_passage_from_grouping_logic
    elif selected_method == 'semantic_splitter':
        selected_chunking_func = chunk_passage_semantic_splitter
    elif selected_method == 'text_splitter':
        selected_chunking_func = chunk_passage_text_splitter
    else:
        raise ValueError(f"Unsupported chunking method: {selected_method}")

    # --- Combine Configurations & Define Paths ---
    DATASET_NAME = common_config['dataset_name']
    DATA_DIR = common_config['data_dir']
    BASE_OUTPUT_DIR = common_config['output_dir']
    INCLUDE_OIE = common_config.get('include_oie', False)

    SHARED_DATA_ROOT_DIR = os.path.join(BASE_OUTPUT_DIR, f"{DATASET_NAME}_shared_original_data")
    os.makedirs(SHARED_DATA_ROOT_DIR, exist_ok=True)
    SHARED_CORPUS_CSV = os.path.join(SHARED_DATA_ROOT_DIR, "corpus.csv")
    SHARED_QUERIES_CSV = os.path.join(SHARED_DATA_ROOT_DIR, "queries.csv")
    SHARED_QRELS_CSV = os.path.join(SHARED_DATA_ROOT_DIR, f"qrels_{split_type}.csv")

    method_oie_suffix = "_oie" if INCLUDE_OIE else ""
    effective_method_name_for_path = f"{selected_method.replace('_', '-')}{method_oie_suffix}"

    BASE_METHOD_DIR = os.path.join(BASE_OUTPUT_DIR, f"{DATASET_NAME}_{effective_method_name_for_path}")
    OUTPUT_DIR_METHOD = find_next_run_dir(BASE_METHOD_DIR, split_type)
    os.makedirs(OUTPUT_DIR_METHOD, exist_ok=True)

    METHOD_AND_SPLIT_PREFIX = f"{DATASET_NAME}_{effective_method_name_for_path}_{split_type}"

    CHUNKS_OUTPUT_PATH = os.path.join(OUTPUT_DIR_METHOD, f"{METHOD_AND_SPLIT_PREFIX}_chunks.jsonl")
    DOC_CHUNK_MAP_PATH = os.path.join(OUTPUT_DIR_METHOD, f"{METHOD_AND_SPLIT_PREFIX}_doc_chunk_map.json")
    CHUNK_DOC_MAP_PATH = os.path.join(OUTPUT_DIR_METHOD, f"{METHOD_AND_SPLIT_PREFIX}_chunk_doc_map.json")
    OUTPUT_TRIPLETS_PATH = os.path.join(OUTPUT_DIR_METHOD, f"{METHOD_AND_SPLIT_PREFIX}_triplets.tsv")
    
    # --- ADDED: Paths for run-specific corpus/query subsets for analysis ---
    RUN_SPECIFIC_CORPUS_INPUT_CSV_PATH = os.path.join(OUTPUT_DIR_METHOD, f"{METHOD_AND_SPLIT_PREFIX}_corpus_for_processing.csv")
    RUN_SPECIFIC_QUERIES_INPUT_CSV_PATH = os.path.join(OUTPUT_DIR_METHOD, f"{METHOD_AND_SPLIT_PREFIX}_queries_for_processing.csv")

    # --- ADDED: Paths for MatchZoo formatted train/dev data ---
    TRAIN_MZ_TSV_PATH = os.path.join(OUTPUT_DIR_METHOD, f"{METHOD_AND_SPLIT_PREFIX}_train_mz.tsv")
    DEV_MZ_TSV_PATH = os.path.join(OUTPUT_DIR_METHOD, f"{METHOD_AND_SPLIT_PREFIX}_dev_mz.tsv")

    # --- Load BEIR Dataset (from CSV or BEIR library) ---
    corpus: Dict[str, Dict[str, str]]
    queries: Dict[str, str]
    qrels: Dict[str, Dict[str, int]]

    print(f"Always loading BEIR dataset: {DATASET_NAME} (split: {split_type}) from {DATA_DIR} as per user request...")
    corpus, queries, qrels = GenericDataLoader(data_folder=DATA_DIR, prefix=None).load(split=split_type)
    print(f"Loaded {len(corpus)} documents, {len(queries)} queries, and qrels for {len(qrels)} queries for split '{split_type}' from {DATA_DIR}.")

    # Now, check if shared files exist. If not, save them.
    # This ensures that the shared CSVs are created once if they don't exist,
    # but subsequent runs (even if this script is run again) won't overwrite them
    # unless they are manually deleted. The primary loading is always from BEIR.

    if not os.path.exists(SHARED_CORPUS_CSV):
        print(f"Shared corpus CSV not found. Saving original corpus to: {SHARED_CORPUS_CSV}...")
        corpus_list = [{"_id": doc_id, "title": data.get("title", ""), "text": data.get("text", "")} for doc_id, data in corpus.items()]
        pd.DataFrame(corpus_list).to_csv(SHARED_CORPUS_CSV, index=False)
        print(f"Shared original corpus saved to: {SHARED_CORPUS_CSV}")
    else:
        print(f"Shared original corpus CSV already exists: {SHARED_CORPUS_CSV}. Skipping save.")

    if not os.path.exists(SHARED_QUERIES_CSV):
        print(f"Shared queries CSV not found. Saving original queries to: {SHARED_QUERIES_CSV}...")
        queries_list = [{"_id": q_id, "text": q_text} for q_id, q_text in queries.items()]
        pd.DataFrame(queries_list).to_csv(SHARED_QUERIES_CSV, index=False)
        print(f"Shared original queries saved to: {SHARED_QUERIES_CSV}")
    else:
        print(f"Shared original queries CSV already exists: {SHARED_QUERIES_CSV}. Skipping save.")

    # For qrels, the SHARED_QRELS_CSV path is already specific to the split_type
    if not os.path.exists(SHARED_QRELS_CSV):
        print(f"Shared qrels CSV for split '{split_type}' not found. Saving to: {SHARED_QRELS_CSV}...")
        qrels_list = []
        for q_id, doc_scores in qrels.items():
            for doc_id, score in doc_scores.items():
                qrels_list.append({"query-id": q_id, "corpus-id": doc_id, "score": score})
        pd.DataFrame(qrels_list).to_csv(SHARED_QRELS_CSV, index=False)
        print(f"Shared original qrels for split '{split_type}' saved to: {SHARED_QRELS_CSV}")
    else:
        print(f"Shared original qrels CSV for split '{split_type}' already exists: {SHARED_QRELS_CSV}. Skipping save.")

    # --- Save Original BEIR Data (JSON per run - kept for logging) ---
    # This part remains, saving to the run-specific directory
    print(f"Saving original BEIR data to {OUTPUT_DIR_METHOD}...")
    original_corpus_path = os.path.join(OUTPUT_DIR_METHOD, f"{METHOD_AND_SPLIT_PREFIX}_original_corpus.json")
    original_queries_path = os.path.join(OUTPUT_DIR_METHOD, f"{METHOD_AND_SPLIT_PREFIX}_original_queries.json")
    original_qrels_path = os.path.join(OUTPUT_DIR_METHOD, f"{METHOD_AND_SPLIT_PREFIX}_original_qrels.json")

    try:
        with open(original_corpus_path, 'w', encoding='utf-8') as f:
            json.dump(corpus, f, ensure_ascii=False, indent=4)
        print(f"Original corpus saved to: {original_corpus_path}")
    except Exception as e_corpus_save:
        print(f"Warning: Could not save original corpus: {e_corpus_save}")

    try:
        with open(original_queries_path, 'w', encoding='utf-8') as f:
            json.dump(queries, f, ensure_ascii=False, indent=4)
        print(f"Original queries saved to: {original_queries_path}")
    except Exception as e_queries_save:
        print(f"Warning: Could not save original queries: {e_queries_save}")

    try:
        with open(original_qrels_path, 'w', encoding='utf-8') as f:
            json.dump(qrels, f, ensure_ascii=False, indent=4)
        print(f"Original qrels saved to: {original_qrels_path}")
    except Exception as e_qrels_save:
        print(f"Warning: Could not save original qrels: {e_qrels_save}")
    # --- KẾT THÚC LƯU TRỮ DỮ LIỆU GỐC ---


    # --- THÊM: Xác định các tài liệu liên quan từ qrels ---
    print("Identifying relevant documents from qrels...")
    relevant_doc_ids = set()
    for qid, doc_scores in qrels.items():
        for doc_id, score in doc_scores.items():
            if score > 0:
                relevant_doc_ids.add(doc_id)
    print(f"Found {len(relevant_doc_ids)} unique relevant documents in qrels.")
    # ----------------------------------------------------

    # == Step 2: Chunk Passages ==
    chunking_needed = not os.path.exists(CHUNKS_OUTPUT_PATH) or \
                      not os.path.exists(DOC_CHUNK_MAP_PATH) or \
                      not os.path.exists(CHUNK_DOC_MAP_PATH)

    if chunking_needed:
        print(f"\nStarting Step 2: Chunking documents using '{selected_method}' method (Parallel)...")
        NUM_WORKERS = 2
        print(f"Using {NUM_WORKERS} worker processes.")

        chunk_to_doc_map = {}
        doc_to_chunk_map = {}
        processed_docs_count = 0
        total_chunks_created = 0

        corpus_to_process: Dict[str, Dict[str, str]] = {}
        if MAX_DOCS_TO_PROCESS:
            print(f"Prioritizing relevant documents for processing (limit: {MAX_DOCS_TO_PROCESS})...")
            relevant_docs_in_corpus = {doc_id: data for doc_id, data in corpus.items() if doc_id in relevant_doc_ids}
            print(f"Found {len(relevant_docs_in_corpus)} relevant documents present in the loaded corpus.")
            relevant_items = list(relevant_docs_in_corpus.items())
            num_relevant_to_take = min(len(relevant_items), MAX_DOCS_TO_PROCESS)
            for i in range(num_relevant_to_take):
                doc_id, data = relevant_items[i]
                corpus_to_process[doc_id] = data
            remaining_needed = MAX_DOCS_TO_PROCESS - len(corpus_to_process)
            if remaining_needed > 0:
                print(f"Adding {remaining_needed} more non-prioritized documents to reach the limit...")
                added_count = 0
                for doc_id, data in corpus.items():
                    if doc_id not in corpus_to_process:
                        corpus_to_process[doc_id] = data
                        added_count += 1
                        if added_count >= remaining_needed:
                            break
            print(f"Final corpus_to_process size: {len(corpus_to_process)}")
        else:
            print("Processing all documents in the corpus.")
            corpus_to_process = corpus

        # --- ADDED: Save the determined corpus_to_process and relevant/all queries to CSV for analysis ---
        print(f"\nSaving subset of data used for this run's processing to CSVs in {OUTPUT_DIR_METHOD}...")
        # Save corpus_to_process
        if corpus_to_process:
            corpus_to_process_list = [{"_id": doc_id, "title": data.get("title", ""), "text": data.get("text", "")} 
                                      for doc_id, data in corpus_to_process.items()]
            pd.DataFrame(corpus_to_process_list).to_csv(RUN_SPECIFIC_CORPUS_INPUT_CSV_PATH, index=False)
            print(f"Saved corpus for processing ({len(corpus_to_process_list)} docs) to: {RUN_SPECIFIC_CORPUS_INPUT_CSV_PATH}")
        else:
            print(f"Corpus for processing is empty. Skipping save to {RUN_SPECIFIC_CORPUS_INPUT_CSV_PATH}.")

        # Save relevant or all queries
        queries_to_save_dict: Dict[str, str] = {}
        if MAX_DOCS_TO_PROCESS and corpus_to_process: # Save relevant queries if corpus was subsetted
            print("Identifying queries relevant to the processed corpus subset...")
            if qrels and queries: # Ensure qrels and queries are available
                for q_id, doc_scores in qrels.items():
                    if q_id in queries: # Ensure query_id from qrels exists in the main queries dict
                        for doc_id_in_qrel, score in doc_scores.items():
                            if score > 0 and doc_id_in_qrel in corpus_to_process:
                                queries_to_save_dict[q_id] = queries[q_id]
                                break # Query added, move to next query_id in qrels
            if queries_to_save_dict:
                queries_to_save_list = [{"_id": q_id, "text": q_text} for q_id, q_text in queries_to_save_dict.items()]
                pd.DataFrame(queries_to_save_list).to_csv(RUN_SPECIFIC_QUERIES_INPUT_CSV_PATH, index=False)
                print(f"Saved relevant queries ({len(queries_to_save_list)}) to: {RUN_SPECIFIC_QUERIES_INPUT_CSV_PATH}")
            else:
                print(f"No queries found to be directly relevant to the processed corpus subset. {RUN_SPECIFIC_QUERIES_INPUT_CSV_PATH} will not be created or will be empty.")
                # Optionally, create an empty file or save all queries as a fallback if desired
                # For now, it just skips or leaves an empty/non-existent file.
        elif not MAX_DOCS_TO_PROCESS and queries: # Save all queries if full corpus is used
            print("Saving all queries (since full corpus is processed)...")
            all_queries_list = [{"_id": q_id, "text": q_text} for q_id, q_text in queries.items()]
            if all_queries_list:
                pd.DataFrame(all_queries_list).to_csv(RUN_SPECIFIC_QUERIES_INPUT_CSV_PATH, index=False)
                print(f"Saved all queries ({len(all_queries_list)}) to: {RUN_SPECIFIC_QUERIES_INPUT_CSV_PATH}")
            else:
                print(f"No queries available to save. {RUN_SPECIFIC_QUERIES_INPUT_CSV_PATH} will not be created.")
        else:
            print(f"Queries list is empty or corpus_to_process is empty. Skipping save to {RUN_SPECIFIC_QUERIES_INPUT_CSV_PATH}.")
        # --- END ADDED SECTION ---

        corpus_items = list(corpus_to_process.items())
        total_docs_to_process = len(corpus_items)

        if total_docs_to_process == 0:
            print("Warning: No documents selected for processing. Skipping chunking.")
        else:
            worker_func_partial = functools.partial(
                chunk_document_worker,
                chunking_func=selected_chunking_func,
                params=method_params
            )

            print(f"Starting parallel chunking of {total_docs_to_process} documents...")
            try:
                # Mở file chunks output trước
                with open(CHUNKS_OUTPUT_PATH, 'w', encoding='utf-8') as f_chunks_out, \
                     ProcessPoolExecutor( # Corrected: No need for concurrent.futures here as it's imported
                        max_workers=NUM_WORKERS,
                        initializer=init_worker_directml # <<< THÊM INITIALIZER
                    ) as executor:

                    # Sử dụng executor.map
                    results = list(tqdm(executor.map(worker_func_partial, corpus_items),
                                        total=total_docs_to_process,
                                        desc=f"Chunking Corpus ({selected_method})"))

                    # Xử lý kết quả
                    print("\nProcessing results from workers...")
                    for doc_id, doc_chunks_list in tqdm(results, desc="Aggregating results"): # Đổi tên biến để rõ ràng hơn
                        if doc_chunks_list: # Kiểm tra xem danh sách chunk có dữ liệu không
                            doc_to_chunk_map[doc_id] = []
                            for chunk_tuple in doc_chunks_list: # Lặp qua từng tuple chunk
                                if len(chunk_tuple) == 3:
                                    chunk_id, chunk_text, oie_str = chunk_tuple
                                elif len(chunk_tuple) == 2: # Trường hợp dự phòng nếu có lúc chỉ trả về 2
                                    chunk_id, chunk_text = chunk_tuple
                                    oie_str = None
                                else:
                                    print(f"Warning: Skipping malformed chunk tuple for doc {doc_id}: {chunk_tuple}")
                                    continue

                                chunk_to_doc_map[chunk_id] = doc_id
                                doc_to_chunk_map[doc_id].append(chunk_id)
                                
                                chunk_json_obj = {'chunk_id': chunk_id, 'text': chunk_text}
                                # Chỉ thêm key 'oie_triples' nếu oie_str có giá trị (không None và không rỗng)
                                # và nếu INCLUDE_OIE là True cho lần chạy này.
                                # Tuy nhiên, oie_str đã được quyết định bởi hàm chunking dựa trên params['include_oie']
                                # nên chỉ cần kiểm tra oie_str là đủ.
                                if oie_str: 
                                    chunk_json_obj['oie_triples_str'] = oie_str # Lưu dưới dạng chuỗi đã format

                                f_chunks_out.write(json.dumps(chunk_json_obj) + '\n')
                                total_chunks_created += 1
                        processed_docs_count += 1

            except Exception as e:
                print(f"\nAn error occurred during parallel chunking: {e}")
                import traceback
                traceback.print_exc()
                exit()

            print(f"\nFinished parallel chunking.")
            print(f"Documents processed (attempted): {processed_docs_count}")
            print(f"Total chunks created: {total_chunks_created}")
            print(f"Chunks saved to: {CHUNKS_OUTPUT_PATH}")

            print("Saving mapping files...")
            try:
                with open(CHUNK_DOC_MAP_PATH, 'w', encoding='utf-8') as f: json.dump(chunk_to_doc_map, f)
                print(f"Chunk -> Doc map saved to: {CHUNK_DOC_MAP_PATH}")
                with open(DOC_CHUNK_MAP_PATH, 'w', encoding='utf-8') as f: json.dump(doc_to_chunk_map, f)
                print(f"Doc -> Chunk map saved to: {DOC_CHUNK_MAP_PATH}")
            except Exception as e: print(f"Error saving mapping files: {e}")

    else:
        print(f"Chunk files for method '{selected_method}' already exist. Skipping Step 2.")
        print(f"Loading existing Doc -> Chunk map from {DOC_CHUNK_MAP_PATH}...")
        try:
            with open(DOC_CHUNK_MAP_PATH, 'r', encoding='utf-8') as f:
                doc_to_chunk_map = json.load(f)
            print(f"Loaded map for {len(doc_to_chunk_map)} documents.")
        except FileNotFoundError:
            print(f"Error: Map file {DOC_CHUNK_MAP_PATH} not found. Please run chunking again.")
            exit()
        except Exception as e:
            print(f"Error loading map file: {e}")
            exit()

    # == Step 3: Generate Triplets ==
    print(f"\nStarting Step 3: Creating Triplets for MatchZoo (Split: {split_type})...")
    # 1. Queries
    print(f"Using {len(queries)} queries loaded from BEIR.")
    # 2. Qrels -> query_positive_docs
    print("Processing qrels from BEIR...")
    query_positive_docs = {}
    for qid, doc_scores in qrels.items():
        positive_docs = {doc_id for doc_id, score in doc_scores.items() if score > 0}
        if positive_docs: query_positive_docs[qid] = positive_docs
    print(f"Processed relevance information for {len(query_positive_docs)} queries.")

    # 3. Load Chunks Data
    print(f"Loading chunks data from {CHUNKS_OUTPUT_PATH}...")
    chunks_data = {} # Lưu trữ {chunk_id: {'text': chunk_text, 'oie_triples_str': oie_string_or_None}}
    try:
        with open(CHUNKS_OUTPUT_PATH, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="Loading Chunks"):
                try:
                    chunk_info = json.loads(line)
                    # Lưu cả text và oie_triples_str (nếu có)
                    chunks_data[chunk_info['chunk_id']] = {
                        'text': chunk_info.get('text', ''),
                        'oie_triples_str': chunk_info.get('oie_triples_str') # Sẽ là None nếu không có
                    }
                except json.JSONDecodeError as e:
                    continue
        print(f"Loaded data for {len(chunks_data)} chunks.")
    except FileNotFoundError:
        print(f"Error: Chunks file not found at {CHUNKS_OUTPUT_PATH}")
        exit()
    except Exception as e:
        print(f"An error occurred loading chunks data: {e}")
        exit()

    all_chunk_ids = list(chunks_data.keys())
    if not all_chunk_ids:
        print("Error: No chunks loaded from file. Cannot generate triplets.")
        exit()

    # 4. Map qid -> set(positive_chunk_id)
    print("Mapping queries to positive chunk IDs...")
    query_positive_chunks = {} # <-- SỬA LỖI: Khởi tạo dictionary Ở ĐÂY, bên ngoài vòng lặp
    found_positive_mappings = 0
    for qid, pos_doc_ids in tqdm(query_positive_docs.items(), desc="Mapping Positive Chunks"):
        # query_positive_chunks[qid] = set() # Khởi tạo set cho mỗi qid có trong qrels
        current_positive_chunks = set()
        for doc_id in pos_doc_ids:
            # Kiểm tra xem doc_id có trong map được tạo/tải từ Step 2 không
            if doc_id in doc_to_chunk_map:
                # Lấy danh sách chunk_id cho doc_id đó
                chunk_ids_for_doc = doc_to_chunk_map.get(doc_id, []) # Dùng .get để an toàn
                for chunk_id in chunk_ids_for_doc:
                    # Kiểm tra xem chunk_id có thực sự tồn tại trong dữ liệu chunks đã tải không
                    if chunk_id in chunks_data:
                        current_positive_chunks.add(chunk_id)
        # Chỉ thêm vào dict nếu tìm thấy chunk dương nào đó
        if current_positive_chunks:
            query_positive_chunks[qid] = current_positive_chunks
            found_positive_mappings += 1

    print(f"Found positive chunk mappings for {found_positive_mappings} queries.")
    if found_positive_mappings == 0:
        print("Warning: No positive chunks could be mapped to any query based on the processed documents and qrels. No triplets will be generated.")
        # Có thể dừng ở đây nếu muốn
        # exit()

    # 5. Generate Triplets
    print(f"Generating all possible triplets...")
    
    all_generated_triplets = [] # <-- THU THẬP TẤT CẢ TRIPLETS VÀO ĐÂY
    
    query_ids_for_triplet_generation = list(queries.keys())
    random.seed(RANDOM_SEED) # Seed cho việc shuffle query_ids nếu muốn, hoặc cho các random khác
    random.shuffle(query_ids_for_triplet_generation)

    for qid in tqdm(query_ids_for_triplet_generation, desc=f"Processing Queries for Triplets ({split_type})"):
        if qid not in queries or qid not in query_positive_chunks:
            continue

        query_text = clean_text(queries[qid])
        positive_chunk_ids_for_query = query_positive_chunks[qid]

        for positive_chunk_id in positive_chunk_ids_for_query:
            positive_chunk_content = chunks_data.get(positive_chunk_id)
            if not positive_chunk_content or not positive_chunk_content.get('text'): continue
            
            # Lấy text và oie_str cho positive chunk
            positive_text_only = positive_chunk_content['text']
            positive_oie_str = positive_chunk_content.get('oie_triples_str', '') # Mặc định là chuỗi rỗng nếu không có
            
            # Nối OIE vào text nếu có (đảm bảo có khoảng trắng nếu cả hai đều tồn tại)
            # Chuỗi OIE đã có dạng " [OIE_TRIPLES] ... [/OIE_TRIPLES]" hoặc rỗng
            final_positive_chunk_text_with_oie = positive_text_only.strip() + (f" {positive_oie_str.strip()}" if positive_oie_str and positive_oie_str.strip() else "")


            negative_chunk_id = None
            potential_negative_pool = [cid for cid in all_chunk_ids if cid not in positive_chunk_ids_for_query]
            
            if not potential_negative_pool: 
                continue

            negative_chunk_id = random.choice(potential_negative_pool)

            if negative_chunk_id:
                negative_chunk_content = chunks_data.get(negative_chunk_id)
                if negative_chunk_content and negative_chunk_content.get('text'):
                    negative_text_only = negative_chunk_content['text']
                    negative_oie_str = negative_chunk_content.get('oie_triples_str', '')

                    final_negative_chunk_text_with_oie = negative_text_only.strip() + (f" {negative_oie_str.strip()}" if negative_oie_str and negative_oie_str.strip() else "")
                    
                    triplet_line = f"{query_text}\t{clean_text(final_positive_chunk_text_with_oie)}\t{clean_text(final_negative_chunk_text_with_oie)}\n"
                    all_generated_triplets.append(triplet_line)

    print(f"\nGenerated a total of {len(all_generated_triplets)} possible triplets.")

    # Lấy mẫu ngẫu nhiên nếu cần
    triplets_to_write = []
    if MAX_TRIPLETS_TO_GENERATE is not None and len(all_generated_triplets) > MAX_TRIPLETS_TO_GENERATE:
        print(f"Sampling {MAX_TRIPLETS_TO_GENERATE} triplets randomly using seed {RANDOM_SEED}...")
        random.seed(RANDOM_SEED) # Đặt lại seed ngay trước khi sampling để đảm bảo tính nhất quán
        triplets_to_write = random.sample(all_generated_triplets, MAX_TRIPLETS_TO_GENERATE)
    elif len(all_generated_triplets) > 0 : # Nếu không có giới hạn, hoặc giới hạn không bị vượt quá
        triplets_to_write = all_generated_triplets
        if MAX_TRIPLETS_TO_GENERATE is not None:
             print(f"Number of generated triplets ({len(all_generated_triplets)}) is within or equal to the limit ({MAX_TRIPLETS_TO_GENERATE}). Using all generated triplets.")
        else:
             print("No limit set for triplets. Using all generated triplets.")
    else:
        print("No triplets were generated. Output file will be empty.")


    # Ghi các triplets đã chọn vào file
    print(f"Writing {len(triplets_to_write)} triplets to {OUTPUT_TRIPLETS_PATH}...")
    try:
        with open(OUTPUT_TRIPLETS_PATH, 'w', encoding='utf-8') as f_out:
            for triplet_line in tqdm(triplets_to_write, desc="Writing Triplets to File"):
                f_out.write(triplet_line)
    except Exception as e:
        print(f"An error occurred writing triplets to file: {e}")
        import traceback
        traceback.print_exc()

    triplets_count = len(triplets_to_write) # Cập nhật số lượng triplets thực tế đã ghi
    print(f"Finished generating and writing {triplets_count} triplets.")
    # <<< SỬA ĐỔI: Sử dụng OUTPUT_TRIPLETS_PATH >>>
    print(f"Output triplets saved to: {OUTPUT_TRIPLETS_PATH}")
    print(f"\n--- Steps 1, 2 and 3 (Chunking: {selected_method}, Triplet Generation for Split: {split_type}) Completed ---")
    # <<< SỬA ĐỔI: Sử dụng OUTPUT_TRIPLETS_PATH >>>
    print(f"Output file for MatchZoo processing: {OUTPUT_TRIPLETS_PATH}")
    if split_type == 'train':
        print("Next step: Use this TSV file to fine-tune your model using MatchZoo (Step 4).")
    else: # test
        print("Next step: Use this TSV file (or the generated chunks/maps) for evaluating your model.")

    # == Step 4: Transform Triplets and Split for MatchZoo ==
    print(f"\\nStarting Step 4: Transforming triplets and splitting for MatchZoo...")
    if not os.path.exists(OUTPUT_TRIPLETS_PATH):
        print(f"Error: Triplets file not found at {OUTPUT_TRIPLETS_PATH}. Cannot proceed with MatchZoo data preparation.")
    else:
        try:
            print(f"Reading triplets from: {OUTPUT_TRIPLETS_PATH}")
            triplets_df = pd.read_csv(OUTPUT_TRIPLETS_PATH, sep='\\t', header=None, names=["query", "pos_chunk", "neg_chunk"], engine='python')
            
            if triplets_df.empty:
                print("Triplets file is empty. Skipping MatchZoo data preparation.")
            else:
                print(f"Transforming {len(triplets_df)} triplets to MatchZoo format...")
                matchzoo_data = []
                for _, row in tqdm(triplets_df.iterrows(), total=triplets_df.shape[0], desc="Transforming to MatchZoo"): # ADDED tqdm
                    # Ensure no NaN values are passed, replace with empty string if necessary
                    query = str(row['query']) if pd.notna(row['query']) else ""
                    pos_chunk = str(row['pos_chunk']) if pd.notna(row['pos_chunk']) else ""
                    neg_chunk = str(row['neg_chunk']) if pd.notna(row['neg_chunk']) else ""

                    matchzoo_data.append({'label': 1, 'text_left': query, 'text_right': pos_chunk})
                    matchzoo_data.append({'label': 0, 'text_left': query, 'text_right': neg_chunk})
                
                matchzoo_df = pd.DataFrame(matchzoo_data)
                # Shuffle the DataFrame to mix positive and negative samples
                matchzoo_df = matchzoo_df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True) # ADDED shuffle
                print(f"Created and shuffled {len(matchzoo_df)} MatchZoo formatted entries.")

                print(f"Splitting data into training and development sets (90/10 split, random_state={RANDOM_SEED})...")
                # shuffle=True is required for stratify. The previous shuffle ensures randomness, this maintains it with stratification.
                train_df, dev_df = train_test_split(matchzoo_df, test_size=0.1, random_state=RANDOM_SEED, shuffle=True, stratify=matchzoo_df['label'] if 'label' in matchzoo_df else None)

                print(f"Saving MatchZoo training data ({len(train_df)} entries) to: {TRAIN_MZ_TSV_PATH}")
                train_df.to_csv(TRAIN_MZ_TSV_PATH, sep='\t', index=False, header=False, columns=['text_left', 'text_right', 'label']) # CHANGED column order

                print(f"Saving MatchZoo development data ({len(dev_df)} entries) to: {DEV_MZ_TSV_PATH}")
                dev_df.to_csv(DEV_MZ_TSV_PATH, sep='\t', index=False, header=False, columns=['text_left', 'text_right', 'label']) # CHANGED column order
                
                print("MatchZoo data preparation completed.")
        except pd.errors.EmptyDataError:
            print(f"Warning: The triplets file {OUTPUT_TRIPLETS_PATH} is empty or not a valid CSV. Skipping MatchZoo data preparation.")
        except Exception as e:
            print(f"An error occurred during MatchZoo data preparation: {e}")
            traceback.print_exc()

    print("\n--- All Processing Steps Completed ---") # ADDED final message
    # Call close_oie_clients() explicitly here as a safeguard, though atexit should handle it.
    # It's good practice if there are any non-standard exit paths or interruptions.
    print("Ensuring OIE clients are closed...")
    close_oie_clients() 
    print("Exiting script.")