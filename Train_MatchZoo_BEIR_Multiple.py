import pandas as pd
import nltk
import os
import re
from tqdm.auto import tqdm
import random
import json
import hashlib
from beir import util
from beir.datasets.data_loader import GenericDataLoader
import numpy as np
from typing import List, Tuple, Dict, Callable, Optional, Union
import concurrent.futures
import functools
from Semantic_Grouping import semantic_chunk_passage_from_grouping_logic
from Semantic_Splitter import chunk_passage_semantic_splitter
from Text_Splitter import chunk_passage_text_splitter    

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
    max_triplets = get_int_input("Max Triplets (0 for unlimited)", default=100000, required=True, min_val=0)
    params['max_triplets'] = max_triplets if max_triplets > 0 else None
    max_docs = get_int_input("Max Documents to Process (0 or empty for all)", default=0, required=False, min_val=0)
    params['max_docs'] = max_docs if max_docs is not None and max_docs > 0 else None
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
def chunk_document_worker(doc_item: Tuple[str, Dict], chunking_func: Callable, params: Dict) -> Tuple[str, List[Tuple[str, str]]]:
    doc_id, doc_data = doc_item
    try:
        print(f"Processing document: {doc_id}")  # Thêm log
        passage_text = doc_data.get("text", "")
        title = doc_data.get("title", "")
        full_text_to_chunk = f"{title}. {passage_text}" if title else passage_text

        if not full_text_to_chunk:
            print(f"Document {doc_id} is empty.")  # Thêm log
            return doc_id, []

        doc_chunks = chunking_func(doc_id, full_text_to_chunk, **params)
        print(f"Document {doc_id} processed successfully.")  # Thêm log
        return doc_id, doc_chunks if doc_chunks else []
    except Exception as e:
        print(f"Error processing document {doc_id}: {e}")  # Thêm log
        return doc_id, []

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

# --- ĐẶT TOÀN BỘ LOGIC CHÍNH VÀO ĐÂY ---
if __name__ == "__main__":

    # --- Get Configuration from User ---
    common_config = get_common_params()
    selected_method = select_chunking_method()
    split_type = get_split_type() # <<< Giữ nguyên việc lấy split_type

    method_params = {}
    if selected_method == 'semantic_grouping':
        method_params = get_semantic_grouping_params(common_config['embedding_model'])
    elif selected_method == 'semantic_splitter':
        method_params = get_semantic_splitter_params(common_config['embedding_model'])
    elif selected_method == 'text_splitter':
        method_params = get_text_splitter_params()
        method_params['model_name'] = common_config['embedding_model'] # Add model_name for consistency

    # --- Combine Configurations ---
    DATASET_NAME = common_config['dataset_name']
    # <<< SỬA ĐỔI: Xác định thư mục cơ sở cho phương thức >>>
    BASE_METHOD_DIR = os.path.join(common_config['output_dir'], f"{DATASET_NAME}_{selected_method.replace('_', '-')}")

    # <<< SỬA ĐỔI: Tìm thư mục chạy tiếp theo và đặt OUTPUT_DIR_METHOD >>>
    OUTPUT_DIR_METHOD = find_next_run_dir(BASE_METHOD_DIR, split_type)

    # Các đường dẫn file output sẽ tự động nằm trong thư mục con có đánh số
    # <<< SỬA ĐỔI: Đảm bảo tên file không lặp lại split_type >>>
    METHOD_AND_SPLIT_PREFIX = f"{DATASET_NAME}_{selected_method.replace('_', '-')}_{split_type}" # Ví dụ: msmarco_semantic-grouping_test
    CHUNKS_OUTPUT_PATH = os.path.join(OUTPUT_DIR_METHOD, f"{METHOD_AND_SPLIT_PREFIX}_chunks.jsonl")
    CHUNK_DOC_MAP_PATH = os.path.join(OUTPUT_DIR_METHOD, f"{METHOD_AND_SPLIT_PREFIX}_chunk_doc_map.json")
    DOC_CHUNK_MAP_PATH = os.path.join(OUTPUT_DIR_METHOD, f"{METHOD_AND_SPLIT_PREFIX}_doc_chunk_map.json")
    OUTPUT_TRIPLETS_PATH = os.path.join(OUTPUT_DIR_METHOD, f"{METHOD_AND_SPLIT_PREFIX}_triplets.tsv")

    EMBEDDING_MODEL_NAME = common_config['embedding_model']
    MAX_TRIPLETS_TO_GENERATE = common_config['max_triplets']
    MAX_DOCS_TO_PROCESS = common_config['max_docs']

    # --- Map chunking method name to function ---
    chunking_functions: Dict[str, Callable] = {
        'semantic_grouping': semantic_chunk_passage_from_grouping_logic,
        'semantic_splitter': chunk_passage_semantic_splitter,
        'text_splitter': chunk_passage_text_splitter,
    }
    selected_chunking_func = chunking_functions.get(selected_method)
    if selected_chunking_func is None or not callable(selected_chunking_func):
        print(f"Error: Chunking function for '{selected_method}' not available/callable.")
        exit()

    # --- Print Final Configuration ---
    print(f"\n--- Final Configuration ---")
    print(f"Dataset: {DATASET_NAME}")
    print(f"Processing Split: {split_type}")
    # <<< SỬA ĐỔI: LOCAL_DATA_PATH có thể cần điều chỉnh nếu bạn muốn nó tách biệt theo dataset >>>
    # Hiện tại nó đang trỏ đến ./Data/msmarco/msmarco dựa trên default input
    # Nếu bạn muốn nó chỉ là ./Data/msmarco, hãy sửa default trong get_common_params
    LOCAL_DATA_PATH = common_config['data_dir'] # Sử dụng trực tiếp data_dir đã nhập
    print(f"BEIR Data Path: {LOCAL_DATA_PATH}")
    print(f"Output Directory: {OUTPUT_DIR_METHOD}") # Đường dẫn này giờ đã bao gồm split và số thứ tự
    print(f"Chunking Method: {selected_method}")
    print(f"Embedding Model: {EMBEDDING_MODEL_NAME}")
    print(f"Max Triplets: {MAX_TRIPLETS_TO_GENERATE if MAX_TRIPLETS_TO_GENERATE else 'Unlimited'}")
    print(f"Max Documents to Process: {MAX_DOCS_TO_PROCESS if MAX_DOCS_TO_PROCESS else 'All'}")
    print(f"Chunking Parameters: {method_params}")
    print(f"---------------------------\n")

    # --- Download NLTK data ---
    try:
        nltk.data.find('tokenizers/punkt')
        print("NLTK 'punkt' resource found.") # Optional confirmation
    except LookupError:
        print("Downloading NLTK 'punkt' tokenizer...")
        nltk.download('punkt', quiet=True)

    # <<< THÊM KHỐI NÀY ĐỂ TẢI punkt_tab >>>
    try:
        nltk.data.find('tokenizers/punkt_tab')
        print("NLTK 'punkt_tab' resource found.") # Optional confirmation
    except LookupError:
        print("Downloading NLTK 'punkt_tab' resource...")
        nltk.download('punkt_tab', quiet=True)
    # <<< KẾT THÚC KHỐI THÊM >>>

    # --- Main Processing ---
    # <<< SỬA ĐỔI: Tạo thư mục output cuối cùng (đã bao gồm số thứ tự) >>>
    os.makedirs(OUTPUT_DIR_METHOD, exist_ok=True)
    # <<< SỬA ĐỔI: Đảm bảo thư mục dữ liệu BEIR gốc tồn tại (nơi chứa thư mục con dataset) >>>
    # Ví dụ: Nếu data_dir là './Data/msmarco/msmarco', thì thư mục gốc là './Data/msmarco'
    beir_base_data_dir = os.path.dirname(common_config['data_dir'])
    os.makedirs(beir_base_data_dir, exist_ok=True)

    # == Step 1: Load BEIR Data ==
    # <<< SỬA ĐỔI: dataset_specific_path là đường dẫn người dùng nhập >>>
    dataset_specific_path = common_config['data_dir'] # Ví dụ: './Data/msmarco/msmarco'
    print(f"\nLoading BEIR dataset: {DATASET_NAME} (split: {split_type}) from {dataset_specific_path}...")

    try:
        # <<< SỬA ĐỔI: Sử dụng dataset_specific_path và bỏ prefix >>>
        corpus, queries, qrels = GenericDataLoader(data_folder=dataset_specific_path).load(split=split_type)
        print(f"Loaded {len(corpus)} documents, {len(queries)} queries, and qrels for {len(qrels)} queries for split '{split_type}' from {dataset_specific_path}.")
    except Exception as e:
        # <<< SỬA ĐỔI: Thông báo lỗi bao gồm đường dẫn đúng >>>
        print(f"Error loading BEIR dataset '{DATASET_NAME}' (split: {split_type}) from {dataset_specific_path}: {e}")
        # <<< SỬA ĐỔI: Tải vào thư mục cha của dataset_specific_path >>>
        download_target_dir = os.path.dirname(dataset_specific_path) # Ví dụ: './Data/msmarco'
        print(f"Attempting to download using 'util.download_and_unzip' into '{download_target_dir}'...")
        try:
            url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{DATASET_NAME}.zip"
            # Tải vào thư mục cha, download_path sẽ là thư mục con được giải nén (dataset_specific_path)
            download_path = util.download_and_unzip(url, download_target_dir)
            print(f"Dataset downloaded and unzipped to: {download_path}") # download_path nên là dataset_specific_path

            # <<< SỬA ĐỔI: Sử dụng download_path (chính là dataset_specific_path) và bỏ prefix >>>
            corpus, queries, qrels = GenericDataLoader(data_folder=download_path).load(split=split_type)
            print(f"Successfully loaded {len(corpus)} documents, {len(queries)} queries, and qrels for {len(qrels)} queries for split '{split_type}' from {download_path}.")
        except Exception as download_err:
            print(f"Failed to download or load the dataset after download attempt: {download_err}")
            # <<< SỬA ĐỔI: Thông báo lỗi bao gồm đường dẫn đúng >>>
            print(f"Please manually check the directory structure. Ensure '{dataset_specific_path}' contains corpus.jsonl, queries.jsonl, and qrels/{split_type}.tsv.")
            exit()

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

        # --- CẤU HÌNH XỬ LÝ SONG SONG ---
        NUM_WORKERS = 2 # **ĐIỀU CHỈNH CẨN THẬN DỰA TRÊN RAM!**
        print(f"Using {NUM_WORKERS} worker processes.")
        # ---------------------------------

        chunk_to_doc_map = {}
        doc_to_chunk_map = {}
        processed_docs_count = 0
        total_chunks_created = 0

        # --- SỬA ĐỔI: Tạo corpus_to_process ưu tiên tài liệu liên quan ---
        if MAX_DOCS_TO_PROCESS:
            print(f"Prioritizing relevant documents for processing (limit: {MAX_DOCS_TO_PROCESS})...")
            corpus_to_process = {}
            relevant_docs_in_corpus = {doc_id: data for doc_id, data in corpus.items() if doc_id in relevant_doc_ids}
            print(f"Found {len(relevant_docs_in_corpus)} relevant documents present in the loaded corpus.")

            # Lấy các tài liệu liên quan trước, tối đa MAX_DOCS_TO_PROCESS
            relevant_items = list(relevant_docs_in_corpus.items())
            num_relevant_to_take = min(len(relevant_items), MAX_DOCS_TO_PROCESS)
            for i in range(num_relevant_to_take):
                doc_id, data = relevant_items[i]
                corpus_to_process[doc_id] = data

            # Nếu chưa đủ MAX_DOCS_TO_PROCESS, bổ sung bằng các tài liệu khác
            remaining_needed = MAX_DOCS_TO_PROCESS - len(corpus_to_process)
            if remaining_needed > 0:
                print(f"Adding {remaining_needed} more non-prioritized documents to reach the limit...")
                added_count = 0
                for doc_id, data in corpus.items():
                    if doc_id not in corpus_to_process: # Chỉ thêm nếu chưa có
                        corpus_to_process[doc_id] = data
                        added_count += 1
                        if added_count >= remaining_needed:
                            break
            print(f"Final corpus_to_process size: {len(corpus_to_process)}")

        else: # Nếu không giới hạn, xử lý tất cả
            print("Processing all documents in the corpus.")
            corpus_to_process = corpus
        # ----------------------------------------------------------------

        corpus_items = list(corpus_to_process.items())
        total_docs_to_process = len(corpus_items)

        if total_docs_to_process == 0:
            print("Warning: No documents selected for processing. Exiting Step 2.")
        else:
            # Chuẩn bị sẵn các tham số cố định cho worker
            worker_func_partial = functools.partial(
                chunk_document_worker,
                chunking_func=selected_chunking_func,
                params=method_params
            )

            print(f"Starting parallel chunking of {total_docs_to_process} documents...")
            try:
                # Mở file chunks output trước
                with open(CHUNKS_OUTPUT_PATH, 'w', encoding='utf-8') as f_chunks_out, \
                     concurrent.futures.ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:

                    # Sử dụng executor.map
                    results = list(tqdm(executor.map(worker_func_partial, corpus_items),
                                        total=total_docs_to_process,
                                        desc=f"Chunking Corpus ({selected_method})"))

                    # Xử lý kết quả
                    print("\nProcessing results from workers...")
                    for doc_id, doc_chunks in tqdm(results, desc="Aggregating results"):
                        if doc_chunks:
                            doc_to_chunk_map[doc_id] = []
                            for chunk_id, chunk_text in doc_chunks:
                                chunk_to_doc_map[chunk_id] = doc_id # Sửa lỗi gõ: chunk_to_doc_map
                                doc_to_chunk_map[doc_id].append(chunk_id)
                                f_chunks_out.write(json.dumps({'chunk_id': chunk_id, 'text': chunk_text}) + '\n')
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
    chunks_data = {}
    try:
        with open(CHUNKS_OUTPUT_PATH, 'r', encoding='utf-8') as f:
            # Thêm tqdm vào đây nếu file chunks lớn
            for line in tqdm(f, desc="Loading Chunks"):
                try:
                    chunk_info = json.loads(line)
                    # Chỉ lưu text nếu cần, hoặc lưu cả object nếu cần thêm thông tin
                    chunks_data[chunk_info['chunk_id']] = chunk_info['text']
                except json.JSONDecodeError as e:
                    # print(f"Skipping invalid JSON line: {e}") # Bỏ comment nếu muốn xem lỗi JSON
                    continue
        print(f"Loaded text for {len(chunks_data)} chunks.")
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
    # <<< SỬA ĐỔI: Sử dụng OUTPUT_TRIPLETS_PATH >>>
    print(f"Generating triplets and saving to {OUTPUT_TRIPLETS_PATH}...")
    triplets_count = 0
    try:
        # <<< SỬA ĐỔI: Sử dụng OUTPUT_TRIPLETS_PATH >>>
        with open(OUTPUT_TRIPLETS_PATH, 'w', encoding='utf-8') as f_out:
            query_ids = list(queries.keys())
            random.shuffle(query_ids)

            for qid in tqdm(query_ids, desc=f"Generating Triplets ({split_type})"):
                # Kiểm tra xem qid có query_text, có trong mapping và có chunk dương không
                if qid not in queries or qid not in query_positive_chunks: # Đã kiểm tra set rỗng ở bước 4
                    continue

                query_text = clean_text(queries[qid])
                positive_chunk_ids_for_query = query_positive_chunks[qid]

                # Lặp qua từng chunk dương cho query này
                for positive_chunk_id in positive_chunk_ids_for_query:
                    positive_chunk_text = chunks_data.get(positive_chunk_id)
                    if not positive_chunk_text: continue # Bỏ qua nếu không tìm thấy text (dù không nên xảy ra)

                    # Lấy mẫu negative chunk
                    negative_chunk_id = None
                    attempts = 0
                    max_attempts = len(all_chunk_ids) # Giới hạn số lần thử để tránh vòng lặp vô hạn
                    while attempts < max_attempts:
                        potential_negative_id = random.choice(all_chunk_ids)
                        # Đảm bảo negative không nằm trong tập positive của query HIỆN TẠI
                        if potential_negative_id not in positive_chunk_ids_for_query:
                            negative_chunk_id = potential_negative_id
                            break
                        attempts += 1

                    # Nếu tìm được negative hợp lệ
                    if negative_chunk_id:
                        negative_chunk_text = chunks_data.get(negative_chunk_id)
                        if negative_chunk_text: # Đảm bảo negative chunk có text
                            # <<< SỬA ĐỔI: Ghi vào f_out >>>
                            f_out.write(f"{query_text}\t{clean_text(positive_chunk_text)}\t{clean_text(negative_chunk_text)}\n")
                            triplets_count += 1
                            # Kiểm tra giới hạn triplets
                            if MAX_TRIPLETS_TO_GENERATE is not None and triplets_count >= MAX_TRIPLETS_TO_GENERATE:
                                print(f"\nReached limit of {MAX_TRIPLETS_TO_GENERATE} triplets.")
                                raise StopIteration # Dùng exception để thoát khỏi các vòng lặp lồng nhau

                # Kiểm tra lại giới hạn sau khi xử lý hết positive chunks cho 1 query
                if MAX_TRIPLETS_TO_GENERATE is not None and triplets_count >= MAX_TRIPLETS_TO_GENERATE:
                    break # Thoát vòng lặp duyệt query

    except StopIteration:
        pass
    except Exception as e:
        print(f"An error occurred generating triplets: {e}")
        import traceback
        traceback.print_exc()

    print(f"Finished generating {triplets_count} triplets.")
    # <<< SỬA ĐỔI: Sử dụng OUTPUT_TRIPLETS_PATH >>>
    print(f"Output triplets saved to: {OUTPUT_TRIPLETS_PATH}")
    print(f"\n--- Steps 1, 2 and 3 (Chunking: {selected_method}, Triplet Generation for Split: {split_type}) Completed ---")
    # <<< SỬA ĐỔI: Sử dụng OUTPUT_TRIPLETS_PATH >>>
    print(f"Output file for MatchZoo processing: {OUTPUT_TRIPLETS_PATH}")
    if split_type == 'train':
        print("Next step: Use this TSV file to fine-tune your model using MatchZoo (Step 4).")
    else: # test
        print("Next step: Use this TSV file (or the generated chunks/maps) for evaluating your model.")
