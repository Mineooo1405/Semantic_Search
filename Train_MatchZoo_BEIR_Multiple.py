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
import traceback
import atexit # <<< THÊM IMPORT NÀY
from Semantic_Grouping import semantic_chunk_passage_from_grouping_logic
from Semantic_Splitter import chunk_passage_semantic_splitter
from Text_Splitter import chunk_passage_text_splitter    
# Sửa đổi import nếu OIE.py đã tích hợp caching vào extract_triples
from Tool.OIE import extract_triples, close_oie_clients # <<< THÊM close_oie_clients
import torch # Thêm import này ở đầu file nếu chưa có
import torch_directml # Thêm import này ở đầu file nếu chưa có

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
def init_worker():
    """Initializer for each worker process."""
    global dml_device_worker
    try:
        # Cố gắng import và lấy thiết bị trong worker
        # Điều này đảm bảo mỗi worker nhận biết được DirectML
        import torch
        import torch_directml
        dml_device_worker = torch_directml.device()
        print(f"Worker {os.getpid()} initialized with DML device: {dml_device_worker}")
    except Exception as e:
        print(f"Worker {os.getpid()} failed to initialize DML device: {e}")
        dml_device_worker = None # Hoặc đặt là 'cpu'

def init_worker():
    """Initializer for each worker process."""
    # global dml_device_worker # Không cần thiết nếu không sử dụng trực tiếp biến global này
    try:
        import torch # Đảm bảo import trong worker
        import torch_directml # Đảm bảo import trong worker
        dml_device = torch_directml.device()
        # print(f"Worker {os.getpid()} initialized with DML device: {dml_device}") # Bỏ comment nếu cần debug
    except Exception as e:
        print(f"Worker {os.getpid()} failed to initialize DML device: {e}")
        # dml_device_worker = None # Không cần gán vào biến global

def chunk_document_worker(
    doc_item: Tuple[str, Dict],
    chunking_func: Callable,
    params: Dict
) -> Tuple[str, List[Tuple[str, str, Optional[str]]]]: # Đảm bảo kiểu trả về nhất quán
    doc_id, doc_data = doc_item
    try:
        passage_text = doc_data.get("text", "")
        title = doc_data.get("title", "")
        full_text_to_chunk = f"{title}. {passage_text}" if title and passage_text else passage_text or title

        if not full_text_to_chunk.strip():
            return doc_id, []

        # Gọi hàm chunking gốc
        # Các hàm semantic_grouping và semantic_splitter đã trả về List[Tuple[str, str, Optional[str]]]
        # Hàm text_splitter trả về List[List[Tuple[str, str, Optional[str]]]]
        raw_doc_chunks_result = chunking_func(doc_id, full_text_to_chunk, **params)

        # Xử lý kết quả để đảm bảo nó luôn là List[Tuple[str, str, Optional[str]]]
        # Điều này đặc biệt quan trọng cho text_splitter
        final_doc_chunks: List[Tuple[str, str, Optional[str]]] = []
        if chunking_func == chunk_passage_text_splitter:
            # text_splitter trả về [[(id, text, oie), ...]] hoặc [[]]
            if raw_doc_chunks_result and isinstance(raw_doc_chunks_result, list) and len(raw_doc_chunks_result) > 0:
                final_doc_chunks = raw_doc_chunks_result[0] # Lấy list bên trong
            else: # Trường hợp trả về [[]] hoặc một cấu trúc không mong muốn
                final_doc_chunks = []
        else: # Các hàm chunking khác đã trả về đúng định dạng
            final_doc_chunks = raw_doc_chunks_result if raw_doc_chunks_result else []


        # Logic OIE được áp dụng sau khi đã có final_doc_chunks
        # Lưu ý: Hiện tại, các hàm chunking semantic đã có logic OIE bên trong.
        # Nếu bạn muốn logic OIE chung ở đây, cần xem xét lại.
        # Đoạn code OIE dưới đây giả định final_doc_chunks là List[Tuple[chunk_id, chunk_text]]
        # và cần được điều chỉnh nếu final_doc_chunks đã chứa OIE.
        # Hiện tại, các hàm chunking semantic đã trả về (chunk_id, chunk_text, oie_str)
        # nên không cần xử lý OIE thêm ở đây nữa nếu include_oie đã được truyền vào params.

        # if params.get('include_oie', False) and final_doc_chunks:
        #     processed_doc_chunks_with_oie = []
        #     for chunk_id, chunk_text, _ in final_doc_chunks: # Giả sử final_doc_chunks có 3 phần tử
        #         # Nếu OIE đã được thêm bởi hàm chunking, không cần làm lại
        #         # Nếu không, bạn có thể gọi get_chunk_text_with_oie ở đây
        #         # augmented_chunk_text = get_chunk_text_with_oie(chunk_text, include_oie=True)
        #         # processed_doc_chunks_with_oie.append((chunk_id, augmented_chunk_text, existing_oie_str))
        #         # Tạm thời giữ nguyên vì các hàm chunking đã xử lý OIE
        #         pass
        #     # final_doc_chunks = processed_doc_chunks_with_oie # Cập nhật nếu có xử lý OIE ở đây

        return doc_id, final_doc_chunks

    except Exception as e:
        print(f"Error processing document {doc_id} in worker: {e}")
        traceback.print_exc()
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

# --- OIE Helper Functions ---
def format_oie_triples_to_string(triples_list: List[Dict[str, str]]) -> str:
    if not triples_list:
        return ""
    formatted_triples = []
    for triple in triples_list:
        s = str(triple.get('subject', '')).replace('\t', ' ').replace('\n', ' ').strip()
        r = str(triple.get('relation', '')).replace('\t', ' ').replace('\n', ' ').strip()
        o = str(triple.get('object', '')).replace('\t', ' ').replace('\n', ' ').strip()
        if s and r and o: # Only include complete triples
            formatted_triples.append(f"({s}; {r}; {o})")
    if not formatted_triples:
        return ""
    return " [OIE_TRIPLES] " + " | ".join(formatted_triples) + " [/OIE_TRIPLES]"

def get_chunk_text_with_oie(chunk_text: str, include_oie: bool) -> str:
    if not include_oie or not chunk_text.strip():
        return chunk_text
    
    aggregated_oie_triples = []
    try:
        sentences_in_chunk = nltk.sent_tokenize(chunk_text)
        if not sentences_in_chunk:
            return chunk_text

        for sentence in sentences_in_chunk:
            if not sentence.strip():
                continue
            try:
                oie_triples_for_sentence = extract_triples(sentence) # From Tool.OIE
                if oie_triples_for_sentence:
                    aggregated_oie_triples.extend(oie_triples_for_sentence)
            except Exception as e_sent_oie:
                print(f"Warning: Error extracting OIE for sentence '{sentence[:50]}...': {e_sent_oie}")
        
        if aggregated_oie_triples:
            oie_string = format_oie_triples_to_string(aggregated_oie_triples)
            return chunk_text.strip() + oie_string 
    except Exception as e_chunk_oie:
        print(f"Warning: Error processing OIE for chunk: {e_chunk_oie}")
    return chunk_text
# --- End OIE Helper Functions ---

# --- ĐẶT TOÀN BỘ LOGIC CHÍNH VÀO ĐÂY ---
if __name__ == "__main__":

    # --- Get Configuration from User ---
    common_config = get_common_params()
    selected_method = select_chunking_method()
    split_type = get_split_type() 

    method_params = {}
    if selected_method == 'semantic_grouping':
        method_params = get_semantic_grouping_params(common_config['embedding_model'])
    elif selected_method == 'semantic_splitter':
        method_params = get_semantic_splitter_params(common_config['embedding_model'])
    elif selected_method == 'text_splitter':
        method_params = get_text_splitter_params()
    
    # Quan trọng: Thêm cờ include_oie vào method_params để worker có thể truy cập
    method_params['include_oie'] = common_config.get('include_oie', False)
    if selected_method != 'text_splitter': # text_splitter không dùng model_name trực tiếp trong params của nó
        method_params['model_name'] = common_config['embedding_model']


    # --- Combine Configurations ---
    DATASET_NAME = common_config['dataset_name']
    INCLUDE_OIE = common_config.get('include_oie', False)

    method_oie_suffix = "_oie" if INCLUDE_OIE else ""
    # Sử dụng selected_method (tên gốc) cho thư mục, thêm hậu tố vào tên file
    # Hoặc có thể thêm vào tên thư mục nếu muốn tách biệt hoàn toàn
    effective_method_name_for_path = f"{selected_method.replace('_', '-')}{method_oie_suffix}"

    BASE_METHOD_DIR = os.path.join(common_config['output_dir'], f"{DATASET_NAME}_{effective_method_name_for_path}")
    OUTPUT_DIR_METHOD = find_next_run_dir(BASE_METHOD_DIR, split_type)

    METHOD_AND_SPLIT_PREFIX = f"{DATASET_NAME}_{effective_method_name_for_path}_{split_type}"
    CHUNKS_OUTPUT_PATH = os.path.join(OUTPUT_DIR_METHOD, f"{METHOD_AND_SPLIT_PREFIX}_chunks.jsonl")
    CHUNK_DOC_MAP_PATH = os.path.join(OUTPUT_DIR_METHOD, f"{METHOD_AND_SPLIT_PREFIX}_chunk_doc_map.json")
    DOC_CHUNK_MAP_PATH = os.path.join(OUTPUT_DIR_METHOD, f"{METHOD_AND_SPLIT_PREFIX}_doc_chunk_map.json")
    OUTPUT_TRIPLETS_PATH = os.path.join(OUTPUT_DIR_METHOD, f"{METHOD_AND_SPLIT_PREFIX}_triplets.tsv")

    EMBEDDING_MODEL_NAME = common_config['embedding_model']
    MAX_TRIPLETS_TO_GENERATE = common_config['max_triplets']
    MAX_DOCS_TO_PROCESS = common_config['max_docs']
    RANDOM_SEED = common_config['random_seed'] # <-- Lấy random seed

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
    LOCAL_DATA_PATH = common_config['data_dir'] 
    print(f"BEIR Data Path: {LOCAL_DATA_PATH}")
    print(f"Output Directory: {OUTPUT_DIR_METHOD}") 
    print(f"Chunking Method: {selected_method}")
    print(f"Include OIE: {'Yes' if INCLUDE_OIE else 'No'}") # Thêm dòng này
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
                     concurrent.futures.ProcessPoolExecutor(
                        max_workers=NUM_WORKERS,
                        initializer=init_worker # <<< THÊM INITIALIZER
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
