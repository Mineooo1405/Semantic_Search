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
# import argparse # REMOVE argparse

try:
    from Semantic_Grouping import semantic_chunk_passage_from_grouping_logic
except ImportError:
    print("WARNING: Could not import 'semantic_chunk_passage_from_grouping_logic' from Semantic_Grouping.py. Semantic Grouping method will not be available.")
    def semantic_chunk_passage_from_grouping_logic(*args, **kwargs):
        print("ERROR: Semantic Grouping logic not available.")
        return []
try:
    from Semantic_Splitter import chunk_passage_semantic_splitter
except ImportError:
    print("WARNING: Could not import 'chunk_passage_semantic_splitter'. Semantic Splitter method will not be available.")
    def chunk_passage_semantic_splitter(*args, **kwargs):
        print("ERROR: Semantic Splitter logic not available.")
        return []
try:
    from Text_Splitter import chunk_passage_text_splitter
except ImportError:
    print("WARNING: Could not import 'chunk_passage_text_splitter'. Text Splitter method will not be available.")
    def chunk_passage_text_splitter(*args, **kwargs):
        print("ERROR: Text Splitter logic not available.")
        return []


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
    params['data_dir'] = get_input("Directory for BEIR data", default="./Data", required=True)
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

# --- Get Configuration from User ---
common_config = get_common_params()
selected_method = select_chunking_method()
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
LOCAL_DATA_PATH = os.path.join(common_config['data_dir'], DATASET_NAME)
METHOD_SUFFIX = selected_method.replace('_', '-')
OUTPUT_DIR_METHOD = os.path.join(common_config['output_dir'], f"{DATASET_NAME}_{METHOD_SUFFIX}")
CHUNKS_OUTPUT_PATH = os.path.join(OUTPUT_DIR_METHOD, f"{DATASET_NAME}_{METHOD_SUFFIX}_chunks.jsonl")
CHUNK_DOC_MAP_PATH = os.path.join(OUTPUT_DIR_METHOD, f"{DATASET_NAME}_{METHOD_SUFFIX}_chunk_doc_map.json")
DOC_CHUNK_MAP_PATH = os.path.join(OUTPUT_DIR_METHOD, f"{DATASET_NAME}_{METHOD_SUFFIX}_doc_chunk_map.json")
TRAIN_TRIPLETS_PATH = os.path.join(OUTPUT_DIR_METHOD, f"{DATASET_NAME}_{METHOD_SUFFIX}_train_triplets.tsv")
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
print(f"BEIR Data Path: {LOCAL_DATA_PATH}")
print(f"Output Directory: {OUTPUT_DIR_METHOD}")
print(f"Chunking Method: {selected_method}")
print(f"Embedding Model: {EMBEDDING_MODEL_NAME}")
print(f"Max Triplets: {MAX_TRIPLETS_TO_GENERATE if MAX_TRIPLETS_TO_GENERATE else 'Unlimited'}")
print(f"Max Documents to Process: {MAX_DOCS_TO_PROCESS if MAX_DOCS_TO_PROCESS else 'All'}")
print(f"Chunking Parameters: {method_params}")
print(f"---------------------------\n")

# --- Download NLTK data ---

try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    print("Downloading NLTK 'punkt' tokenizer...")
    nltk.download('punkt', quiet=True)

# --- Helper Functions ---
def clean_text(text):
    text = str(text)
    text = text.replace('\n', ' ').replace('\r', ' ')
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# --- Main Processing ---
os.makedirs(OUTPUT_DIR_METHOD, exist_ok=True)
os.makedirs(LOCAL_DATA_PATH, exist_ok=True)

# == Step 1: Load BEIR Data ==
print(f"\nLoading BEIR dataset: {DATASET_NAME} from {LOCAL_DATA_PATH}...")
data_folder_to_load = LOCAL_DATA_PATH
try:
    corpus, queries, qrels = GenericDataLoader(data_folder=data_folder_to_load).load(split="train")
    print(f"Loaded {len(corpus)} documents, {len(queries)} queries, and qrels for {len(qrels)} queries from {data_folder_to_load}.")
except Exception as e:
    print(f"Error loading BEIR dataset '{DATASET_NAME}' from {data_folder_to_load}: {e}")
    print(f"Attempting to download using 'util.download_and_unzip'...")
    try:
        url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{DATASET_NAME}.zip"
        download_target_dir = os.path.dirname(LOCAL_DATA_PATH)
        if not download_target_dir: download_target_dir = "."
        util.download_and_unzip(url, download_target_dir)
        print(f"Dataset downloaded and unzipped potentially into: {LOCAL_DATA_PATH}")
        potential_subfolder_path = os.path.join(LOCAL_DATA_PATH, DATASET_NAME)
        if os.path.exists(os.path.join(potential_subfolder_path, "corpus.jsonl")):
            print(f"Adjusting load path to subfolder: {potential_subfolder_path}")
            data_folder_to_load = potential_subfolder_path
        corpus, queries, qrels = GenericDataLoader(data_folder=data_folder_to_load).load(split="train")
        print(f"Successfully loaded {len(corpus)} documents, {len(queries)} queries, and qrels for {len(qrels)} queries from {data_folder_to_load}.")
    except Exception as download_err:
        print(f"Failed to download or load the dataset after download attempt: {download_err}")
        print(f"Please manually check the directory structure in '{common_config['data_dir']}' and ensure '{DATASET_NAME}' contains corpus.jsonl, queries.jsonl, and qrels/train.tsv.")
        exit()


# == Step 2: Chunk Passages ==
if not os.path.exists(CHUNKS_OUTPUT_PATH) or not os.path.exists(DOC_CHUNK_MAP_PATH):
    print(f"\nStarting Step 2: Chunking documents using '{selected_method}' method...")
    if MAX_DOCS_TO_PROCESS:
        print(f"Processing a maximum of {MAX_DOCS_TO_PROCESS} documents.")
    chunk_to_doc_map = {}
    doc_to_chunk_map = {}
    processed_docs = 0
    docs_processed_count = 0

    try:
        corpus_iterable = corpus.items()
        if MAX_DOCS_TO_PROCESS:
            corpus_iterable = list(corpus.items())[:MAX_DOCS_TO_PROCESS]
            tqdm_total = MAX_DOCS_TO_PROCESS
        else:
            tqdm_total = len(corpus)

        with open(CHUNKS_OUTPUT_PATH, 'w', encoding='utf-8') as f_chunks_out:
            for doc_id, doc_data in tqdm(corpus_iterable, desc=f"Chunking Corpus ({selected_method})", total=tqdm_total):
                passage_text = doc_data.get("text", "")
                title = doc_data.get("title", "")
                full_text_to_chunk = f"{title}. {passage_text}" if title else passage_text
                if not full_text_to_chunk: continue

                # --- MODIFIED CALL TO CHUNKING FUNCTION ---
                # Pass all method_params using **kwargs
                # The chunking function will pick the ones it needs
                doc_chunks = selected_chunking_func(
                    doc_id,
                    full_text_to_chunk,
                    **method_params # Pass the whole dictionary
                )

                if doc_chunks:
                    doc_to_chunk_map[doc_id] = []
                    for chunk_id, chunk_text in doc_chunks:
                        chunk_to_doc_map[chunk_id] = doc_id
                        doc_to_chunk_map[doc_id].append(chunk_id)
                        f_chunks_out.write(json.dumps({'chunk_id': chunk_id, 'text': chunk_text}) + '\n')
                processed_docs += 1
                docs_processed_count += 1

    except Exception as e:
        print(f"\nAn error occurred during chunking: {e}") # Add newline for clarity
        import traceback
        traceback.print_exc()
        exit()

    print(f"\nFinished chunking {processed_docs} documents.") # Add newline
    print(f"Total chunks created: {len(chunk_to_doc_map)}")
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
print(f"\nStarting Step 3: Creating Triplets for MatchZoo...")
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
        for line in tqdm(f, desc="Loading Chunks"):
            try:
                chunk_info = json.loads(line)
                chunks_data[chunk_info['chunk_id']] = chunk_info['text']
            except json.JSONDecodeError: continue
    print(f"Loaded text for {len(chunks_data)} chunks.")
except FileNotFoundError: print(f"Error: Chunks file not found at {CHUNKS_OUTPUT_PATH}"); exit()
except Exception as e: print(f"An error occurred loading chunks data: {e}"); exit()
all_chunk_ids = list(chunks_data.keys())
if not all_chunk_ids: print("Error: No chunks loaded."); exit()
# 4. Map qid -> set(positive_chunk_id)
query_positive_chunks = {}
print("Mapping queries to positive chunk IDs...")
for qid, pos_doc_ids in tqdm(query_positive_docs.items(), desc="Mapping Positive Chunks"):
    query_positive_chunks[qid] = set()
    for doc_id in pos_doc_ids:
        if doc_id in doc_to_chunk_map:
            for chunk_id in doc_to_chunk_map[doc_id]:
                if chunk_id in chunks_data: query_positive_chunks[qid].add(chunk_id)
# 5. Generate Triplets
print(f"Generating triplets and saving to {TRAIN_TRIPLETS_PATH}...")
triplets_count = 0
try:
    with open(TRAIN_TRIPLETS_PATH, 'w', encoding='utf-8') as f_train_out:
        query_ids = list(queries.keys())
        random.shuffle(query_ids)
        for qid in tqdm(query_ids, desc="Generating Triplets"):
            if qid not in queries or qid not in query_positive_chunks or not query_positive_chunks[qid]: continue
            query_text = clean_text(queries[qid])
            positive_chunk_ids_for_query = query_positive_chunks[qid]
            for positive_chunk_id in positive_chunk_ids_for_query:
                positive_chunk_text = chunks_data.get(positive_chunk_id)
                if not positive_chunk_text: continue
                negative_chunk_id = None
                for _ in range(10):
                    potential_negative_id = random.choice(all_chunk_ids)
                    if potential_negative_id not in positive_chunk_ids_for_query:
                        negative_chunk_id = potential_negative_id
                        break
                if negative_chunk_id:
                    negative_chunk_text = chunks_data.get(negative_chunk_id)
                    if negative_chunk_text:
                        f_train_out.write(f"{query_text}\t{positive_chunk_text}\t{negative_chunk_text}\n")
                        triplets_count += 1
                        if MAX_TRIPLETS_TO_GENERATE is not None and triplets_count >= MAX_TRIPLETS_TO_GENERATE:
                            print(f"\nReached limit of {MAX_TRIPLETS_TO_GENERATE} triplets.")
                            raise StopIteration
            if MAX_TRIPLETS_TO_GENERATE is not None and triplets_count >= MAX_TRIPLETS_TO_GENERATE: break
except StopIteration: pass
except Exception as e:
    print(f"An error occurred generating triplets: {e}")
    import traceback
    traceback.print_exc()

print(f"Finished generating {triplets_count} triplets.")
print(f"Training triplets saved to: {TRAIN_TRIPLETS_PATH}")
print(f"\n--- Steps 1, 2 and 3 (Chunking: {selected_method}, Triplet Generation for MatchZoo using BEIR) Completed ---")
print(f"Output file for MatchZoo training: {TRAIN_TRIPLETS_PATH}")
