import subprocess
import os
import sys
import csv
import random
from beir import util
from beir.datasets.data_loader import GenericDataLoader
from typing import Dict, List, Tuple, Optional, Union # Added Dict
from tqdm.auto import tqdm # For progress bars
import traceback # For detailed error logging

# --- Configuration for the controller ---
PYTHON_EXECUTABLE = sys.executable # Uses the same python interpreter that runs controller.py
# Or, specify a path directly:
# PYTHON_EXECUTABLE = "d:/SemanticSearch/env_3124/Scripts/python.exe" 
MAIN_SCRIPT_PATH = "Train_MatchZoo_BEIR_Multiple.py" 

# Default parameters that will be used for all runs
# You can modify these if needed
COMMON_DEFAULTS = {
    "dataset_name": "msmarco",
    "data_dir": "./Data/msmarco", # This should be the path to the specific dataset folder, e.g., ./Data/msmarco/msmarco
    "output_dir": "D:/SemanticSearch/TrainingData_MatchZoo_BEIR",
    "embedding_model": "thenlper/gte-large",
    "max_triplets": "0", # 0 for unlimited, for doc-level triplets
    "max_docs": "20", # Max docs for chunking in Train_MatchZoo_BEIR_Multiple.py
    "random_seed": "42",
    "split_type": "train"
}

SEMANTIC_GROUPING_DEFAULTS = {
    "initial_threshold": "auto",
    "decay_factor": "0.85",
    "min_threshold": "auto",
    "initial_percentile": "95",
    "min_percentile": "10",
    "embedding_batch_size": "24"
}

SEMANTIC_SPLITTER_DEFAULTS = {
    "initial_threshold": "0.6",
    "decay_factor": "0.95",
    "min_threshold": "0.35",
    "min_chunk_len": "2",
    "max_chunk_len": "8",
    "window_size": "3",
    "embedding_batch_size": "32"
}

TEXT_SPLITTER_DEFAULTS = {
    "chunk_size": "1000",
    "chunk_overlap": "200"
}

# Define the sequence of runs
RUN_CONFIGURATIONS = [
    {"method_name": "semantic_grouping", "method_choice": "1", "include_oie": "no", "params": SEMANTIC_GROUPING_DEFAULTS},
    {"method_name": "semantic_grouping", "method_choice": "1", "include_oie": "yes", "params": SEMANTIC_GROUPING_DEFAULTS},
    {"method_name": "semantic_splitter", "method_choice": "2", "include_oie": "no", "params": SEMANTIC_SPLITTER_DEFAULTS},
    {"method_name": "semantic_splitter", "method_choice": "2", "include_oie": "yes", "params": SEMANTIC_SPLITTER_DEFAULTS},
    {"method_name": "text_splitter", "method_choice": "3", "include_oie": "no", "params": TEXT_SPLITTER_DEFAULTS},
    {"method_name": "text_splitter", "method_choice": "3", "include_oie": "yes", "params": TEXT_SPLITTER_DEFAULTS},
]

# Global variable to store loaded BEIR data to avoid reloading
LOADED_BEIR_DATA = None

def load_beir_data_once(dataset_name: str, data_path: str) -> Tuple[Dict[str, Dict[str, str]], Dict[str, str], Dict[str, Dict[str, int]]]:
    """Loads BEIR dataset (corpus, queries, qrels) once and stores it globally."""
    global LOADED_BEIR_DATA
    if LOADED_BEIR_DATA is None:
        try:
            print(f"Loading BEIR dataset: {dataset_name} from {data_path}...")
            corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="train")
            LOADED_BEIR_DATA = (corpus, queries, qrels)
            print(f"Successfully loaded {len(corpus)} documents, {len(queries)} queries, and {len(qrels)} qrels.")
        except Exception as e:
            print(f"Failed to load data from {data_path}. Attempting to download {dataset_name}...")
            try:
                url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset_name}.zip"
                download_path = util.download_and_unzip(url, "datasets")
                corpus, queries, qrels = GenericDataLoader(data_folder=download_path).load(split="train")
                LOADED_BEIR_DATA = (corpus, queries, qrels)
                print(f"Successfully downloaded and loaded {dataset_name}.")
            except Exception as download_e:
                print(f"Error during download or loading of {dataset_name}: {download_e}")
                print(traceback.format_exc())
                raise
    else:
        print(f"Using already loaded BEIR dataset: {dataset_name}.")
    return LOADED_BEIR_DATA

def build_input_sequence(config: dict, doc_level_triplets_path_arg: str) -> str:
    """
    Builds the newline-separated string of inputs for Train_MatchZoo_BEIR_Multiple.py
    """
    inputs = []
    
    # Common parameters for Train_MatchZoo_BEIR_Multiple.py
    inputs.append(COMMON_DEFAULTS["dataset_name"])
    inputs.append(COMMON_DEFAULTS["data_dir"])
    inputs.append(COMMON_DEFAULTS["output_dir"])
    inputs.append(COMMON_DEFAULTS["embedding_model"])
    # inputs.append(COMMON_DEFAULTS["max_triplets"]) # Removed, Train_Script doesn't generate from scratch
    inputs.append(COMMON_DEFAULTS["max_docs"])
    inputs.append(config["include_oie"]) # This is specific to the run
    inputs.append(COMMON_DEFAULTS["random_seed"]) # For reproducibility within Train_Script (e.g., chunk sampling)
    inputs.append(doc_level_triplets_path_arg) # New: Path to doc-level triplets

    # Select Chunking Method
    inputs.append(config["method_choice"])

    # Select Data Split
    inputs.append(COMMON_DEFAULTS["split_type"])

    # Method-Specific Parameters
    method_params = config["params"]
    if config["method_name"] == "semantic_grouping":
        inputs.append(method_params["initial_threshold"])
        inputs.append(method_params["decay_factor"])
        inputs.append(method_params["min_threshold"])
        if method_params["initial_threshold"] == "auto" or method_params["min_threshold"] == "auto":
            inputs.append(method_params["initial_percentile"])
            inputs.append(method_params["min_percentile"])
        inputs.append(method_params["embedding_batch_size"])
    elif config["method_name"] == "semantic_splitter":
        inputs.append(method_params["initial_threshold"])
        inputs.append(method_params["decay_factor"])
        inputs.append(method_params["min_threshold"])
        inputs.append(method_params["min_chunk_len"])
        inputs.append(method_params["max_chunk_len"])
        inputs.append(method_params["window_size"])
        inputs.append(method_params["embedding_batch_size"])
    elif config["method_name"] == "text_splitter":
        inputs.append(method_params["chunk_size"])
        inputs.append(method_params["chunk_overlap"])
        
    return "\n".join(inputs) + "\n"


def run_main_script(config: dict, doc_level_triplets_path_for_script: str):
    """
    Runs the Train_MatchZoo_BEIR_Multiple.py script with the given configuration.
    """
    print(f"\n{'='*20} Starting Run {'='*20}")
    print(f"Method: {config['method_name']}, OIE: {config['include_oie'].upper()}")
    print(f"{'-'*50}")

    input_sequence = build_input_sequence(config, doc_level_triplets_path_for_script)

    try:
        process = subprocess.Popen(
            [PYTHON_EXECUTABLE, MAIN_SCRIPT_PATH],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,  # Ensures stdin/stdout/stderr are treated as text
            cwd=os.path.dirname(os.path.abspath(__file__)) # Run from controller's directory
        )
        stdout, stderr = process.communicate(input=input_sequence, timeout=7200) # 2 hour timeout

        print("--- STDOUT ---")
        print(stdout)
        
        if process.returncode != 0:
            print(f"--- STDERR (Return Code: {process.returncode}) ---")
            print(stderr)
            print(f"Run failed for: Method: {config['method_name']}, OIE: {config['include_oie'].upper()}")
        else:
            print(f"Run completed successfully for: Method: {config['method_name']}, OIE: {config['include_oie'].upper()}")

    except subprocess.TimeoutExpired:
        print(f"--- TIMEOUT ---")
        print(f"Run timed out for: Method: {config['method_name']}, OIE: {config['include_oie'].upper()}")
        process.kill()
        stdout, stderr = process.communicate()
        print("--- STDOUT (on timeout) ---")
        print(stdout)
        print("--- STDERR (on timeout) ---")
        print(stderr)
    except Exception as e:
        print(f"An unexpected error occurred while running the script for {config['method_name']} (OIE: {config['include_oie']}): {e}")
        import traceback
        traceback.print_exc()

    print(f"{'-'*50}\n")


def generate_and_save_doc_level_triplets(corpus: Dict[str, Dict[str, str]], 
                                         queries: Dict[str, str], 
                                         qrels: Dict[str, Dict[str, int]], 
                                         output_folder: str, 
                                         dataset_name: str,
                                         max_triplets_per_query: int = 10,
                                         random_seed: int = 42) -> str:
    """
    Generates document-level triplets (query_text, positive_doc_id, negative_doc_id)
    and saves them to a TSV file.
    Returns the path to the saved triplet file.
    """
    random.seed(random_seed)
    # Sanitize dataset_name for use in filenames
    safe_dataset_name = "".join(c if c.isalnum() else "_" for c in dataset_name)
    triplets_file_path = os.path.join(output_folder, f"{safe_dataset_name}_doc_level_triplets.tsv")

    if os.path.exists(triplets_file_path):
        print(f"Document-level triplets file already exists: {triplets_file_path}. Skipping generation.")
        return triplets_file_path

    os.makedirs(output_folder, exist_ok=True)
    doc_level_triplets = []
    all_doc_ids = list(corpus.keys())
    if not all_doc_ids:
        print("Corpus is empty. Cannot generate triplets.")
        return triplets_file_path # Return path even if empty, Train_MatchZoo will handle it

    print(f"Generating document-level triplets for {dataset_name}...")
    for query_id, query_text in tqdm(queries.items(), desc="Generating doc triplets"):
        if query_id not in qrels:
            continue

        positive_doc_ids = {doc_id for doc_id, score in qrels[query_id].items() if score > 0}
        if not positive_doc_ids:
            continue

        # Create a pool of potential negative document IDs for this query
        # These are documents not in the positive set for the current query
        potential_negative_doc_ids = [doc_id for doc_id in all_doc_ids if doc_id not in positive_doc_ids]
        
        if not potential_negative_doc_ids:
            # This case is rare but possible if all docs are positive for a query
            # Or if the corpus is very small and only contains positive documents for this query
            print(f"Warning: No potential negative documents found for query_id {query_id}. Skipping triplet generation for this query.")
            continue

        triplets_for_this_query = 0
        for pos_doc_id in positive_doc_ids:
            if triplets_for_this_query >= max_triplets_per_query:
                break
            
            # Sample a negative document ID from the potential negatives
            # Ensure it's different from the positive document ID (already handled by construction of potential_negative_doc_ids)
            neg_doc_id = random.choice(potential_negative_doc_ids)
            doc_level_triplets.append((query_text, pos_doc_id, neg_doc_id))
            triplets_for_this_query += 1
    
    print(f"Generated {len(doc_level_triplets)} document-level triplets.")

    with open(triplets_file_path, 'w', newline='', encoding='utf-8') as f_out:
        writer = csv.writer(f_out, delimiter='\t')
        writer.writerow(["query_text", "positive_doc_id", "negative_doc_id"]) # Header
        writer.writerows(doc_level_triplets)
    
    print(f"Document-level triplets saved to: {triplets_file_path}")
    return triplets_file_path


if __name__ == "__main__":
    if not os.path.exists(MAIN_SCRIPT_PATH):
        print(f"Error: Main script '{MAIN_SCRIPT_PATH}' not found in the current directory.")
        print(f"Please ensure '{MAIN_SCRIPT_PATH}' is in the same directory as controller.py or update MAIN_SCRIPT_PATH.")
        exit()
        
    print(f"Using Python executable: {PYTHON_EXECUTABLE}")
    print(f"Controlling script: {MAIN_SCRIPT_PATH}")

    # --- Load BEIR data and Generate/Ensure Document-Level Triplets ---
    print("\n--- Preparing shared data (BEIR load & Doc-Level Triplets) ---")
    
    # COMMON_DEFAULTS["data_dir"] should point to the folder containing the actual dataset files
    # e.g., "./Data/msmarco/msmarco" if the files are directly in a subfolder named like the dataset.
    # The load_beir_data_once function handles the case where it might need to download into a parent of this.
    
    corpus, queries, qrels = load_beir_data_once(
        dataset_name=COMMON_DEFAULTS["dataset_name"],
        data_folder=COMMON_DEFAULTS["data_dir"], 
        split=COMMON_DEFAULTS["split_type"] 
    )

    # Define where the document-level triplets will be stored
    # This path should be inside the general output_dir, under a subfolder for the dataset
    doc_level_triplets_output_base_dir = os.path.join(COMMON_DEFAULTS["output_dir"], COMMON_DEFAULTS["dataset_name"])
    doc_level_triplets_file = os.path.join(doc_level_triplets_output_base_dir, f"doc_level_triplets_{COMMON_DEFAULTS['split_type']}.tsv")
    
    generate_and_save_doc_level_triplets(
        corpus, queries, qrels, COMMON_DEFAULTS, doc_level_triplets_file
    )
    print("--- Shared data preparation complete ---")
    # --- End of shared data preparation ---

    for i, run_conf in enumerate(RUN_CONFIGURATIONS):
        print(f"\n>>> Processing Configuration {i+1} of {len(RUN_CONFIGURATIONS)} <<<")
        run_main_script(run_conf, doc_level_triplets_file)

    print("\nAll configured runs have been attempted.")