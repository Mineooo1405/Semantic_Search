import subprocess
import os
import sys

# --- Configuration for the controller ---
PYTHON_EXECUTABLE = sys.executable # Uses the same python interpreter that runs controller.py
# Or, specify a path directly:
# PYTHON_EXECUTABLE = "d:/SemanticSearch/env_3124/Scripts/python.exe" 
MAIN_SCRIPT_PATH = "Train_MatchZoo_BEIR_Multiple.py" 

# Default parameters that will be used for all runs
# You can modify these if needed
COMMON_DEFAULTS = {
    "dataset_name": "msmarco",
    "data_dir": "./Data/msmarco",
    "output_dir": "D:/SemanticSearch/TrainingData_MatchZoo_BEIR",
    "embedding_model": "thenlper/gte-large",
    "max_triplets": "0", # 0 for unlimited
    "max_docs": "20",
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

def build_input_sequence(config: dict) -> str:
    """
    Builds the newline-separated string of inputs for Train_MatchZoo_BEIR_Multiple.py
    """
    inputs = []
    
    # Common parameters
    inputs.append(COMMON_DEFAULTS["dataset_name"])
    inputs.append(COMMON_DEFAULTS["data_dir"])
    inputs.append(COMMON_DEFAULTS["output_dir"])
    inputs.append(COMMON_DEFAULTS["embedding_model"])
    inputs.append(COMMON_DEFAULTS["max_triplets"])
    inputs.append(COMMON_DEFAULTS["max_docs"])
    inputs.append(config["include_oie"]) # This is specific to the run
    inputs.append(COMMON_DEFAULTS["random_seed"])

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


def run_main_script(config: dict):
    """
    Runs the Train_MatchZoo_BEIR_Multiple.py script with the given configuration.
    """
    print(f"\n{'='*20} Starting Run {'='*20}")
    print(f"Method: {config['method_name']}, OIE: {config['include_oie'].upper()}")
    print(f"{'-'*50}")

    input_sequence = build_input_sequence(config)

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


if __name__ == "__main__":
    if not os.path.exists(MAIN_SCRIPT_PATH):
        print(f"Error: Main script '{MAIN_SCRIPT_PATH}' not found in the current directory.")
        print(f"Please ensure '{MAIN_SCRIPT_PATH}' is in the same directory as controller.py or update MAIN_SCRIPT_PATH.")
        exit()
        
    print(f"Using Python executable: {PYTHON_EXECUTABLE}")
    print(f"Controlling script: {MAIN_SCRIPT_PATH}")

    for i, run_conf in enumerate(RUN_CONFIGURATIONS):
        print(f"\n>>> Processing Configuration {i+1} of {len(RUN_CONFIGURATIONS)} <<<")
        run_main_script(run_conf)

    print("\nAll configured runs have been attempted.")