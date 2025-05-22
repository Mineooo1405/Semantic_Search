import os
import subprocess
import sys

# Define the directory containing the training scripts
TRAIN_SCRIPTS_DIR = os.path.join(os.path.dirname(__file__), "train")

def get_available_training_scripts():
    """Scans the TRAIN_SCRIPTS_DIR for Python training scripts."""
    scripts = []
    if not os.path.isdir(TRAIN_SCRIPTS_DIR):
        print(f"Error: Training scripts directory not found: {TRAIN_SCRIPTS_DIR}")
        return scripts
    for fname in os.listdir(TRAIN_SCRIPTS_DIR):
        if fname.endswith("_training_script.py") and os.path.isfile(os.path.join(TRAIN_SCRIPTS_DIR, fname)):
            # Extract model name from filename (e.g., "MatchLSTM" from "MatchLSTM_training_script.py")
            model_name = fname.replace("_training_script.py", "")
            scripts.append({"name": model_name, "path": os.path.join(TRAIN_SCRIPTS_DIR, fname)})
    return scripts

def get_user_data_paths():
    """Prompts the user to enter paths for train, dev, and test datasets."""
    print("--- Dataset Configuration ---")
    while True:
        train_path = input("Enter the absolute path for the TRAINING dataset (e.g., D:\\\\\\\\data\\\\\\\\train.tsv or D:/data/train.tsv): ").strip()
        if os.path.isfile(train_path):
            break
        print(f"Error: File not found or path is not a file: {train_path}. Please try again.")

    while True:
        dev_path = input("Enter the absolute path for the DEVELOPMENT/VALIDATION dataset (e.g., D:\\\\\\\\data\\\\\\\\dev.tsv or D:/data/dev.tsv): ").strip()
        if os.path.isfile(dev_path):
            break
        print(f"Error: File not found or path is not a file: {dev_path}. Please try again.")

    while True:
        test_path = input("Enter the absolute path for the TEST dataset (e.g., D:\\\\\\\\data\\\\\\\\test.tsv or D:/data/test.tsv): ").strip()
        if os.path.isfile(test_path):
            break
        print(f"Error: File not found or path is not a file: {test_path}. Please try again.")
    return train_path, dev_path, test_path # Corrected: test_file to test_path

def modify_script_paths(script_path, train_file, dev_file, test_file):
    try:
        with open(script_path, 'r', encoding='utf-8') as f:
            content = f.readlines()
    except Exception as e:
        print(f"Error reading script {script_path}: {e}")
        return None

    modified_content = []
    path_vars = {
        "TRAIN_FILE_PATH": train_file,
        "DEV_FILE_PATH": dev_file,
        "TEST_FILE_PATH": test_file
    }

    for line in content:
        modified_line = line
        stripped_line = line.strip()

        for var_name, new_path_value in path_vars.items():
            # DEBUG: Print path details
            # print(f"DEBUG: Checking var: {var_name} in line: {stripped_line}") # Can be too verbose
            if stripped_line.startswith(var_name) and "=" in stripped_line:
                parts = stripped_line.split("=", 1)
                if parts[0].strip() == var_name:
                    # print(f"DEBUG: Matched variable {var_name} in line: {line.strip()}")
                    
                    current_normalized_path = str(new_path_value).replace('\\\\', '/')
                    # print(f"DEBUG: For {var_name}, attempting to set path to: {current_normalized_path}")

                    assignment_value_part = parts[1].strip()
                    raw_prefix = ""
                    quote_char = ''

                    # Determine original quoting style
                    if assignment_value_part.startswith("r\\\"") and assignment_value_part.endswith("\\\""):
                        raw_prefix = "r"
                        quote_char = '\\\"' 
                        # print(f"DEBUG: Quoting style for {var_name}: r\\\"...\\\" (raw double quotes)")
                    elif assignment_value_part.startswith("\\\"") and assignment_value_part.endswith("\\\""):
                        quote_char = '\\\"'
                        # print(f"DEBUG: Quoting style for {var_name}: \\\"...\\\" (standard double quotes)")
                    elif assignment_value_part.startswith("r\'") and assignment_value_part.endswith("\'"):
                        raw_prefix = "r"
                        quote_char = "\\\'"
                        # print(f"DEBUG: Quoting style for {var_name}: r\'...\' (raw single quotes)")
                    elif assignment_value_part.startswith("\'") and assignment_value_part.endswith("\'"):
                        quote_char = "\\\'"
                        # print(f"DEBUG: Quoting style for {var_name}: \'...\' (standard single quotes)")
                    else:
                        # print(f"DEBUG: Quoting style for {var_name} NOT MATCHED. Assignment part: >>>{assignment_value_part}<<< Original line: >>>{line.strip()}<<<")
                        # If quoting is unusual or not a path string, skip modification for this line to be safe
                        continue 

                    new_line_content = f"{var_name} = {raw_prefix}{quote_char}{current_normalized_path}{quote_char}"
                    
                    indentation = line[:len(line) - len(line.lstrip())]
                    modified_line = indentation + new_line_content + "\\n"
                    # print(f"DEBUG: Replaced line for {var_name}: {modified_line.strip()}")
                    break 
        modified_content.append(modified_line)
    
    final_script_str = "".join(modified_content)
    # print("\\nDEBUG: First 25 lines of modified script content to be written:\\n" + "-"*30) # Keep this for verification
    # print("\\n".join(final_script_str.splitlines()[:25]))
    # print("-"*30 + "\\n")
    return final_script_str

def run_training_script(script_path, train_file, dev_file, test_file):
    """
    Modifies the target script with new data paths, saves it to a temporary file,
    runs the temporary file, and then deletes it.
    """
    print(f"--- Training Model from: {os.path.basename(script_path)} ---")

    modified_script_content = modify_script_paths(script_path, train_file, dev_file, test_file)

    if not modified_script_content:
        print(f"Failed to modify script {script_path}. Skipping.")
        return False

    # Create a temporary file path
    base, ext = os.path.splitext(script_path)
    temp_script_path = f"{base}_temp{ext}" # e.g., DRMM_training_script_temp.py

    try:
        with open(temp_script_path, 'w', encoding='utf-8') as f:
            f.write(modified_script_content)
        # print(f"DEBUG: Successfully wrote modified content to temporary file: {temp_script_path}")

        # DEBUG: Optionally, read back and print a few lines from the temp file
        # try:
        #     with open(temp_script_path, 'r', encoding='utf-8') as f_check:
        #         print(f"\\\\nDEBUG: First 10 lines read back from temp file '{temp_script_path}':\\\\n" + "-"*30)
        #         print("".join(f_check.readlines()[:10]))
        #         print("-"*30 + "\\\\n")
        # except Exception as e_read_check:
        #     print(f"DEBUG: Could not read back temp file for verification: {e_read_check}")

        project_root_dir = os.path.dirname(os.path.abspath(__file__))
        print(f"Running modified temporary script: {temp_script_path} with cwd: {project_root_dir}...")
        process = subprocess.run(
            [sys.executable, temp_script_path],
            capture_output=False, text=True, check=False,
            cwd=project_root_dir  # Set current working directory
        )

        if process.returncode == 0:
            print(f"Successfully completed training for {os.path.basename(script_path)}.")
            success = True
        else:
            print(f"Error during training of {os.path.basename(script_path)} (using {temp_script_path}).")
            print(f"Return code: {process.returncode}")
            print("Check the output above for specific error messages from the script.")
            success = False
        return success

    except Exception as e:
        print(f"An error occurred while preparing or running {temp_script_path} for {script_path}: {e}")
        return False
    finally:
        # Clean up: remove temporary script
        if os.path.exists(temp_script_path):
            try:
                os.remove(temp_script_path)
                # print(f"DEBUG: Successfully deleted temporary script: {temp_script_path}")
            except Exception as e_del:
                print(f"Warning: Could not delete temporary script {temp_script_path}: {e_del}")
        # Original script at script_path is not modified by this function.

def main():
    print("--- Master Training Script ---")
    available_scripts = get_available_training_scripts()

    if not available_scripts:
        print("No training scripts found in the 'train' directory. Exiting.")
        return

    print("Available models for training:")
    for i, script_info in enumerate(available_scripts):
        print(f"{i+1}. {script_info['name']}")

    selected_indices = []
    while True:
        try:
            choices_str = input("Enter the numbers of the models to train (comma-separated, e.g., 1,3 or 'all'): ").strip().lower()
            if choices_str == 'all':
                selected_indices = list(range(len(available_scripts)))
                break
            selected_indices = [int(x.strip()) - 1 for x in choices_str.split(',')]
            if all(0 <= idx < len(available_scripts) for idx in selected_indices):
                break
            else:
                print("Invalid selection. Please enter numbers from the list.")
        except ValueError:
            print("Invalid input. Please enter numbers separated by commas or 'all'.")

    if not selected_indices:
        print("No models selected for training. Exiting.")
        return

    train_file, dev_file, test_file = get_user_data_paths()
    
    print(f"Will use the following datasets for all selected models:")
    print(f"  Train: {train_file}")
    print(f"  Dev:   {dev_file}")
    print(f"  Test:  {test_file}")
    
    confirm = input("Proceed with training? (yes/no): ").strip().lower()
    if confirm != 'yes':
        print("Training cancelled by user.")
        return

    for idx in selected_indices:
        script_info = available_scripts[idx]
        run_training_script(script_info['path'], train_file, dev_file, test_file)

    print("\\n--- All selected training processes finished. ---")

if __name__ == "__main__":
    main()
