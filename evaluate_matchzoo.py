import matchzoo as mz
import pandas as pd
import os
import dill # Đảm bảo dill được import ở đầu
import sys # Để sử dụng sys.exit

# --- Helper function to get and validate paths ---
def get_user_path(prompt_message: str, default_path: str, is_directory: bool = False, is_file: bool = False) -> str:
    """
    Prompts the user for a path and validates its existence.
    """
    while True:
        user_input = input(f"{prompt_message}\n(default: {default_path}): ").strip()
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

# --- 1. Get Paths from User ---
print("--- MatchZoo Evaluation Setup ---")

# Get SAVED_MODEL_PATH from user
default_model_dir = r"D:\SemanticSearch\TrainedModels\model_KNRM_grouping_20250507" # Cập nhật default này nếu cần
SAVED_MODEL_PATH = get_user_path(
    prompt_message="Enter the full path to the TRAINED MODEL DIRECTORY:",
    default_path=default_model_dir,
    is_directory=True
)
SAVED_PREPROCESSOR_PATH = os.path.join(SAVED_MODEL_PATH, "preprocessor.dill")

# Get TEST_TRIPLETS_FILE from user
default_test_file = r"D:/SemanticSearch/TrainingData_MatchZoo_BEIR/msmarco_semantic-grouping/test_1/msmarco_semantic-grouping_test_triplets.tsv"
TEST_TRIPLETS_FILE = get_user_path(
    prompt_message="Enter the full path to the TEST TRIPLETS FILE (e.g., .tsv):",
    default_path=default_test_file,
    is_file=True
)

print("\n--- Paths Configured ---")
print(f"Model Directory: {SAVED_MODEL_PATH}")
print(f"Preprocessor Path: {SAVED_PREPROCESSOR_PATH}")
print(f"Test Triplets File: {TEST_TRIPLETS_FILE}")
print("------------------------\n")


print("--- MatchZoo Evaluation ---")

# 1. Tải Preprocessor và Model
print(f"Loading preprocessor from: {SAVED_PREPROCESSOR_PATH}")
if not os.path.exists(SAVED_PREPROCESSOR_PATH):
    print(f"Error: Preprocessor not found at {SAVED_PREPROCESSOR_PATH}")
    sys.exit(1) 
elif not os.path.isfile(SAVED_PREPROCESSOR_PATH):
     print(f"Error: Path exists but is not a file: {SAVED_PREPROCESSOR_PATH}")
     sys.exit(1) 

print(f"Attempting manual load with dill for: {SAVED_PREPROCESSOR_PATH}")
try:
    with open(SAVED_PREPROCESSOR_PATH, 'rb') as f:
        preprocessor = dill.load(f)
    print("Manual load with dill successful. Using manually loaded preprocessor.")
except FileNotFoundError:
    print(f"!!! Manual load with dill failed with FileNotFoundError for {SAVED_PREPROCESSOR_PATH} !!!")
    sys.exit(1) 
except Exception as manual_err:
    print(f"!!! Manual load with dill failed with other error: {manual_err} !!!")
    sys.exit(1) 

print(f"Loading model from: {SAVED_MODEL_PATH}")
try:
    model = mz.load_model(SAVED_MODEL_PATH)
    TASK = model.params['task']
    print(f"Model loaded successfully. Task type: {TASK}")
except FileNotFoundError:
    print(f"Error: Model not found at {SAVED_MODEL_PATH}")
    sys.exit(1) 
except Exception as e:
    print(f"Error loading model: {e}")
    sys.exit(1) 


# 2. Chuẩn bị Dữ liệu Test
print(f"Loading test triplets from: {TEST_TRIPLETS_FILE}")
if not os.path.exists(TEST_TRIPLETS_FILE):
    print(f"Error: Test triplets file not found at {TEST_TRIPLETS_FILE}")
    sys.exit(1) 

try:
    # Đọc file triplets (giả sử là TSV không có header: query, doc_pos, doc_neg)
    triplets_df = pd.read_csv(TEST_TRIPLETS_FILE, sep='\t', header=None, names=['query', 'doc_pos', 'doc_neg'], quoting=3) # quoting=3 để xử lý lỗi quote
    print(f"Loaded {len(triplets_df)} triplets.")

    # Chuyển đổi triplets thành các cặp cho MatchZoo
    pairs_list = []
    for index, row in triplets_df.iterrows():
        query_text = str(row['query'])
        doc_pos_text = str(row['doc_pos'])
        doc_neg_text = str(row['doc_neg'])

        # Cặp positive
        pairs_list.append({
            'id_left': f"q_{index}",    # ID cho query (có thể là index)
            'id_right': f"dp_{index}",  # ID cho document positive
            'text_left': query_text,
            'text_right': doc_pos_text,
            'label': 1                  # Nhãn positive
        })
        # Cặp negative
        pairs_list.append({
            'id_left': f"q_{index}",    # ID cho query
            'id_right': f"dn_{index}",  # ID cho document negative
            'text_left': query_text,
            'text_right': doc_neg_text,
            'label': 0                  # Nhãn negative
        })
    
    pairs_df = pd.DataFrame(pairs_list)
    print(f"Transformed into {len(pairs_df)} pairs for evaluation.")

    # Đảm bảo các cột văn bản là kiểu string
    pairs_df['text_left'] = pairs_df['text_left'].astype(str)
    pairs_df['text_right'] = pairs_df['text_right'].astype(str)
    pairs_df['label'] = pairs_df['label'].astype(int)


    # Tạo DataPack từ DataFrame đã xử lý
    test_pack_raw = mz.pack(
        data_pack=pairs_df, # Truyền DataFrame vào đây
        task=TASK           # TASK được lấy từ model đã load
    )
    print(f"Raw test data packed: {len(test_pack_raw.relation)} relations.")

    print("Applying preprocessor to test data...")
    test_pack_processed = preprocessor.transform(test_pack_raw, verbose=0)
    print("Preprocessing finished.")

except pd.errors.ParserError as pe:
    print(f"Error parsing TSV file: {pe}")
    print("Please ensure the test file is a valid TSV with 3 columns and no header, or adjust reading parameters.")
    sys.exit(1)
except Exception as e:
    print(f"Error processing test data: {e}")
    sys.exit(1) 

# 3. Dự đoán
print("Predicting using the loaded model...")
try:
    predictions = model.predict(test_pack_processed)
    print(f"Prediction finished. Got {len(predictions)} predictions.")
except Exception as e:
    print(f"Error during prediction: {e}")
    sys.exit(1) 

# 4. Đánh giá
print("Evaluating predictions...")

try:
    # Lấy nhãn thực tế từ test_pack_processed, nơi chúng đã được lưu trữ
    true_labels = test_pack_processed.relation['label'].tolist()


    if isinstance(TASK, mz.tasks.Ranking): # Kiểm tra type của TASK chính xác hơn
        print("Evaluating ranking task...")
        # Đối với ranking, MatchZoo thường đánh giá dựa trên các metric như MAP, NDCG trên một tập các item được xếp hạng cho mỗi query.
        # Tuy nhiên, vì chúng ta đã chuyển đổi thành các cặp (query, doc) với nhãn 0/1,
        # chúng ta có thể đánh giá như một bài toán classification trên các cặp này.
        print("Note: Evaluating triplets by converting to pairwise classification (Accuracy, F1).")

        # Predictions có thể là scores (float) hoặc probabilities (cho mỗi class nếu model output vậy)
        if predictions.ndim > 1 and predictions.shape[1] > 1: # Output là probabilities cho nhiều class
             predicted_classes = predictions.argmax(axis=1)
        elif predictions.ndim == 1: # Output là scores hoặc prob cho class positive
             predicted_classes = (predictions > 0.5).astype(int) # Ngưỡng 0.5 cho scores/probs
        else:
             print(f"Warning: Cannot determine predicted classes from prediction shape {predictions.shape}. Using raw predictions.")
             predicted_classes = predictions

        evaluator = mz.metrics.Evaluator(metrics=['accuracy', 'f1', mz.metrics.Precision(), mz.metrics.Recall()])
        results = evaluator.evaluate(y_true=true_labels, y_pred=predicted_classes)
        print("\n--- Pairwise Evaluation Results (Ranking as Classification) ---")
        for metric_name, metric_val in results.items():
            print(f"{metric_name}: {metric_val}")
        print("-------------------------------------------------------------")

    elif isinstance(TASK, mz.tasks.Classification): # Kiểm tra type của TASK
        print("Evaluating classification task...")
        evaluator = mz.metrics.Evaluator(metrics=['accuracy', 'f1', mz.metrics.Precision(), mz.metrics.Recall()])
        if predictions.ndim > 1 and predictions.shape[1] > 1:
             predicted_classes = predictions.argmax(axis=1)
        elif predictions.ndim == 1:
             predicted_classes = (predictions > 0.5).astype(int)
        else:
             print(f"Warning: Cannot determine predicted classes from prediction shape {predictions.shape}. Using raw predictions.")
             predicted_classes = predictions

        results = evaluator.evaluate(y_true=true_labels, y_pred=predicted_classes)
        print("\n--- Classification Evaluation Results ---")
        for metric_name, metric_val in results.items():
            print(f"{metric_name}: {metric_val}")
        print("---------------------------------------")

    else:
        print(f"Error: Unsupported task type for evaluation: {type(TASK)}")

except Exception as e:
    print(f"Error during evaluation: {e}")

print("\nEvaluation script finished.")