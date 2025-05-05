import matchzoo as mz
import pandas as pd
import os
import dill # Đảm bảo dill được import ở đầu

# --- !!! THAY ĐỔI CÁC ĐƯỜNG DẪN SAU !!! ---
# Sử dụng os.path.join để tạo đường dẫn một cách an toàn
BASE_MODEL_DIR = r"D:\SemanticSearch\TrainedModels" # Dùng \ cho Windows
KNRM_DIR = os.path.join(BASE_MODEL_DIR, "KNRM_mz210")

SAVED_MODEL_PATH = KNRM_DIR
SAVED_PREPROCESSOR_PATH = os.path.join(KNRM_DIR, "preprocessor.dill")

# Đường dẫn đến file triplets để đánh giá (giữ nguyên hoặc dùng os.path.join nếu muốn)
TEST_TRIPLETS_FILE = r"D:/SemanticSearch/TrainingData_MatchZoo_BEIR/msmarco_semantic-grouping/test_3/msmarco_semantic-grouping_test_triplets.tsv"
# Hoặc:
# TEST_TRIPLETS_FILE = os.path.join("D:", "SemanticSearch", "TrainingData_MatchZoo_BEIR", "msmarco_semantic-grouping", "test_3", "msmarco_semantic-grouping_test_triplets.tsv")

# --- ------------------------------------ ---

print("--- MatchZoo Evaluation ---")

# 1. Tải Preprocessor và Model
print(f"Loading preprocessor from: {SAVED_PREPROCESSOR_PATH}")
# <<< THÊM: Kiểm tra sự tồn tại của file trước khi tải >>>
if not os.path.exists(SAVED_PREPROCESSOR_PATH):
    print(f"Error: File check failed. Preprocessor not found at {SAVED_PREPROCESSOR_PATH}")
    exit()
elif not os.path.isfile(SAVED_PREPROCESSOR_PATH):
     print(f"Error: Path exists but is not a file: {SAVED_PREPROCESSOR_PATH}")
     exit()

# <<< SỬA ĐỔI: Tải thủ công bằng dill và sử dụng kết quả >>>
print(f"Attempting manual load with dill for: {SAVED_PREPROCESSOR_PATH}")
try:
    with open(SAVED_PREPROCESSOR_PATH, 'rb') as f:
        # Gán trực tiếp kết quả vào biến preprocessor
        preprocessor = dill.load(f)
    print("Manual load with dill successful. Using manually loaded preprocessor.")
    # Kiểm tra kiểu dữ liệu nếu cần
    # print(f"Loaded preprocessor type: {type(preprocessor)}")
    # <<< COMMENT OUT DÒNG GÂY LỖI >>>
    # if not isinstance(preprocessor, mz.preprocessors.BasePreprocessor):
    #      print(f"Warning: Manually loaded object might not be a MatchZoo BasePreprocessor (type: {type(preprocessor)}). Proceeding anyway.")

except FileNotFoundError:
    # Lỗi này không nên xảy ra nữa
    print("!!! Manual load with dill failed with FileNotFoundError !!!")
    exit()
except Exception as manual_err:
    print(f"!!! Manual load with dill failed with other error: {manual_err} !!!")
    exit()
# <<< KẾT THÚC TẢI THỦ CÔNG >>>

# <<< XÓA HOẶC COMMENT OUT KHỐI TRY...EXCEPT CỦA mz.load_preprocessor >>>
# try:
#     # preprocessor = mz.load_preprocessor(SAVED_PREPROCESSOR_PATH) # Không cần nữa
#     pass # Bỏ qua khối này
# except FileNotFoundError:
#     print(f"Error: mz.load_preprocessor reported FileNotFoundError for {SAVED_PREPROCESSOR_PATH}")
#     exit()
# except Exception as e:
#     print(f"Error loading preprocessor via mz.load_preprocessor: {e}")
#     exit()

print(f"Loading model from: {SAVED_MODEL_PATH}")
try:
    model = mz.load_model(SAVED_MODEL_PATH)
    # Lấy task từ model (quan trọng cho việc load data và đánh giá)
    TASK = model.params['task']
    print(f"Model loaded successfully. Task type: {TASK}")
except FileNotFoundError:
    print(f"Error: Model not found at {SAVED_MODEL_PATH}")
    exit()
except Exception as e:
    print(f"Error loading model: {e}")
    exit()


# 2. Chuẩn bị Dữ liệu Test
print(f"Loading test triplets from: {TEST_TRIPLETS_FILE}")
if not os.path.exists(TEST_TRIPLETS_FILE):
    print(f"Error: Test triplets file not found at {TEST_TRIPLETS_FILE}")
    exit()

try:
    # Sử dụng mz.pack để tải dữ liệu triplets
    test_pack_raw = mz.pack(
        {'relation_file': TEST_TRIPLETS_FILE,
         'type': 'relation',
         'task': TASK} # Sử dụng task của model
    )
    print(f"Raw test data loaded: {len(test_pack_raw.relation)} relations.")

    # Áp dụng preprocessor
    print("Applying preprocessor to test data...")
    test_pack_processed = preprocessor.transform(test_pack_raw, verbose=0)
    print("Preprocessing finished.")

except Exception as e:
    print(f"Error processing test data: {e}")
    exit()

# 3. Dự đoán
print("Predicting using the loaded model...")
try:
    # Dự đoán điểm số cho các cặp trong test_pack_processed
    # Kết quả trả về phụ thuộc vào task (ranking: scores, classification: probabilities)
    predictions = model.predict(test_pack_processed)
    print(f"Prediction finished. Got {len(predictions)} predictions.")
except Exception as e:
    print(f"Error during prediction: {e}")
    exit()

# 4. Đánh giá
print("Evaluating predictions...")

try:
    # Lấy ground truth labels từ dữ liệu đã xử lý
    # Trong file triplets, positive pair có label 1, negative có label 0 (mặc định của mz.pack)
    true_labels = test_pack_processed.relation['label'].tolist()

    if TASK == mz.tasks.Ranking():
        print("Evaluating ranking task...")
        # Đánh giá ranking từ file triplets có thể phức tạp vì cần nhóm theo query
        # và tạo danh sách xếp hạng. MatchZoo có thể không hỗ trợ trực tiếp từ format này.
        # Cách đơn giản hơn là xem xét như bài toán classification trên từng cặp.
        print("Note: Evaluating triplets directly as ranking is complex in MatchZoo.")
        print("Evaluating as pairwise classification instead (Accuracy, F1).")

        # Chuyển đổi scores thành lớp (ví dụ: > 0.5 là lớp 1) nếu cần
        # Hoặc nếu model trả về dạng [prob_class_0, prob_class_1]
        if predictions.ndim > 1 and predictions.shape[1] == 2:
             predicted_classes = predictions.argmax(axis=1)
        elif predictions.ndim == 1: # Giả sử là score, ngưỡng 0.5
             predicted_classes = (predictions > 0.5).astype(int)
        else:
             print("Warning: Cannot determine predicted classes from prediction shape.")
             predicted_classes = predictions # Giữ nguyên để evaluator xử lý nếu có thể

        evaluator = mz.metrics.Evaluator(metrics=['accuracy', 'f1'])
        results = evaluator.evaluate(y_true=true_labels, y_pred=predicted_classes)
        print("\n--- Pairwise Evaluation Results ---")
        print(results)
        print("-----------------------------------")


    elif TASK == mz.tasks.Classification():
        print("Evaluating classification task...")
        evaluator = mz.metrics.Evaluator(metrics=['accuracy', 'f1'])
        # Giả sử predictions là probabilities hoặc logits, cần argmax
        if predictions.ndim > 1:
             predicted_classes = predictions.argmax(axis=1)
        else:
             predicted_classes = predictions # Nếu model trả về trực tiếp class index

        results = evaluator.evaluate(y_true=true_labels, y_pred=predicted_classes)
        print("\n--- Classification Evaluation Results ---")
        print(results)
        print("---------------------------------------")

    else:
        print(f"Error: Unsupported task type for evaluation: {TASK}")

except Exception as e:
    print(f"Error during evaluation: {e}")

print("\nEvaluation script finished.")