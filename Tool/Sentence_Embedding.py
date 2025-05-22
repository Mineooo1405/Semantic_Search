import os
os.environ["HF_HOME"] = "D:/SemanticSearch/EmbeddingModel"
os.environ["SENTENCE_TRANSFORMERS_HOME"] = "D:/SemanticSearch/EmbeddingModel"

from sentence_transformers import SentenceTransformer
import torch
import torch_directml # Đảm bảo import này có mặt

# Biến toàn cục để lưu trữ thiết bị, khởi tạo một lần
# Điều này hữu ích nếu hàm sentence_embedding được gọi nhiều lần trong cùng một tiến trình
# mà không muốn lặp lại việc lấy device.
# Tuy nhiên, trong ngữ cảnh ProcessPoolExecutor với init_worker,
# mỗi worker sẽ có instance riêng của module này.
_dml_device = None

def get_dml_device():
    """Lấy thiết bị DirectML. Khởi tạo nếu chưa có."""
    global _dml_device
    if _dml_device is None:
        try:
            if torch_directml.is_available():
                _dml_device = torch_directml.device()
                print(f"Sentence_Embedding: Successfully got DML device: {_dml_device} in process {os.getpid()}")
            else:
                print(f"Sentence_Embedding: DirectML not available, falling back to CPU in process {os.getpid()}.")
                _dml_device = torch.device("cpu")
        except Exception as e:
            print(f"Sentence_Embedding: Error getting DML device in process {os.getpid()}: {e}. Falling back to CPU.")
            _dml_device = torch.device("cpu")
    return _dml_device

# Cache cho model để không phải tải lại mỗi lần gọi hàm trong cùng một tiến trình
# Key sẽ là model_name_or_path, value là đối tượng model đã tải
_model_cache = {}

def sentence_embedding(text, model_name_or_path="thenlper/gte-large", batch_size=32, device=None): # Added device parameter
    """
    Nhúng văn bản sử dụng SentenceTransformer, ưu tiên DirectML.
    Nếu 'device' được cung cấp, sẽ sử dụng device đó.
    Ngược lại, sẽ cố gắng lấy DML device qua get_dml_device().
    """
    global _model_cache
    
    target_device = None
    if device is not None:
        target_device = device
        # print(f"Sentence_Embedding: Using provided device: {target_device} in process {os.getpid()}")
    else:
        target_device = get_dml_device() # Lấy thiết bị (DML hoặc CPU) nếu không có device nào được truyền vào
        # print(f"Sentence_Embedding: No device provided, got device via get_dml_device(): {target_device} in process {os.getpid()}")

    # Kiểm tra xem model đã được tải cho device này chưa
    # Sử dụng str(target_device) để đảm bảo key là hashable và nhất quán
    cache_key = (model_name_or_path, str(target_device))

    if cache_key not in _model_cache:
        print(f"Sentence_Embedding: Loading model '{model_name_or_path}' onto device '{target_device}' in process {os.getpid()}...")
        try:
            # Truyền device vào SentenceTransformer
            model = SentenceTransformer(model_name_or_path, device=target_device)
            _model_cache[cache_key] = model
            print(f"Sentence_Embedding: Model '{model_name_or_path}' loaded successfully on '{target_device}'.")
        except Exception as e:
            print(f"Sentence_Embedding: Error loading model '{model_name_or_path}' on device '{target_device}': {e}")
            # Nếu lỗi khi tải lên device được chỉ định (hoặc DML), thử tải lên CPU
            # Chỉ thử CPU nếu device ban đầu không phải là CPU
            if str(target_device).lower() != "cpu":
                print(f"Sentence_Embedding: Falling back to CPU for model '{model_name_or_path}'.")
                cpu_device = torch.device("cpu")
                # Cập nhật cache key cho CPU
                cpu_cache_key = (model_name_or_path, str(cpu_device))
                if cpu_cache_key not in _model_cache: # Kiểm tra lại cache cho CPU
                    try:
                        model = SentenceTransformer(model_name_or_path, device=cpu_device)
                        _model_cache[cpu_cache_key] = model # Cache với device là CPU
                        print(f"Sentence_Embedding: Model '{model_name_or_path}' loaded successfully on CPU after fallback.")
                    except Exception as e_cpu:
                        print(f"Sentence_Embedding: Error loading model '{model_name_or_path}' on CPU during fallback: {e_cpu}")
                        # Nếu vẫn lỗi trên CPU, raise lỗi hoặc trả về None/Exception tùy theo yêu cầu xử lý lỗi
                        raise # Hoặc xử lý khác
                else:
                    model = _model_cache[cpu_cache_key]
                target_device = cpu_device # Cập nhật target_device để encode dùng CPU
            else: # Nếu target_device ban đầu đã là CPU và vẫn lỗi, thì raise lỗi
                raise
    else:
        # print(f"Sentence_Embedding: Using cached model '{model_name_or_path}' from device '{target_device}'.") # Bỏ comment nếu muốn log
        model = _model_cache[cache_key]

    # Đảm bảo model đang ở đúng target_device trước khi encode
    # Điều này quan trọng nếu model được lấy từ cache và target_device có thể đã thay đổi (ví dụ: fallback sang CPU)
    # Tuy nhiên, SentenceTransformer quản lý device của nó. Nếu model được load đúng device, không cần .to() nữa.
    # if model.device != target_device: # model.device là torch.device object
    #    model.to(target_device) # Chỉ di chuyển nếu device của model khác với target_device hiện tại

    # print(f"Sentence_Embedding: Encoding text on device '{model.device}' (target: '{target_device}')...") # Bỏ comment nếu muốn log chi tiết
    return model.encode(text, batch_size=batch_size, show_progress_bar=False) # Tắt progress bar mặc định ở đây

# Hàm dọn dẹp cache (tùy chọn, có thể không cần thiết nếu tiến trình tự kết thúc)
def clear_embedding_cache():
    global _model_cache
    global _dml_device
    print("Sentence_Embedding: Clearing model cache and DML device.")
    _model_cache.clear()
    _dml_device = None
    # Có thể thêm torch.cuda.empty_cache() nếu dùng CUDA, hoặc tương tự cho DML nếu có
    if torch_directml and hasattr(torch_directml, 'empty_cache'): # Kiểm tra xem có hàm empty_cache không
        try:
            torch_directml.empty_cache()
            print("Sentence_Embedding: Called torch_directml.empty_cache()")
        except Exception as e:
            print(f"Sentence_Embedding: Error calling torch_directml.empty_cache(): {e}")

# Đăng ký hàm dọn dẹp khi script thoát (nếu cần)
# import atexit
# atexit.register(clear_embedding_cache)