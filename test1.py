# filepath: d:\SemanticSearch\test_directml_article.py
import torch
import torch_directml # Quan trọng: import gói này

print(f"PyTorch version: {torch.__version__}")

try:
    print("Attempting to get DirectML device using torch_directml.device()...")
    dml_device = torch_directml.device()
    print(f"Successfully got DML device: {dml_device}")

    print("Attempting to create tensors on the DML device...")
    tensor1 = torch.tensor([1]).to(dml_device)
    tensor2 = torch.tensor([2]).to(dml_device)
    print(f"Tensor1: {tensor1}, Device: {tensor1.device}")
    print(f"Tensor2: {tensor2}, Device: {tensor2.device}")

    print("Attempting to add tensors on the DML device...")
    dml_algebra = tensor1 + tensor2
    result = dml_algebra.item()
    print(f"Result of addition: {result}")

    if result == 3:
        print("DirectML test PASSED!")
    else:
        print(f"DirectML test FAILED! Expected 3, got {result}")

except Exception as e:
    print(f"An error occurred: {e}")
    print("DirectML test FAILED!")

# Kiểm tra thêm thông tin device nếu có thể
try:
    if hasattr(torch_directml, 'is_available') and torch_directml.is_available():
        print("torch_directml.is_available(): True")
        print(f"torch_directml.device_name(): {torch_directml.device_name(0)}") # Giả sử device_id là 0
        print(f"torch_directml.device_count(): {torch_directml.device_count()}")
    else:
        print("torch_directml.is_available(): False or not found")
except Exception as e_info:
    print(f"Could not get additional DirectML info: {e_info}")