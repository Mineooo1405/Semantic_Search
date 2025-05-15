import torch
print(torch.cuda.is_available()) # Kiểm tra CUDA (sẽ là False cho AMD)
print(hasattr(torch, 'dml') and torch.dml.is_available()) # Kiểm tra DirectML