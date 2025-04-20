import os
os.environ["HF_DATASETS_CACHE"] = "D:/SemanticSearch/Data"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

from datasets import load_dataset

# Tải phiên bản v1.1 của MS MARCO
dataset = load_dataset("microsoft/ms_marco", "v1.1")

# Lấy 1000 đoạn văn đầu tiên
passages = dataset['train'].select(range(1000))

# In thử vài dòng
for i in range(5):
    print(passages[i])
