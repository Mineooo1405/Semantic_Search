import os
os.environ["HF_HOME"] = "D:/SemanticSearch/EmbeddingModel"
os.environ["SENTENCE_TRANSFORMERS_HOME"] = "D:/SemanticSearch/EmbeddingModel"

from sentence_transformers import SentenceTransformer
#"all-mpnet-base-v2",         # Mô hình gốc bạn đang dùng
#"paraphrase-MiniLM-L6-v2",   # Mô hình nhỏ hơn, nhanh hơn
#"paraphrase-multilingual-mpnet-base-v2", # Mô hình đa ngôn ngữ tốt
#"sentence-transformers/msmarco-distilbert-base-tas-b", # Mô hình tốt cho tìm kiếm (TAS-B)
# Thêm các mô hình khác nếu muốn, ví dụ:
# "sentence-transformers/all-MiniLM-L12-v2",
# "sentence-transformers/multi-qa-mpnet-base-dot-v1", # Tốt cho Question Answering
# "thenlper/gte-large" # Mô hình lớn hơn, có thể cần nhiều tài nguyên hơn

def sentence_embedding(text, model_name="thenlper/gte-large"):
    
    model = SentenceTransformer(model_name)
    
    return model.encode(text)
