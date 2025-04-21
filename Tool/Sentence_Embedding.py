import os
os.environ["HF_HOME"] = "D:/SemanticSearch/EmbeddingModel"
os.environ["SENTENCE_TRANSFORMERS_HOME"] = "D:/SemanticSearch/EmbeddingModel"

from sentence_transformers import SentenceTransformer

def sentence_embedding(text):
    
    model = SentenceTransformer("all-mpnet-base-v2")
    
    return model.encode(text)