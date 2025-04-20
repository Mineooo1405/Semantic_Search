from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-mpnet-base-v2")  # Hoặc: all-MiniLM-L6-v2
vec = model.encode("This is an example sentence.")
print(vec)