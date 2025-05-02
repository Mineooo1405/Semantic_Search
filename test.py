from Semantic_Grouping import process_and_store_grouped_chunks
import pandas as pd


passages = pd.read_csv("d:/SemanticSearch/passages_10.csv")
passages = passages['passage_text'].tolist()
print(f"Đã trích xuất {len(passages)} đoạn văn")

process_and_store_grouped_chunks(passages, 0.6, 0.7, 0.1,"thenlper/gte-large",True)