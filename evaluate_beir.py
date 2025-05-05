import os
import json
import logging
import time
from typing import Dict, List, Tuple

# BEIR imports
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES

# Model import (ví dụ)
from sentence_transformers import SentenceTransformer

# --- Cấu hình Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- !!! THAY ĐỔI CÁC ĐƯỜNG DẪN VÀ THAM SỐ SAU !!! ---
# Đường dẫn đến thư mục chứa dữ liệu BEIR gốc (queries.jsonl, qrels/test.tsv)
ORIGINAL_BEIR_DATA_PATH = "./Data/msmarco/msmarco"
# Đường dẫn đến thư mục chứa kết quả xử lý (chunks.jsonl, chunk_doc_map.json)
PROCESSED_DATA_DIR = "D:/SemanticSearch/TrainingData_MatchZoo_BEIR/msmarco_semantic-grouping/test_3"
# Đường dẫn hoặc tên Hugging Face của mô hình retrieval đã huấn luyện/fine-tune
# QUAN TRỌNG: Thay thế bằng mô hình thực tế của bạn!
RETRIEVAL_MODEL_PATH = "thenlper/gte-large" # Ví dụ: Sử dụng mô hình gốc
# RETRIEVAL_MODEL_PATH = "path/to/your/fine-tuned/model" # Hoặc đường dẫn đến mô hình đã fine-tune

# Tên file output từ script xử lý trước đó
CHUNKS_FILENAME = "msmarco_semantic-grouping_test_chunks.jsonl"
CHUNK_DOC_MAP_FILENAME = "msmarco_semantic-grouping_test_chunk_doc_map.json"

# Tham số đánh giá
TOP_K_RETRIEVAL = 100 # Số lượng kết quả chunk trả về cho mỗi query
BATCH_SIZE_ENCODE = 128 # Batch size khi tạo embedding (điều chỉnh theo VRAM/RAM)
SCORE_FUNCTION = "cos_sim" # "cos_sim" hoặc "dot" (phụ thuộc vào mô hình)
K_VALUES_EVAL = [1, 3, 5, 10, 20, 100] # Các giá trị k để tính metrics
# --- ------------------------------------------------- ---

def load_beir_data(data_path: str, split: str = "test") -> Tuple[Dict[str, str], Dict[str, int]]:
    """Tải queries và qrels từ dataset BEIR gốc."""
    logging.info(f"Loading original BEIR {split} data from: {data_path}")
    if not os.path.exists(data_path):
        logging.error(f"Original BEIR data path not found: {data_path}")
        raise FileNotFoundError(f"Directory not found: {data_path}")
    try:
        _, queries, qrels = GenericDataLoader(data_folder=data_path).load(split=split)
        logging.info(f"Loaded {len(queries)} {split} queries and qrels for {len(qrels)} queries.")
        return queries, qrels
    except Exception as e:
        logging.error(f"Error loading original BEIR {split} data: {e}", exc_info=True)
        raise

def load_processed_data(processed_dir: str, chunks_fname: str, map_fname: str) -> Tuple[Dict[str, Dict[str, str]], Dict[str, str]]:
    """Tải chunks và map chunk-to-document đã tạo."""
    chunks_file_path = os.path.join(processed_dir, chunks_fname)
    chunk_doc_map_file_path = os.path.join(processed_dir, map_fname)

    # Tải Chunks
    logging.info(f"Loading generated chunks from: {chunks_file_path}")
    if not os.path.exists(chunks_file_path):
        logging.error(f"Chunks file not found: {chunks_file_path}")
        raise FileNotFoundError(f"File not found: {chunks_file_path}")
    chunks_corpus = {}
    try:
        with open(chunks_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                chunk_id = data.get('chunk_id')
                text = data.get("text", "")
                if chunk_id:
                    chunks_corpus[chunk_id] = {"text": text} # Format yêu cầu bởi BEIR
        logging.info(f"Loaded {len(chunks_corpus)} chunks.")
        if not chunks_corpus:
            raise ValueError("Chunks corpus is empty. Cannot proceed.")
    except Exception as e:
        logging.error(f"Error loading chunks file: {e}", exc_info=True)
        raise

    # Tải Map Chunk -> Doc
    logging.info(f"Loading chunk-to-doc map from: {chunk_doc_map_file_path}")
    if not os.path.exists(chunk_doc_map_file_path):
        logging.error(f"Chunk-to-doc map file not found: {chunk_doc_map_file_path}")
        raise FileNotFoundError(f"File not found: {chunk_doc_map_file_path}")
    try:
        with open(chunk_doc_map_file_path, 'r', encoding='utf-8') as f:
            chunk_to_doc_map = json.load(f)
        logging.info(f"Loaded chunk-to-doc map for {len(chunk_to_doc_map)} chunks.")
        if not chunk_to_doc_map:
            logging.warning("Chunk-to-doc map is empty.")
    except Exception as e:
        logging.error(f"Error loading chunk-to-doc map: {e}", exc_info=True)
        raise

    return chunks_corpus, chunk_to_doc_map

def load_retrieval_model(model_path: str):
    """Tải mô hình retrieval (ví dụ: SentenceTransformer)."""
    logging.info(f"Loading retrieval model from: {model_path}")
    try:
        # Giả sử dùng SentenceTransformer - Thay đổi nếu dùng mô hình khác
        model = SentenceTransformer(model_path)
        logging.info("Retrieval model loaded successfully.")
        return model
    except Exception as e:
        logging.error(f"Error loading retrieval model: {e}", exc_info=True)
        raise

def perform_retrieval(model, chunks_corpus: Dict[str, Dict[str, str]], queries: Dict[str, str], top_k: int, batch_size: int, score_function: str) -> Dict[str, Dict[str, float]]:
    """Thực hiện tìm kiếm dense trên chunks."""
    logging.info("Initializing Dense Retrieval Search...")
    beir_search_model = DRES(model, batch_size=batch_size)

    logging.info(f"Performing retrieval (Top {top_k}) on {len(chunks_corpus)} chunks for {len(queries)} queries...")
    start_time = time.time()
    try:
        chunk_results = beir_search_model.search(
            corpus=chunks_corpus,
            queries=queries,
            top_k=top_k,
            score_function=score_function
        )
        end_time = time.time()
        logging.info(f"Retrieval finished in {end_time - start_time:.2f} seconds.")
        if not chunk_results:
            raise ValueError("Retrieval returned no results.")
        return chunk_results
    except Exception as e:
        logging.error(f"Error during retrieval: {e}", exc_info=True)
        raise

def map_chunks_to_docs_maxp(chunk_results: Dict[str, Dict[str, float]], chunk_to_doc_map: Dict[str, str]) -> Dict[str, Dict[str, float]]:
    """Ánh xạ kết quả từ chunk ID sang document ID sử dụng MaxP aggregation."""
    logging.info("Mapping chunk results to document results using MaxP aggregation...")
    doc_results = {}
    mapping_errors = 0
    mapped_queries = 0
    for qid, chunk_scores in chunk_results.items():
        doc_scores_for_query = {}
        for chunk_id, score in chunk_scores.items():
            doc_id = chunk_to_doc_map.get(chunk_id)
            if doc_id:
                # Giữ điểm cao nhất cho mỗi document (MaxP)
                if doc_id not in doc_scores_for_query or score > doc_scores_for_query[doc_id]:
                    doc_scores_for_query[doc_id] = score
            else:
                mapping_errors += 1
        if doc_scores_for_query:
            doc_results[qid] = doc_scores_for_query
            mapped_queries += 1

    if mapping_errors > 0:
        logging.warning(f"Could not map {mapping_errors} chunk results to documents (chunk_id not found in map).")
    logging.info(f"Mapping finished. Got document results for {mapped_queries} queries.")
    if not doc_results:
        raise ValueError("No document results after mapping. Cannot evaluate.")
    return doc_results

def evaluate_beir_results(qrels: Dict[str, Dict[str, int]], doc_results: Dict[str, Dict[str, float]], k_values: List[int]) -> Dict[str, Dict[str, float]]:
    """Đánh giá kết quả retrieval bằng BEIR."""
    logging.info("Evaluating document results using BEIR evaluator...")
    try:
        evaluator = EvaluateRetrieval()
        results_metrics = evaluator.evaluate(qrels, doc_results, k_values)
        return results_metrics
    except Exception as e:
        logging.error(f"Error during evaluation: {e}", exc_info=True)
        raise

def print_results(results_metrics: Dict[str, Dict[str, float]]):
    """In kết quả đánh giá ra console."""
    print("\n--- BEIR Evaluation Results ---")
    for metric in sorted(results_metrics.keys()):
        print(f"\n# {metric.upper()} Scores #")
        for k, score in results_metrics[metric].items():
            print(f"{metric}@{k}: {score:.4f}")
    print("-----------------------------")

def main():
    """Hàm chính điều khiển luồng đánh giá."""
    print("--- BEIR Evaluation Script ---")
    try:
        # 1. Tải dữ liệu
        queries, qrels = load_beir_data(ORIGINAL_BEIR_DATA_PATH, split="test")
        chunks_corpus, chunk_to_doc_map = load_processed_data(PROCESSED_DATA_DIR, CHUNKS_FILENAME, CHUNK_DOC_MAP_FILENAME)

        # 2. Tải mô hình
        retrieval_model = load_retrieval_model(RETRIEVAL_MODEL_PATH)

        # 3. Thực hiện retrieval
        chunk_results = perform_retrieval(retrieval_model, chunks_corpus, queries, TOP_K_RETRIEVAL, BATCH_SIZE_ENCODE, SCORE_FUNCTION)

        # 4. Map kết quả
        doc_results = map_chunks_to_docs_maxp(chunk_results, chunk_to_doc_map)

        # 5. Đánh giá
        evaluation_results = evaluate_beir_results(qrels, doc_results, K_VALUES_EVAL)

        # 6. In kết quả
        print_results(evaluation_results)

    except (FileNotFoundError, ValueError, Exception) as e:
        logging.error(f"Script failed: {e}", exc_info=True)
        print(f"\nScript execution failed. Check logs for details.")

    print("\nBEIR evaluation script finished.")

if __name__ == "__main__":
    main()