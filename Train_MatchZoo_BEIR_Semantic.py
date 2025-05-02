import pandas as pd
import nltk
# Đảm bảo các import từ Semantic_Grouping.py hoạt động
# Thêm đường dẫn gốc vào sys.path nếu cần
# import sys
# sys.path.append('d:\\SemanticSearch') # Hoặc đường dẫn phù hợp
from Tool.Sentence_Detector import extract_and_simplify_sentences
from Tool.Sentence_Embedding import sentence_embedding as embed_text_list
from Semantic_Grouping import (
    create_semantic_matrix,
    analyze_similarity_distribution,
    semantic_spreading_grouping
) # Import các hàm cần thiết
import os
import re
from tqdm.auto import tqdm
import random
import json
import hashlib
from beir import util
from beir.datasets.data_loader import GenericDataLoader
import numpy as np # Cần cho percentile

# --- Configuration ---
DATASET_NAME = "msmarco"
# Tải và lưu vào thư mục cục bộ ./Data/msmarco
LOCAL_DATA_PATH = os.path.join("./Data", DATASET_NAME)

OUTPUT_DIR = "D:/SemanticSearch/TrainingData_MatchZoo_BEIR_Semantic" # Thư mục lưu kết quả
CHUNKS_OUTPUT_PATH = os.path.join(OUTPUT_DIR, f"{DATASET_NAME}_semantic_chunks.jsonl")
CHUNK_DOC_MAP_PATH = os.path.join(OUTPUT_DIR, f"{DATASET_NAME}_semantic_chunk_doc_map.json")
DOC_CHUNK_MAP_PATH = os.path.join(OUTPUT_DIR, f"{DATASET_NAME}_semantic_doc_chunk_map.json")
TRAIN_TRIPLETS_PATH = os.path.join(OUTPUT_DIR, f"{DATASET_NAME}_semantic_train_triplets.tsv") # File output cho MatchZoo

# Semantic Chunking parameters (sử dụng auto hoặc đặt giá trị cụ thể)
CHUNK_INITIAL_THRESHOLD = 'auto'
CHUNK_DECAY_FACTOR = 0.85
CHUNK_MIN_THRESHOLD = 'auto'
CHUNK_AUTO_PERCENTILES = (85, 25) # (initial_p, min_p)
EMBEDDING_MODEL_NAME = "thenlper/gte-large" # Model dùng để chunking

# Giới hạn số lượng triplets (đặt None để dùng hết)
MAX_TRIPLETS_TO_GENERATE = 100000 # Ví dụ: tạo 100k triplets

# --- Download NLTK data (nếu chưa có) ---
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    print("Downloading NLTK 'punkt' tokenizer...")
    nltk.download('punkt', quiet=True)

# --- Helper Functions ---
def clean_text(text):
    text = str(text)
    text = text.replace('\n', ' ').replace('\r', ' ')
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def semantic_chunk_passage_from_grouping_logic(
    doc_id,
    passage_text,
    model_name=EMBEDDING_MODEL_NAME,
    initial_threshold=CHUNK_INITIAL_THRESHOLD,
    decay_factor=CHUNK_DECAY_FACTOR,
    min_threshold=CHUNK_MIN_THRESHOLD,
    auto_percentiles=CHUNK_AUTO_PERCENTILES
):
    """
    Sử dụng logic từ Semantic_Grouping.py để chunk một passage.
    Trả về list of (chunk_id, chunk_text).
    """
    chunks_result = []
    try:
        # a. Tách câu
        sentences = extract_and_simplify_sentences(passage_text, simplify=False)
        if not sentences or len(sentences) < 2:
            if sentences: # Nếu chỉ có 1 câu, coi đó là 1 chunk
                chunk_text = sentences[0]
                chunk_hash = hashlib.md5(chunk_text.encode()).hexdigest()[:10]
                chunk_id = f"{doc_id}_{chunk_hash}"
                chunks_result.append((chunk_id, chunk_text))
            return chunks_result

        # b. Tạo embedding cho câu
        sentence_vectors = embed_text_list(sentences, model_name=model_name)
        if sentence_vectors is None: return []

        # c. Tạo ma trận tương đồng
        sim_matrix = create_semantic_matrix(sentence_vectors)

        # d. Xác định ngưỡng (nếu auto)
        current_initial_threshold = initial_threshold
        current_min_threshold = min_threshold
        if initial_threshold == 'auto' or min_threshold == 'auto':
            percentiles = analyze_similarity_distribution(sim_matrix)
            if not percentiles:
                 current_initial_threshold = 0.8 # Default fallback
                 current_min_threshold = 0.2   # Default fallback
            else:
                try:
                    initial_p, min_p = auto_percentiles
                    initial_key, min_key = f'p{initial_p}', f'p{min_p}'

                    if initial_threshold == 'auto':
                        current_initial_threshold = percentiles.get(initial_key, 0.8)
                    elif isinstance(initial_threshold, str): current_initial_threshold = float(initial_threshold)

                    if min_threshold == 'auto':
                        current_min_threshold = percentiles.get(min_key, 0.2)
                    elif isinstance(min_threshold, str): current_min_threshold = float(min_threshold)

                    # Đảm bảo min < initial
                    if current_min_threshold >= current_initial_threshold:
                        current_min_threshold = max(0.1, current_initial_threshold * 0.5)
                except Exception:
                    current_initial_threshold = 0.8 if initial_threshold == 'auto' else float(initial_threshold)
                    current_min_threshold = 0.2 if min_threshold == 'auto' else float(min_threshold)
        elif isinstance(initial_threshold, str): current_initial_threshold = float(initial_threshold)
        elif isinstance(min_threshold, str): current_min_threshold = float(min_threshold)


        # e. Phân nhóm ngữ nghĩa
        groups = semantic_spreading_grouping(sim_matrix, current_initial_threshold, decay_factor, current_min_threshold)

        # f. Tạo chunk từ nhóm
        for group_idx, group in enumerate(groups):
            chunk_sentences = [sentences[sent_idx] for sent_idx in group]
            chunk_text = " ".join(chunk_sentences).strip()
            if chunk_text:
                chunk_hash = hashlib.md5(chunk_text.encode()).hexdigest()[:10]
                chunk_id = f"{doc_id}_{chunk_hash}_{group_idx}" # Thêm group_idx để tránh trùng hash
                chunks_result.append((chunk_id, chunk_text))

    except Exception as e:
        print(f"Error chunking doc {doc_id}: {e}") # Log lỗi
    return chunks_result

# --- Main Processing ---
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LOCAL_DATA_PATH, exist_ok=True) # Tạo thư mục Data/msmarco

# == Bước 1: Tải dữ liệu bằng BEIR vào thư mục cục bộ ==
print(f"Loading BEIR dataset: {DATASET_NAME} into {LOCAL_DATA_PATH}...")
try:
    # Chỉ định data_folder để BEIR tải/load từ đó
    corpus, queries, qrels = GenericDataLoader(data_folder=LOCAL_DATA_PATH).load(split="train")
    print(f"Loaded {len(corpus)} documents, {len(queries)} queries, and qrels for {len(qrels)} queries from BEIR.")
except Exception as e:
    print(f"Error loading BEIR dataset '{DATASET_NAME}' from {LOCAL_DATA_PATH}.")
    print(f"Attempting to download using 'util.download_and_unzip'...")
    try:
        # URL có thể cần cập nhật nếu BEIR thay đổi
        url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{DATASET_NAME}.zip"
        # Tải về thư mục cha của LOCAL_DATA_PATH để giải nén đúng cấu trúc
        download_target_dir = os.path.dirname(LOCAL_DATA_PATH)
        util.download_and_unzip(url, download_target_dir) # BEIR sẽ tạo thư mục con msmarco
        print(f"Dataset downloaded and unzipped to: {LOCAL_DATA_PATH}")
        # Thử load lại
        corpus, queries, qrels = GenericDataLoader(data_folder=LOCAL_DATA_PATH).load(split="train")
        print(f"Successfully loaded {len(corpus)} documents, {len(queries)} queries, and qrels for {len(qrels)} queries.")
    except Exception as download_err:
        print(f"Failed to download or load the dataset: {download_err}")
        print(f"Please ensure the dataset '{DATASET_NAME}' exists at '{LOCAL_DATA_PATH}' or can be downloaded.")
        exit()

# == Bước 2: Chunk passages (documents) bằng Semantic Grouping Logic ==
if not os.path.exists(CHUNKS_OUTPUT_PATH) or not os.path.exists(DOC_CHUNK_MAP_PATH):
    print(f"\nStarting Step 2: Chunking documents using Semantic Grouping logic...")
    chunk_to_doc_map = {}
    doc_to_chunk_map = {}
    processed_docs = 0

    try:
        with open(CHUNKS_OUTPUT_PATH, 'w', encoding='utf-8') as f_chunks_out:
            for doc_id, doc_data in tqdm(corpus.items(), desc="Chunking Corpus"):
                passage_text = doc_data.get("text", "")
                title = doc_data.get("title", "")
                full_text_to_chunk = f"{title}. {passage_text}" if title else passage_text
                if not full_text_to_chunk: continue

                # Sử dụng hàm chunking mới
                doc_chunks = semantic_chunk_passage_from_grouping_logic(
                    doc_id,
                    full_text_to_chunk,
                    model_name=EMBEDDING_MODEL_NAME,
                    initial_threshold=CHUNK_INITIAL_THRESHOLD,
                    decay_factor=CHUNK_DECAY_FACTOR,
                    min_threshold=CHUNK_MIN_THRESHOLD,
                    auto_percentiles=CHUNK_AUTO_PERCENTILES
                )

                if doc_chunks:
                    doc_to_chunk_map[doc_id] = []
                    for chunk_id, chunk_text in doc_chunks:
                        chunk_to_doc_map[chunk_id] = doc_id
                        doc_to_chunk_map[doc_id].append(chunk_id)
                        f_chunks_out.write(json.dumps({'chunk_id': chunk_id, 'text': chunk_text}) + '\n')
                processed_docs += 1

    except Exception as e:
        print(f"An error occurred during chunking: {e}")
        exit()

    print(f"Finished chunking {processed_docs} documents.")
    print(f"Total chunks created: {len(chunk_to_doc_map)}")
    print(f"Chunks saved to: {CHUNKS_OUTPUT_PATH}")

    print("Saving mapping files...")
    try:
        with open(CHUNK_DOC_MAP_PATH, 'w', encoding='utf-8') as f: json.dump(chunk_to_doc_map, f)
        print(f"Chunk -> Doc map saved to: {CHUNK_DOC_MAP_PATH}")
        with open(DOC_CHUNK_MAP_PATH, 'w', encoding='utf-8') as f: json.dump(doc_to_chunk_map, f)
        print(f"Doc -> Chunk map saved to: {DOC_CHUNK_MAP_PATH}")
    except Exception as e: print(f"Error saving mapping files: {e}")
    del chunk_to_doc_map # Giải phóng bộ nhớ

else:
    print("Chunk files already exist. Skipping Step 2.")
    print(f"Loading existing Doc -> Chunk map from {DOC_CHUNK_MAP_PATH}...")
    try:
        with open(DOC_CHUNK_MAP_PATH, 'r', encoding='utf-8') as f:
            doc_to_chunk_map = json.load(f)
        print(f"Loaded map for {len(doc_to_chunk_map)} documents.")
    except FileNotFoundError:
        print(f"Error: Map file {DOC_CHUNK_MAP_PATH} not found. Please run chunking again.")
        exit()
    except Exception as e:
        print(f"Error loading map file: {e}")
        exit()

# == Bước 3: Gán nhãn và Tạo Triplets ==
print(f"\nStarting Step 3: Creating Triplets for MatchZoo...")

# 1. Queries
print(f"Using {len(queries)} queries loaded from BEIR.")

# 2. Qrels -> query_positive_docs
print("Processing qrels from BEIR...")
query_positive_docs = {}
for qid, doc_scores in qrels.items():
    positive_docs = {doc_id for doc_id, score in doc_scores.items() if score > 0}
    if positive_docs:
        query_positive_docs[qid] = positive_docs
print(f"Processed relevance information for {len(query_positive_docs)} queries.")

# 3. Load Chunks Data (!!! CẢNH BÁO BỘ NHỚ !!!)
print(f"Loading chunks data from {CHUNKS_OUTPUT_PATH}...")
chunks_data = {}
try:
    with open(CHUNKS_OUTPUT_PATH, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Loading Chunks"):
            try:
                chunk_info = json.loads(line)
                chunks_data[chunk_info['chunk_id']] = chunk_info['text']
            except json.JSONDecodeError: continue
    print(f"Loaded text for {len(chunks_data)} chunks.")
except FileNotFoundError: print(f"Error: Chunks file not found at {CHUNKS_OUTPUT_PATH}"); exit()
except Exception as e: print(f"An error occurred loading chunks data: {e}"); exit()

all_chunk_ids = list(chunks_data.keys())
if not all_chunk_ids: print("Error: No chunks loaded."); exit()

# 4. Map qid -> set(positive_chunk_id)
query_positive_chunks = {}
print("Mapping queries to positive chunk IDs...")
for qid, pos_doc_ids in tqdm(query_positive_docs.items(), desc="Mapping Positive Chunks"):
    query_positive_chunks[qid] = set()
    for doc_id in pos_doc_ids:
        if doc_id in doc_to_chunk_map:
            for chunk_id in doc_to_chunk_map[doc_id]:
                if chunk_id in chunks_data:
                    query_positive_chunks[qid].add(chunk_id)

# 5. Generate Triplets
print(f"Generating triplets and saving to {TRAIN_TRIPLETS_PATH}...")
triplets_count = 0
try:
    with open(TRAIN_TRIPLETS_PATH, 'w', encoding='utf-8') as f_train_out:
        query_ids = list(queries.keys())
        random.shuffle(query_ids)

        for qid in tqdm(query_ids, desc="Generating Triplets"):
            if qid not in queries or qid not in query_positive_chunks or not query_positive_chunks[qid]:
                continue

            query_text = clean_text(queries[qid])
            positive_chunk_ids_for_query = query_positive_chunks[qid]

            # Tạo một triplet cho mỗi positive chunk
            for positive_chunk_id in positive_chunk_ids_for_query:
                positive_chunk_text = chunks_data.get(positive_chunk_id)
                if not positive_chunk_text: continue

                # Sample negative chunk
                negative_chunk_id = None
                for _ in range(10): # Thử 10 lần để tìm negative
                    potential_negative_id = random.choice(all_chunk_ids)
                    if potential_negative_id not in positive_chunk_ids_for_query:
                        negative_chunk_id = potential_negative_id
                        break

                if negative_chunk_id:
                    negative_chunk_text = chunks_data.get(negative_chunk_id)
                    if negative_chunk_text:
                        # Ghi ra file theo định dạng MatchZoo (tab separated)
                        f_train_out.write(f"{query_text}\t{positive_chunk_text}\t{negative_chunk_text}\n")
                        triplets_count += 1

                        if MAX_TRIPLETS_TO_GENERATE is not None and triplets_count >= MAX_TRIPLETS_TO_GENERATE:
                            print(f"\nReached limit of {MAX_TRIPLETS_TO_GENERATE} triplets.")
                            raise StopIteration # Dừng vòng lặp ngoài

            if MAX_TRIPLETS_TO_GENERATE is not None and triplets_count >= MAX_TRIPLETS_TO_GENERATE:
                 break # Dừng hẳn nếu đã đủ triplets

except StopIteration:
    pass # Bắt lỗi StopIteration để kết thúc bình thường khi đủ triplets
except Exception as e:
    print(f"An error occurred generating triplets: {e}")

print(f"Finished generating {triplets_count} triplets.")
print(f"Training triplets saved to: {TRAIN_TRIPLETS_PATH}")

print("\n--- Steps 1, 2 and 3 (with Semantic Chunking and Triplet Generation for MatchZoo using BEIR) Completed ---")
print(f"Output file for MatchZoo training: {TRAIN_TRIPLETS_PATH}")
print("Next step: Use this TSV file to fine-tune your bi-encoder (retriever) model using MatchZoo (Step 4).")
