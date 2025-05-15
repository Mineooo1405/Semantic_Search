import os
import re
import nltk
import time
import json
import pandas as pd
from typing import List, Dict, Union, Optional, Callable, Tuple # Sửa lại type hint cho hàm cuối
from dotenv import load_dotenv
from Tool.Sentence_Detector import extract_and_simplify_sentences
from Tool.Database import connect_to_db
# from Tool.OIE import extract_triples_for_search
import hashlib
import traceback # Thêm import này

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

load_dotenv()

def to_sentences(passage):
    sub_sentences = extract_and_simplify_sentences(passage, simplify=True)
    return sub_sentences

def calculate_chunk_stats(chunks):
    stats = {
        'count': len(chunks),
        'avg_length': sum(len(c) for c in chunks) / len(chunks) if chunks else 0,
        'min_length': min(len(c) for c in chunks) if chunks else 0,
        'max_length': max(len(c) for c in chunks) if chunks else 0
    }
    return stats

def split_by_sentence(text, chunk_size=1000, chunk_overlap=200):
    sentences = to_sentences(text)
    # Bỏ print không cần thiết ở đây, vì nó có thể gây nhiễu log khi chạy song song
    # print(f"[INFO] Đã tách thành {len(sentences)} câu") 
    
    if not sentences:
        return [], [], []
    
    if len(text) <= chunk_size:
        sentence_indices = list(range(len(sentences)))
        return [text], sentences, [sentence_indices]
    
    sentence_lengths = [len(s) for s in sentences]
    chunks = []
    current_chunk_sentences = []
    current_size = 0
    sentence_groups = []
    current_group_indices = []
    
    for i, sentence in enumerate(sentences):
        sentence_len = sentence_lengths[i]
        
        if sentence_len > chunk_size:
            if current_chunk_sentences:
                chunks.append(' '.join(current_chunk_sentences))
                sentence_groups.append(list(current_group_indices)) # Tạo copy
                current_chunk_sentences = []
                current_group_indices = []
                current_size = 0
            chunks.append(sentence)
            sentence_groups.append([i])
            continue
        
        if current_size + sentence_len + (1 if current_chunk_sentences else 0) > chunk_size and current_chunk_sentences:
            chunks.append(' '.join(current_chunk_sentences))
            sentence_groups.append(list(current_group_indices)) # Tạo copy
            
            if chunk_overlap > 0 and len(current_chunk_sentences) > 1:
                overlap_char_count = 0
                temp_overlap_sentences = []
                temp_overlap_indices = []
                
                for sent_idx_in_group, original_sent_idx in reversed(list(enumerate(current_group_indices))):
                    s = current_chunk_sentences[sent_idx_in_group]
                    s_len = len(s)
                    if overlap_char_count + s_len + (1 if temp_overlap_sentences else 0) > chunk_overlap and temp_overlap_sentences:
                        break
                    temp_overlap_sentences.insert(0, s)
                    temp_overlap_indices.insert(0, original_sent_idx)
                    overlap_char_count += s_len + (1 if len(temp_overlap_sentences) > 1 else 0)
                
                if temp_overlap_sentences and temp_overlap_sentences != current_chunk_sentences: # Đảm bảo temp_overlap_sentences không rỗng
                    current_chunk_sentences = temp_overlap_sentences
                    current_group_indices = temp_overlap_indices
                    current_size = overlap_char_count - (1 if overlap_char_count > 0 and len(current_chunk_sentences) > 1 else 0)
                else:
                    current_chunk_sentences = []
                    current_group_indices = []
                    current_size = 0
            else:
                current_chunk_sentences = []
                current_group_indices = []
                current_size = 0
        
        current_chunk_sentences.append(sentence)
        current_group_indices.append(i)
        current_size += sentence_len + (1 if len(current_chunk_sentences) > 1 else 0)
    
    if current_chunk_sentences:
        chunks.append(' '.join(current_chunk_sentences))
        sentence_groups.append(list(current_group_indices)) # Tạo copy
    
    # Bỏ print không cần thiết ở đây
    # stats = calculate_chunk_stats(chunks)
    # if stats['count'] > 0:
    #     print(f"[INFO] Chia theo câu: tạo {stats['count']} chunk")
    #     print(f"[INFO] Độ dài trung bình: {stats['avg_length']:.1f} ký tự")
    # else:
    #     print(f"[INFO] Chia theo câu: không tạo được chunk nào.")
        
    return chunks, sentences, sentence_groups

def chunk_passage_text_splitter(
    doc_id: str,
    passage_text: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    include_oie: bool = False,
    **kwargs 
) -> List[List[Tuple[str, str, Optional[str]]]]: # <<< SỬA ĐỔI TYPE HINT
    """
    Chunk một passage sử dụng logic Text Splitter (split_by_sentence).
    Trả về một list chứa một list các chunk tuples.
    """
    actual_chunks_result: List[Tuple[str, str, Optional[str]]] = []
    try:
        chunk_texts, _, _ = split_by_sentence(passage_text, chunk_size, chunk_overlap)

        for chunk_idx, chunk_text_content in enumerate(chunk_texts):
            if chunk_text_content:
                chunk_hash = hashlib.md5(chunk_text_content.encode('utf-8')).hexdigest()[:10]
                chunk_id = f"{doc_id}_{chunk_hash}_{chunk_idx}"
                
                oie_triples_str = None
                if include_oie:
                    # Nếu bạn có logic OIE cho TextSplitter, hãy thêm vào đây.
                    # Ví dụ:
                    # from Tool.OIE import extract_triples # Đảm bảo import đúng
                    # oie_triples_str = extract_triples(chunk_text_content, doc_id, chunk_id) 
                    pass 

                actual_chunks_result.append((chunk_id, chunk_text_content, oie_triples_str))

    except Exception as e:
        # print(f"Error chunking doc {doc_id} with Text Splitter: {e}") # Log này có thể đã có ở tầng cao hơn
        # traceback.print_exc()
        # Trả về một cấu trúc rỗng nhưng vẫn đúng định dạng để không làm hỏng quá trình unpack
        return [[]] # Trả về list chứa một list rỗng

    # Luôn trả về một list chứa list các chunk (actual_chunks_result)
    # Nếu actual_chunks_result rỗng, nó sẽ là [[]]
    # Nếu actual_chunks_result có chunk, nó sẽ là [[(chunk1_data), (chunk2_data)]]
    return [actual_chunks_result] # <<< SỬA ĐỔI QUAN TRỌNG

