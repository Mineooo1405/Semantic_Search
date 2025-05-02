import os
import re
import nltk
import time
import json
import pandas as pd
from typing import List, Dict, Union, Optional, Callable, Tuple
from dotenv import load_dotenv
from Tool.Sentence_Detector import extract_and_simplify_sentences
from Tool.Database import connect_to_db
from Tool.OIE import extract_triples_for_search
import hashlib

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

load_dotenv()

def to_sentences(passage):
    """
    Tách đoạn văn thành các câu riêng biệt tối ưu cho OIE
    
    Args:
        passage: Đoạn văn bản cần tách câu
        
    Returns:
        list: Danh sách các câu đã được chia nhỏ
    """
    # Lấy ra câu đã chia nhỏ (phần tử thứ 2 của tuple)
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
    """
    Chia văn bản thành các đoạn dựa trên ranh giới câu
    
    Args:
        text: Văn bản cần chia
        chunk_size: Kích thước tối đa của mỗi chunk (ký tự)
        chunk_overlap: Số câu chồng lấp giữa các chunk liên tiếp
        
    Returns:
        list: Danh sách các chunk văn bản
        list: Danh sách các câu gốc
    """
    # Tách câu
    sentences = to_sentences(text)
    print(f"[INFO] Đã tách thành {len(sentences)} câu")
    
    if not sentences:
        return [], []
    
    if len(text) <= chunk_size:
        return [text], sentences
    
    # Tính kích thước của các câu
    sentence_lengths = [len(s) for s in sentences]
    
    chunks = []
    current_chunk = []
    current_size = 0
    
    # Danh sách để theo dõi các câu thuộc từng chunk
    sentence_groups = []
    current_group = []
    
    for i, sentence in enumerate(sentences):
        sentence_len = sentence_lengths[i]
        
        # Nếu câu hiện tại quá dài, chia nhỏ câu đó
        if sentence_len > chunk_size:
            # Xử lý chunk hiện tại trước
            if current_chunk:
                chunks.append(' '.join(current_chunk))
                sentence_groups.append(current_group)
                current_chunk = []
                current_group = []
                current_size = 0
            
            # Đối với câu dài, tạo chunk riêng và ghi nhận chỉ có một câu
            chunks.append(sentence)
            sentence_groups.append([i])
            continue
        
        # Nếu thêm câu hiện tại vào chunk sẽ vượt quá kích thước tối đa
        if current_size + sentence_len + 1 > chunk_size and current_chunk:  # +1 cho khoảng trắng
            chunks.append(' '.join(current_chunk))
            sentence_groups.append(current_group)
            
            # Tính toán số câu cần giữ lại cho overlap
            if chunk_overlap > 0:
                # Số câu chồng lấp dựa trên kích thước
                overlap_size = 0
                overlap_sentences = []
                overlap_indices = []
                
                # Đảo ngược để lấy từ cuối lên
                for idx, s in reversed(list(zip(current_group, current_chunk))):
                    s_len = len(s)
                    if overlap_size + s_len > chunk_overlap:
                        break
                    overlap_sentences.insert(0, s)
                    overlap_indices.insert(0, idx)
                    overlap_size += s_len + 1  # +1 cho khoảng trắng
                
                current_chunk = overlap_sentences
                current_group = overlap_indices
                current_size = overlap_size
            else:
                current_chunk = []
                current_group = []
                current_size = 0
        
        # Thêm câu hiện tại vào chunk
        current_chunk.append(sentence)
        current_group.append(i)
        current_size += sentence_len + 1  # +1 cho khoảng trắng
    
    # Thêm chunk cuối cùng nếu có
    if current_chunk:
        chunks.append(' '.join(current_chunk))
        sentence_groups.append(current_group)
    
    # Hiển thị thống kê
    stats = calculate_chunk_stats(chunks)
    print(f"[INFO] Chia theo câu: tạo {stats['count']} chunk")
    print(f"[INFO] Độ dài trung bình: {stats['avg_length']:.1f} ký tự")
    
    return chunks, sentences, sentence_groups

def chunk_passage_text_splitter(
    doc_id: str,
    passage_text: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    **kwargs # Bắt các tham số không dùng đến
) -> List[Tuple[str, str]]:
    """
    Chunk một passage sử dụng logic Text Splitter (split_by_sentence).

    Args:
        doc_id (str): ID của document gốc.
        passage_text (str): Nội dung của passage.
        chunk_size (int): Kích thước chunk tối đa (ký tự).
        chunk_overlap (int): Số ký tự chồng lấp (ước lượng qua câu).

    Returns:
        List[Tuple[str, str]]: Danh sách các tuple (chunk_id, chunk_text).
    """
    chunks_result = []
    try:
        # a. Chia văn bản thành các chunk dựa trên câu
        # Hàm split_by_sentence trả về (chunks, sentences, sentence_groups)
        # Chúng ta chỉ cần chunks ở đây
        chunks, _, _ = split_by_sentence(passage_text, chunk_size, chunk_overlap)

        # b. Tạo chunk_id và định dạng kết quả
        for chunk_idx, chunk_text in enumerate(chunks):
            if chunk_text:
                chunk_hash = hashlib.md5(chunk_text.encode()).hexdigest()[:10]
                chunk_id = f"{doc_id}_{chunk_hash}_{chunk_idx}" # Thêm chunk_idx để tránh trùng hash
                chunks_result.append((chunk_id, chunk_text))

    except Exception as e:
        print(f"Error chunking doc {doc_id} with Text Splitter: {e}")
    return chunks_result
