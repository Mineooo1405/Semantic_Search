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
from Tool.OIE import extract_relations_from_paragraph # MODIFIED OIE import
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
    device: Optional[object] = None, # Added device, though not used by this splitter
    **kwargs 
) -> List[Tuple[str, str, Optional[str]]]: # MODIFIED: Return type to match others
    """
    Chunk một passage sử dụng logic Text Splitter (split_by_sentence).
    Trả về một list các chunk tuples (chunk_id, chunk_text, oie_string_or_None).
    Device parameter is accepted for consistency but not used as this splitter is rule-based.
    """
    actual_chunks_result: List[Tuple[str, str, Optional[str]]] = []
    # Log that device is not used if provided, for clarity
    if device is not None:
        print(f"  TextSplitter: Device parameter provided but not used by this rule-based splitter for doc_id {doc_id}.")
    print(f"  TextSplitter: Processing doc_id {doc_id}...")
    print(f"  TextSplitter: Chunk size: {chunk_size}, Chunk overlap: {chunk_overlap}")
    print(f"  TextSplitter: Include OIE: {include_oie}")

    try:
        chunk_texts, _, _ = split_by_sentence(passage_text, chunk_size, chunk_overlap)

        for chunk_idx, chunk_text_content in enumerate(chunk_texts):
            chunk_id = f"{doc_id}_textsplit_{chunk_idx}"
            oie_string = None
            if include_oie and chunk_text_content.strip():
                try:
                    relations = extract_relations_from_paragraph(chunk_text_content, use_enhanced_settings=True)
                    if relations:
                        # Need format_oie_triples_to_string here or import it
                        oie_string = format_oie_triples_to_string_for_text_splitter(relations) # Call local version
                except Exception as e_oie:
                    print(f"Error during OIE for text_splitter chunk {chunk_id}: {e_oie}")
            
            actual_chunks_result.append((chunk_id, chunk_text_content, oie_string))

    except Exception as e:
        print(f"Error chunking doc {doc_id} with Text Splitter: {e}")
        # traceback.print_exc()
        # Return empty list in case of error, but matching the expected type
        return [] 

    return actual_chunks_result # MODIFIED: Return List directly

# ADD HELPER FOR OIE STRING FORMATTING (specific to this file to avoid import issues)
def format_oie_triples_to_string_for_text_splitter(triples_list: List[Dict[str, str]]) -> str:
    if not triples_list:
        return ""
    formatted_triples = []
    for triple in triples_list:
        s = str(triple.get('subject', '')).replace('\\t', ' ').replace('\\n', ' ').strip()
        r = str(triple.get('relation', '')).replace('\\t', ' ').replace('\\n', ' ').strip()
        o = str(triple.get('object', '')).replace('\\t', ' ').replace('\\n', ' ').strip()
        if s and r and o:
            formatted_triples.append(f"({s}; {r}; {o})")
    if not formatted_triples:
        return ""
    return " [OIE_TRIPLES] " + " | ".join(formatted_triples) + " [/OIE_TRIPLES]"

