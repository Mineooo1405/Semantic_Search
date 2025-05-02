import pandas as pd
from Tool.Database import connect_to_db # Keep for potential future use, but not for chunking function
import json
import numpy as np
from dotenv import load_dotenv
import os
from Tool.Sentence_Detector import extract_and_simplify_sentences
# Use the standard embedding function
from Tool.Sentence_Embedding import sentence_embedding as embed_text_list
from Tool.OIE import extract_triples_for_search # Keep for potential future use
import matplotlib.pyplot as plt # Keep for potential future use
import seaborn as sns # Keep for potential future use
import time
import pickle # Keep for potential future use
import spacy # Keep for potential future use
import re
import psycopg2 # Keep for potential future use
from psycopg2 import extras as psycopg2_extras # Keep for potential future use
from pgvector.psycopg2 import register_vector # Keep for potential future use
import hashlib # Add hashlib
from typing import List, Tuple, Dict, Union # Add typing

load_dotenv()

def process_oie_in_groups(sentences, groups):
    """
    Trích xuất OpenIE triples cho tất cả các câu trong các nhóm
    
    Args:
        sentences: Danh sách các câu
        groups: Danh sách các nhóm, mỗi nhóm chứa các chỉ số của câu
        
    Returns:
        tuple: (all_triples, sentence_triples)
            - all_triples: Danh sách tất cả các triples đã trích xuất
            - sentence_triples: Danh sách các triples theo từng câu
    """
    print("\nĐang trích xuất quan hệ (OpenIE) cho các câu...")
    
    all_triples = []
    sentence_triples = [[] for _ in range(len(sentences))]  # Danh sách triples cho mỗi câu
    
    # Hiển thị tiến độ
    total = len(sentences)
    start_time = time.time()
    
    for i, sentence in enumerate(sentences):
        if i % 5 == 0 or i == total - 1:
            elapsed = time.time() - start_time
            eta = (elapsed / (i + 1)) * (total - i - 1) if i > 0 else 0
            print(f"Trích xuất quan hệ: [{i+1}/{total}] - {(i+1)/total*100:.1f}% - ETA: {eta:.1f}s", end="\r")
        
        # Trích xuất triples cho câu
        triples = extract_triples_for_search(sentence)
        
        # Lưu trữ triples
        sentence_triples[i] = triples
        all_triples.extend(triples)
    
    print(f"\nĐã trích xuất tổng cộng {len(all_triples)} quan hệ từ {total} câu")
    
    return all_triples, sentence_triples

def to_vectors(sentences, use_cache=True, cache_prefix="passage_vectors_"):
    cache_file = f"{cache_prefix}{hash(str(sentences))}.pkl"
    
    # Kiểm tra xem có thể sử dụng cache không
    if (use_cache and os.path.exists(cache_file)):
        print("Đang tải embedding vector từ cache")
        with open(cache_file, 'rb') as f:
            cached_data = pickle.load(f)
            if len(cached_data) == len(sentences):
                print("Đã tải vector embedding vector từ cache thành công!")
                return cached_data
            print("Số lượng vector trong cache không khớp với số câu, tính toán lại")
    
    vectors = []
    total = len(sentences)
    start_time = time.time()
    # Hiển thị tiến độ
    for i, sentence in enumerate(sentences):
        if i % 5 == 0 or i == total - 1:
            elapsed = time.time() - start_time
            eta = (elapsed / (i + 1)) * (total - i - 1) if i > 0 else 0
            print(f"Đang tạo vector [{i+1}/{total}] - {(i+1)/total*100:.1f}% - ETA: {eta:.1f}s", end="\r")
        
        vectors.append(embed_text_list(sentence))
    
    print("\nĐã hoàn thành tạo vector!")
    
    # Lưu vào cache cho lần sau
    if use_cache:
        print("Đang lưu vector vào cache")
        os.makedirs(os.path.dirname(cache_file) if os.path.dirname(cache_file) else '.', exist_ok=True)
        with open(cache_file, 'wb') as f:
            pickle.dump(vectors, f)
        print("Đã lưu vector vào cache!")
    
    return vectors

def create_semantic_matrix(vectors):
 
    vectors_array = np.array(vectors)
    n = len(vectors_array)
    similarity_matrix = np.zeros((n, n))

    # Hiển thị tiến độ tính toán
    total_comparisons = n * n
    completed = 0
    start_time = time.time()
    
    for i in range(n):
        vector_i = vectors_array[i]
        norm_i = np.sqrt(np.sum(vector_i ** 2))
        
        for j in range(n):
            vector_j = vectors_array[j]
            
            # Tính tích vô hướng (dot product)
            dot_product = np.sum(vector_i * vector_j)
            
            # Tính độ dài (norm) của vector j
            norm_j = np.sqrt(np.sum(vector_j ** 2))
            
            # Tránh chia cho 0
            if norm_i == 0 or norm_j == 0:
                similarity_matrix[i, j] = 0
            else:
                # Công thức tính cosine similarity
                similarity_matrix[i, j] = dot_product / (norm_i * norm_j)
            
            completed += 1
            if completed % (n * 5) == 0 or completed == total_comparisons:
                elapsed = time.time() - start_time
                progress = completed / total_comparisons
                eta = (elapsed / progress) * (1 - progress) if progress > 0 else 0
                print(f"Tiến độ: {completed}/{total_comparisons} ({progress*100:.1f}%) - ETA: {eta:.1f}s", end="\r")
    
    print(f"\nThống kê ma trận: min={similarity_matrix.min():.4f}, max={similarity_matrix.max():.4f}, mean={similarity_matrix.mean():.4f}")
    return similarity_matrix

def analyze_similarity_distribution(sim_matrix):
    """
    Phân tích phân bố độ tương đồng trong ma trận, bỏ qua giá trị 1.0.

    Args:
        sim_matrix: Ma trận numpy chứa độ tương đồng cosine.

    Returns:
        dict or None: Dictionary chứa các thống kê (min, max thực tế < 1, mean, std, percentiles)
                      hoặc None nếu không có đủ dữ liệu.
    """
    if not isinstance(sim_matrix, np.ndarray) or sim_matrix.ndim != 2 or sim_matrix.shape[0] < 2:
        # print("  Ma trận không hợp lệ hoặc quá nhỏ để phân tích.") # Optional: Keep if needed
        return None

    # Lấy các giá trị ở tam giác trên (không bao gồm đường chéo chính)
    upper_triangle_indices = np.triu_indices_from(sim_matrix, k=1)
    similarities = sim_matrix[upper_triangle_indices]

    # --- MODIFICATION START: Filter out values close to 1.0 ---
    # Lọc bỏ các giá trị rất gần hoặc bằng 1.0
    # Sử dụng một ngưỡng nhỏ để tránh vấn đề dấu phẩy động
    epsilon = 1e-5
    filtered_similarities = similarities[similarities < (1.0 - epsilon)]
    # --- MODIFICATION END ---

    if filtered_similarities.size == 0:
        # print("  Không có đủ cặp câu (với similarity < 1.0) để phân tích phân bố.") # Optional: Keep if needed
        # Có thể trả về giá trị mặc định hoặc None tùy theo cách xử lý ở nơi gọi
        # Nếu tất cả các cặp đều có sim = 1.0, có thể coi đây là một khối duy nhất
        # Trả về stats dựa trên giá trị gốc nếu muốn, hoặc None
        if similarities.size > 0: # Nếu có giá trị gốc nhưng tất cả đều là 1.0
             original_max = np.max(similarities) # Sẽ là 1.0
             return {
                'min': original_max, 'max': original_max, 'mean': original_max, 'std': 0.0,
                **{f'p{p}': original_max for p in [10, 25, 50, 75, 80, 85, 90, 95]}
             }
        return None # Trường hợp không có cặp nào ban đầu

    # Tính toán thống kê trên các giá trị đã lọc
    percentiles = {
        f'p{p}': np.percentile(filtered_similarities, p) for p in [10, 25, 50, 75, 80, 85, 90, 95]
    }
    stats = {
        'min': np.min(filtered_similarities),
        'max': np.max(filtered_similarities), # Max thực tế < 1.0
        'mean': np.mean(filtered_similarities),
        'std': np.std(filtered_similarities),
        **percentiles
    }

    # Các lệnh print đã được comment out ở bước trước
    # print("\nPhân bố độ tương đồng (giữa các câu khác nhau, sim < 1.0):")
    # ... (print logic) ...

    return stats

def semantic_spreading_grouping(sim_matrix, initial_threshold, decay_factor, min_threshold):
    num_sentences = sim_matrix.shape[0]
    ungrouped_indices = list(range(num_sentences))
    groups = []
    current_threshold = initial_threshold
    group_count = 1

    while len(ungrouped_indices) > 0:
        best_pair_score = -1
        best_pair = None

        # Tìm cặp có độ tương đồng cao nhất trong các câu chưa được nhóm
        possible_pairs = []
        for i in range(len(ungrouped_indices)):
            for j in range(i + 1, len(ungrouped_indices)):
                idx1 = ungrouped_indices[i]
                idx2 = ungrouped_indices[j]
                score = sim_matrix[idx1, idx2]
                if score >= current_threshold:
                    possible_pairs.append(((idx1, idx2), score))

        if not possible_pairs:
            current_threshold = max(min_threshold, current_threshold * decay_factor)
            if current_threshold == min_threshold and len(ungrouped_indices) > 0:
                 for idx in ungrouped_indices:
                     groups.append([idx])
                 ungrouped_indices = [] # Kết thúc vòng lặp
            continue # Thử lại với threshold mới hoặc kết thúc

        # Sắp xếp các cặp tìm được theo score giảm dần
        possible_pairs.sort(key=lambda x: x[1], reverse=True)
        best_pair, best_pair_score = possible_pairs[0]

        # Tạo nhóm mới bắt đầu từ cặp tốt nhất
        current_group = list(best_pair)
        ungrouped_indices.remove(best_pair[0])
        ungrouped_indices.remove(best_pair[1])

        # Mở rộng nhóm: Thêm các câu chưa nhóm có độ tương đồng đủ cao với BẤT KỲ câu nào trong nhóm hiện tại
        added_in_iteration = True
        while added_in_iteration:
            added_in_iteration = False
            indices_to_add = []
            remaining_ungrouped = list(ungrouped_indices) # Tạo bản sao để duyệt an toàn

            for ungrouped_idx in remaining_ungrouped:
                should_add = False
                max_sim_to_group = -1
                for group_member_idx in current_group:
                    similarity = sim_matrix[ungrouped_idx, group_member_idx]
                    max_sim_to_group = max(max_sim_to_group, similarity)
                    if similarity >= current_threshold:
                        should_add = True
                        break # Chỉ cần một câu trong nhóm thỏa mãn là đủ

                if should_add:
                    indices_to_add.append(ungrouped_idx)

            if indices_to_add:
                for idx_to_add in indices_to_add:
                    if idx_to_add in ungrouped_indices: # Kiểm tra lại phòng trường hợp đã bị thêm bởi logic khác
                        current_group.append(idx_to_add)
                        ungrouped_indices.remove(idx_to_add)
                        added_in_iteration = True

        groups.append(sorted(current_group))
        group_count += 1

        # Giảm threshold cho nhóm tiếp theo, nhưng không thấp hơn min_threshold
        current_threshold = max(min_threshold, initial_threshold * (decay_factor ** (group_count -1))) # Hoặc logic giảm khác

    return groups

def visualize_similarity_matrix(matrix, groups=None, title="Semantic Similarity Matrix"):
    """
    Hiển thị ma trận relationship với phân nhóm
    
    Args:
        matrix: Ma trận relationship
        groups: Danh sách các nhóm (optional)
        title: Tiêu đề của biểu đồ
    """
    plt.figure(figsize=(10, 8))
    
    # Vẽ heatmap với colorbar
    ax = sns.heatmap(matrix, annot=False, cmap='viridis')
    
    # Nếu có thông tin nhóm, hiển thị các đường biên nhóm
    if groups:
        group_boundaries = [0]
        for group in groups:
            group_boundaries.append(group_boundaries[-1] + len(group))
        
        # Vẽ các đường phân cách giữa các nhóm
        for boundary in group_boundaries[1:-1]:
            plt.axhline(y=boundary, color='r', linestyle='-')
            plt.axvline(x=boundary, color='r', linestyle='-')
    
    plt.title(title)
    plt.tight_layout()
    plt.savefig('similarity_matrix_with_groups.png')
    plt.show()

def display_oie_triples(groups, sentences, sentence_triples, max_display=3):
    """
    Hiển thị các triples được trích xuất từ các nhóm câu
    
    Args:
        groups: Các nhóm câu
        sentences: Danh sách các câu
        sentence_triples: Danh sách triples theo từng câu
        max_display: Số triples tối đa hiển thị cho mỗi câu
    """
    print("\n=== TRÍCH XUẤT QUAN HỆ THEO NHÓM ===")
    
    for i, group in enumerate(groups):
        print(f"\nNhóm {i+1}:")
        group_has_triples = False
        
        for idx in group:
            triples = sentence_triples[idx]
            if triples:
                group_has_triples = True
                sentence_preview = sentences[idx][:50] + "..." if len(sentences[idx]) > 50 else sentences[idx]
                print(f"  Câu {idx}: {sentence_preview}")
                
                # Hiển thị một số triples
                for j, triple in enumerate(triples[:max_display]):
                    print(f"    - {triple['subject']} | {triple['relation']} | {triple['object']}")
                
                if len(triples) > max_display:
                    print(f"    - ... và {len(triples) - max_display} quan hệ khác")
        
        if not group_has_triples:
            print("  Không có quan hệ nào được trích xuất trong nhóm này")

def save_to_database(query, passage, sentences, vectors, chunks):
    """Lưu kết quả phân nhóm vào database"""
    conn = connect_to_db()
    cursor = conn.cursor()
    
    # Chuyển đổi dữ liệu Python thành JSON để lưu trữ
    sentences_json = json.dumps([str(s) for s in sentences])
    
    # Đối với vectors, cần chuyển numpy arrays thành lists trước
    vectors_list = [v.tolist() for v in vectors]
    vectors_json = json.dumps(vectors_list)
    
    # Chuyển chunks thành JSON
    chunks_json = json.dumps([[int(idx) for idx in group] for group in chunks])
    
    try:
        cursor.execute(
            """
            INSERT INTO Semantic_Grouping 
            (query, Original_Paragraph, Sentences, Embedding_Vectors, Semantic_Chunking)
            VALUES (%s, %s, %s, %s, %s)
            RETURNING id
            """,
            (
                query,
                passage,
                sentences_json,
                vectors_json,
                chunks_json
            )
        )
        
        inserted_id = cursor.fetchone()[0]
        conn.commit()
        print(f"Đã lưu kết quả vào database với ID: {inserted_id}")
        
    except Exception as e:
        conn.rollback()
        print(f"Lỗi khi lưu vào database: {e}")
    finally:
        cursor.close()
        conn.close()

def save_to_database_with_oie(query, passage, sentences, vectors, chunks, all_triples, oie_sentence_groups):
    """Lưu kết quả phân nhóm và trích xuất OIE vào database"""
    conn = connect_to_db()
    cursor = conn.cursor()
    
    # Chuyển đổi dữ liệu Python thành JSON để lưu trữ
    sentences_json = json.dumps([str(s) for s in sentences])
    
    # Đối với vectors, cần chuyển numpy arrays thành lists trước
    vectors_list = [v.tolist() for v in vectors]
    vectors_json = json.dumps(vectors_list)
    
    # Chuyển chunks thành JSON
    chunks_json = json.dumps([[int(idx) for idx in group] for group in chunks])
    
    # JSON cho OpenIE triples
    all_triples_json = json.dumps(all_triples)
    oie_sentence_groups_json = json.dumps(oie_sentence_groups)
    
    try:
        cursor.execute(
            """
            INSERT INTO Semantic_Grouping 
            (query, Original_Paragraph, Sentences, Embedding_Vectors, Semantic_Chunking, OIE_Triples, OIE_Sentence_Groups)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            RETURNING id
            """,
            (
                query,
                passage,
                sentences_json,
                vectors_json,
                chunks_json,
                all_triples_json,
                oie_sentence_groups_json
            )
        )
        
        inserted_id = cursor.fetchone()[0]
        conn.commit()
        print(f"Đã lưu kết quả vào database với ID: {inserted_id}")
        
    except Exception as e:
        conn.rollback()
        print(f"Lỗi khi lưu vào database: {e}")
    finally:
        cursor.close()
        conn.close()

def test_document():
    print("\n=== PHÂN NHÓM CHO TÀI LIỆU ===")
    print("1. Nhập tài liệu trực tiếp")
    print("2. Tải tài liệu từ file văn bản")
    
    sub_choice = input("Lựa chọn của bạn (1/2): ")
    
    document = ""
    if sub_choice == '1':
        # Nhập tài liệu trực tiếp
        print("\nNhập nội dung tài liệu (kết thúc bằng một dòng chỉ có '###'):")
        lines = []
        while True:
            line = input()
            if line == '###':
                break
            lines.append(line)
        
        if not lines:
            print("Không có nội dung được nhập, hủy phân tích.")
            return
        
        document = "\n".join(lines)
        
    elif sub_choice == '2':
        # Tải tài liệu từ file
        file_path = input("\nNhập đường dẫn đến file tài liệu: ")
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                document = file.read()
            print(f"Đã tải thành công tài liệu từ {file_path}")
        except Exception as e:
            print(f"Lỗi khi đọc file: {e}")
            return
    else:
        print("Lựa chọn không hợp lệ.")
        return
    
    # Hiển thị thông tin cơ bản về tài liệu
    print(f"\nThông tin tài liệu:")
    print(f"- Kích thước: {len(document)} ký tự")
    print(f"- Số dòng: {document.count(chr(10)) + 1}")
    
    query = input("Nhập query hoặc chủ đề tài liệu (có thể để trống): ")
    
    # Xử lý tài liệu nhưng không lưu vào database
    process_document(document, "Tài liệu thủ công", query=query, save_to_db=False)

def process_document(document, document_id, query="", visualize=True, save_to_db=True, extract_oie=True):
    """
    Xử lý một tài liệu/đoạn văn: phát hiện câu, nhúng, phân nhóm, trích xuất OIE và lưu kết quả
    
    Args:
        document: Văn bản cần xử lý
        document_id: ID của tài liệu/đoạn văn
        query: Query hoặc mô tả
        visualize: Có trực quan hóa ma trận similarity hay không
        save_to_db: Có lưu kết quả vào database hay không
        extract_oie: Có trích xuất OIE triples hay không
        
    Returns:
        tuple: (sentences, vectors, groups, all_triples)
    """
    print(f"\n--- Đang xử lý {document_id} ---")
    
    # Bước 1: Tách câu
    sentences = extract_and_simplify_sentences(document, simplify=True)
    print(f"Đã tách thành {len(sentences)} câu tối ưu cho OIE")
    
    # Nếu không có câu nào, trả về kết quả rỗng
    if not sentences:
        print("Không có câu nào trong văn bản, bỏ qua")
        return [], [], [], []
    
    # Bước 2: Tạo vector nhúng
    vectors = to_vectors(sentences)
    
    # Bước 3: Tạo ma trận similarity
    sim_matrix = create_semantic_matrix(vectors)
    
    # Bước 4: Phân tích phân bố relationship và đưa ra đề xuất threshold
    percentiles = analyze_similarity_distribution(sim_matrix, sentences)
    
    # Bước 5: Yêu cầu người dùng chọn các tham số
    while True:
        try:
            initial_threshold = float(input(f"\nNhập threshold relationship ban đầu (0.0-1.0, đề xuất: {percentiles['75%']:.4f}): ") or percentiles['75%'])
            if 0 <= initial_threshold <= 1:
                break
            else:
                print("threshold phải nằm trong khoảng từ 0 đến 1.")
        except ValueError:
            print("Vui lòng nhập một số thực.")
    
    while True:
        try:
            decay_factor = float(input("Nhập hệ số giảm threshold (0.7-0.95, đề xuất: 0.9): ") or "0.9")
            if 0.7 <= decay_factor <= 0.95:
                break
            else:
                print("Hệ số giảm nên nằm trong khoảng từ 0.7 đến 0.95.")
        except ValueError:
            print("Vui lòng nhập một số thực.")
    
    while True:
        try:
            min_threshold = float(input(f"Nhập threshold tối thiểu (0.1-0.5, đề xuất: {percentiles['25%']:.4f}): ") or percentiles['25%'])
            if 0.1 <= min_threshold <= 0.5:
                break
            else:
                print("threshold tối thiểu nên nằm trong khoảng từ 0.1 đến 0.5.")
        except ValueError:
            print("Vui lòng nhập một số thực.")
    
    # Bước 6: Phân nhóm câu với threshold giảm dần
    groups = semantic_spreading_grouping(sim_matrix, initial_threshold, decay_factor, min_threshold)
    print(f"Đã phân thành {len(groups)} nhóm ")
    
    # Bước 7: Hiển thị các nhóm câu
    for i, group in enumerate(groups):
        group_sentences = [sentences[idx] for idx in group]
        print(f"\nNhóm {i+1} ({len(group)} câu):")
        for s in group_sentences[:3]:  # Hiển thị tối đa 3 câu đầu tiên
            print(f" - {s[:100]}..." if len(s) > 100 else f" - {s}")
        if len(group_sentences) > 3:
            print(f" - ... và {len(group_sentences)-3} câu khác")
    
    # Bước 8: Trích xuất OIE triples
    all_triples = []
    sentence_triples = []
    
    if extract_oie:
        all_triples, sentence_triples = process_oie_in_groups(sentences, groups)
        print(f"\n[SUCCESS] Đã trích xuất {len(all_triples)} quan hệ")
    
    # Bước 9: Hiển thị OIE triples theo nhóm
    if extract_oie and all_triples:
        display_oie_triples(groups, sentences, sentence_triples)
    
    # Bước 10: Trực quan hóa ma trận
    if visualize:
        visualize_similarity_matrix(sim_matrix, groups, 
                                  title=f"Ma trận similarity - {document_id}")
    
    # Bước 11: Xuất kết quả ra file
    export_results_to_file(document, sentences, groups, sentence_triples, document_id)
    
    # Bước 12: Lưu kết quả vào database nếu được yêu cầu
    if save_to_db:
        if extract_oie:
            # Tạo cấu trúc OIE_Sentence_Groups
            oie_sentence_groups = []
            for group in groups:
                group_triples = []
                for idx in group:
                    group_triples.append(sentence_triples[idx])
                oie_sentence_groups.append(group_triples)
                
            save_to_database_with_oie(query, document, sentences, vectors, groups, all_triples, oie_sentence_groups)
        else:
            save_to_database(query, document, sentences, vectors, groups)
    
    return sentences, vectors, groups, all_triples

def generate_output_filename(document_id=None, prefix="semantic", suffix=".txt"):
    """
    Tạo tên file chuẩn hóa và nhất quán cho các kết quả xuất ra
    
    Args:
        document_id: ID của tài liệu (có thể None)
        prefix: Tiền tố xác định loại phân tích ("grouping", "splitter")
        suffix: Hậu tố file (.txt, .json, etc.)
        
    Returns:
        str: Tên file đã được chuẩn hóa với timestamp
    """
    # Chuẩn hóa document_id
    if document_id is None:
        document_id = f"doc_{int(time.time())}"
    
    # Làm sạch document_id, loại bỏ các ký tự không hợp lệ
    safe_id = re.sub(r'[^a-zA-Z0-9_-]', '_', str(document_id).replace(' ', '_'))
    
    # Thêm thông tin thời gian
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    # Tạo tên file với định dạng nhất quán
    return f"{prefix}_{safe_id}_{timestamp}{suffix}"

def export_results_to_file(document, sentences, groups, sentence_triples=None, document_id=None):
    """
    Xuất kết quả phân nhóm và OIE ra file văn bản
    
    Args:
        document: Văn bản gốc
        sentences: Danh sách các câu
        groups: Các nhóm câu
        sentence_triples: Danh sách triples theo từng câu (tùy chọn)
        document_id: ID của tài liệu
    """
    if document_id is None:
        document_id = f"document_{int(time.time())}"
        
    filename = generate_output_filename(document_id, prefix="grouping")
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(f"KẾT QUẢ PHÂN NHÓM VÀ TRÍCH XUẤT QUAN HỆ (OIE)\n")
            f.write(f"Tài liệu: {document_id}\n")
            f.write(f"Thời gian: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write(f"Tổng số câu: {len(sentences)}\n")
            f.write(f"Số nhóm: {len(groups)}\n\n")
            
            # Đếm tổng số triples nếu có
            if sentence_triples:
                total_triples = sum(len(triples) for triples in sentence_triples)
                f.write(f"Tổng số quan hệ trích xuất: {total_triples}\n\n")
            
            for i, group in enumerate(groups):
                f.write(f"=== NHÓM {i+1} ({len(group)} câu) ===\n")
                
                # Tính số triples trong nhóm nếu có
                if sentence_triples:
                    group_triples_count = sum(len(sentence_triples[idx]) for idx in group)
                    f.write(f"Số quan hệ trong nhóm: {group_triples_count}\n\n")
                
                for idx in group:
                    f.write(f"[{idx}] {sentences[idx]}\n")
                    
                    # Ghi các triples của câu nếu có
                    if sentence_triples:
                        triples = sentence_triples[idx]
                        if triples:
                            f.write(f"  Quan hệ trích xuất ({len(triples)}):\n")
                            for triple in triples:
                                f.write(f"  - {triple['subject']} | {triple['relation']} | {triple['object']}\n")
                        else:
                            f.write("  Không có quan hệ được trích xuất\n")
                    f.write("\n")
                
                f.write("\n")
        
        print(f"\nĐã xuất kết quả chi tiết ra file: {filename}")
    except Exception as e:
        print(f"Lỗi khi xuất kết quả ra file: {e}")

# --- ADD THE NEW CHUNKING FUNCTION ---
def semantic_chunk_passage_from_grouping_logic(
    doc_id: str,
    passage_text: str,
    model_name: str = "thenlper/gte-large",
    initial_threshold: Union[float, str] = 'auto',
    decay_factor: float = 0.85,
    min_threshold: Union[float, str] = 'auto',
    auto_percentiles: Tuple[int, int] = (85, 25),
    **kwargs # Catch unused args
) -> List[Tuple[str, str]]:
    """
    Chunks a passage using the Semantic Grouping logic (spreading grouping).
    Returns a list of (chunk_id, chunk_text).
    """
    chunks_result = []
    try:
        # a. Split sentences
        sentences = extract_and_simplify_sentences(passage_text, simplify=False)
        if not sentences: return []
        if len(sentences) == 1: # Handle single sentence passage
            chunk_text = sentences[0]
            chunk_hash = hashlib.md5(chunk_text.encode()).hexdigest()[:10]
            chunk_id = f"{doc_id}_{chunk_hash}"
            return [(chunk_id, chunk_text)]

        # b. Embed sentences
        sentence_vectors = embed_text_list(sentences, model_name=model_name)
        if sentence_vectors is None: return []

        # c. Create similarity matrix
        sim_matrix = create_semantic_matrix(sentence_vectors)

        # d. Determine thresholds (if 'auto')
        current_initial_threshold = initial_threshold
        current_min_threshold = min_threshold

        if initial_threshold == 'auto' or min_threshold == 'auto':
            percentiles = analyze_similarity_distribution(sim_matrix) # Don't pass sentences here
            if not percentiles:
                 current_initial_threshold = 0.8 # Default fallback
                 current_min_threshold = 0.2   # Default fallback
                 print("  WARNING: Could not analyze distribution, using default thresholds (0.8, 0.2)")
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

                    # Ensure min < initial, adjust if necessary
                    if current_min_threshold >= current_initial_threshold:
                        print(f"  WARNING: Auto min threshold ({current_min_threshold:.4f}) >= initial ({current_initial_threshold:.4f}). Adjusting min.")
                        # Try a lower percentile or a fraction of initial
                        lower_min_key = f'p{max(10, min_p - 15)}'
                        if lower_min_key in percentiles:
                            current_min_threshold = percentiles[lower_min_key]
                        else:
                            current_min_threshold = current_initial_threshold * 0.5
                        current_min_threshold = max(0.1, current_min_threshold) # Ensure not too low
                        print(f"  Adjusted min threshold: {current_min_threshold:.4f}")

                except Exception as auto_err:
                    print(f"  ERROR determining auto thresholds: {auto_err}. Using defaults (0.8, 0.2).")
                    current_initial_threshold = 0.8 if initial_threshold == 'auto' else float(initial_threshold)
                    current_min_threshold = 0.2 if min_threshold == 'auto' else float(min_threshold)
        # Ensure thresholds are float if provided as strings initially
        elif isinstance(initial_threshold, str): current_initial_threshold = float(initial_threshold)
        elif isinstance(min_threshold, str): current_min_threshold = float(min_threshold)

        # Final check min < initial
        if current_min_threshold >= current_initial_threshold:
             print(f"  WARNING: Final check failed: min ({current_min_threshold:.4f}) >= initial ({current_initial_threshold:.4f}). Setting min = initial * 0.5")
             current_min_threshold = max(0.1, current_initial_threshold * 0.5)

        # e. Perform semantic grouping
        groups = semantic_spreading_grouping(
            sim_matrix,
            current_initial_threshold,
            decay_factor,
            current_min_threshold
        )

        # f. Create chunks from groups
        for group_idx, group in enumerate(groups):
            chunk_sentences = [sentences[sent_idx] for sent_idx in group]
            chunk_text = " ".join(chunk_sentences).strip()
            if chunk_text:
                chunk_hash = hashlib.md5(chunk_text.encode()).hexdigest()[:10]
                chunk_id = f"{doc_id}_{chunk_hash}_{group_idx}" # Add group_idx for uniqueness
                chunks_result.append((chunk_id, chunk_text))

    except Exception as e:
        print(f"Error chunking doc {doc_id} with Semantic Grouping: {e}")
    return chunks_result

