import pandas as pd
from Tool.Database import connect_to_db
import json
import numpy as np
from dotenv import load_dotenv
import os
from Tool.Sentence_Detector import sentence_detector, enhanced_sentence_detector, spacy_sentence_splitter,to_sentences
from Tool.Sentence_Embedding import sentence_embedding
from Tool.OIE import extract_triples_for_search
import re
import matplotlib.pyplot as plt
import seaborn as sns
import time
import pickle
import spacy

load_dotenv()

def to_vectors(sentences, use_cache=True, cache_prefix="passage_vectors_"):
    cache_file = f"{cache_prefix}{hash(str(sentences))}.pkl"
    
    # Kiểm tra xem có thể sử dụng cache không
    if (use_cache and os.path.exists(cache_file)):
        print("Đang tải embedding vector từ cache...")
        with open(cache_file, 'rb') as f:
            cached_data = pickle.load(f)
            if len(cached_data) == len(sentences):
                print("Đã tải vector embedding vector từ cache thành công!")
                return cached_data
            print("Số lượng vector trong cache không khớp với số câu, tính toán lại...")
    
    vectors = []
    total = len(sentences)
    start_time = time.time()
    # Hiển thị tiến độ
    for i, sentence in enumerate(sentences):
        if i % 5 == 0 or i == total - 1:
            elapsed = time.time() - start_time
            eta = (elapsed / (i + 1)) * (total - i - 1) if i > 0 else 0
            print(f"Đang tạo vector [{i+1}/{total}] - {(i+1)/total*100:.1f}% - ETA: {eta:.1f}s", end="\r")
        
        vectors.append(sentence_embedding(sentence))
    
    print("\nĐã hoàn thành tạo vector!")
    
    # Lưu vào cache cho lần sau
    if use_cache:
        print("Đang lưu vector vào cache...")
        os.makedirs(os.path.dirname(cache_file) if os.path.dirname(cache_file) else '.', exist_ok=True)
        with open(cache_file, 'wb') as f:
            pickle.dump(vectors, f)
        print("Đã lưu vector vào cache!")
    
    return vectors

def create_semantic_matrix(vectors):
    """Tạo ma trận dựa trên cosine similarity"""
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

def analyze_similarity_distribution(similarity_matrix, sentences=None):
    """Phân tích phân bố relationship và đưa ra đề xuất threshold"""
    n = len(similarity_matrix)
    similarity_pairs = []
    
    # Lấy các giá trị relationship của các cặp khác nhau (loại bỏ đường chéo)
    for i in range(n):
        for j in range(i+1, n):  # Chỉ lấy nửa trên của ma trận (không tính đường chéo)
            similarity_pairs.append((i, j, similarity_matrix[i][j]))
    
    # Sắp xếp các cặp câu theo relationship giảm dần
    similarity_pairs.sort(key=lambda x: x[2], reverse=True)
    
    # Hiển thị các cặp câu và relationship của chúng
    print("\nCác cặp câu và relationship (sắp xếp theo relationship giảm dần):")
    for i, j, sim in similarity_pairs[:20]:  # Hiển thị 20 cặp đầu tiên
        if sentences:
            sentence_i = sentences[i].replace("\n", " ").strip()
            sentence_j = sentences[j].replace("\n", " ").strip()
            
            sentence_i = sentences[i][:30] + "..." if len(sentences[i]) > 30 else sentences[i]
            sentence_j = sentences[j][:30] + "..." if len(sentences[j]) > 30 else sentences[j]
            print(f"Câu {i} và câu {j}: {sim:.4f} - [{sentence_i}] và [{sentence_j}]")
        else:
            print(f"Câu {i} và câu {j}: {sim:.4f}")
            
    if len(similarity_pairs) > 20:
        print(f"... và {len(similarity_pairs)-20} cặp khác")

    # Chuyển danh sách relationship thành mảng numpy để phân tích
    similarities = np.array([sim for _, _, sim in similarity_pairs])
    
    # Phân tích phân bố
    percentiles = {
        '10%': np.percentile(similarities, 10),
        '25%': np.percentile(similarities, 25),
        'median': np.median(similarities),
        '75%': np.percentile(similarities, 75),
        '90%': np.percentile(similarities, 90)
    }

    # Đề xuất threshold
    print("\nĐề xuất threshold relationship (threshold):")
    print(f"  threshold nghiêm ngặt (ít nhóm hơn): {percentiles['90%']:.4f}")
    print(f"  threshold cân bằng: {percentiles['75%']:.4f}")
    print(f"  threshold thoải mái (nhiều nhóm hơn): {percentiles['median']:.4f}")
    
    return percentiles

def get_parameters_from_user(percentiles):
    """
    Lấy các tham số từ người dùng
    
    Args:
        percentiles: Kết quả phân tích phân bố similarity
        
    Returns:
        tuple: (initial_threshold, decay_factor, min_threshold, min_chunk_len, max_chunk_len)
    """
    # Đề xuất giá trị mặc định từ phân tích
    suggested_threshold = percentiles['75%']
    
    while True:
        try:
            initial_threshold = float(input(f"\nNhập threshold relationship ban đầu (0.0-1.0, đề xuất: {suggested_threshold:.4f}): ") or suggested_threshold)
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
    
    while True:
        try:
            min_chunk_len = int(input("Nhập độ dài tối thiểu của đoạn (1-5, đề xuất: 2): ") or "2")
            if 1 <= min_chunk_len <= 5:
                break
            else:
                print("Độ dài tối thiểu nên nằm trong khoảng từ 1 đến 5.")
        except ValueError:
            print("Vui lòng nhập một số nguyên.")
            
    while True:
        try:
            max_chunk_len = int(input("Nhập độ dài tối đa của đoạn (5-15, đề xuất: 8): ") or "8")
            if 5 <= max_chunk_len <= 15:
                break
            else:
                print("Độ dài tối đa nên nằm trong khoảng từ 5 đến 15.")
        except ValueError:
            print("Vui lòng nhập một số nguyên.")
    
    return initial_threshold, decay_factor, min_threshold, min_chunk_len, max_chunk_len

def semantic_sequential_spreading(sentences, similarity_matrix, 
                                 initial_threshold=0.6, 
                                 decay_factor=0.95,
                                 min_threshold=0.35,
                                 min_chunk_len=1, 
                                 max_chunk_len=10,
                                 window_size=3):
    """
    Phân đoạn văn bản theo tuần tự với lan truyền và ngưỡng động cải tiến
    
    Args:
        sentences: Danh sách các câu
        similarity_matrix: Ma trận tương đồng
        initial_threshold: Ngưỡng tương đồng ban đầu
        decay_factor: Hệ số giảm ngưỡng sau mỗi đoạn
        min_threshold: Ngưỡng tương đồng tối thiểu
        min_chunk_len: Độ dài tối thiểu của mỗi đoạn (số câu)
        max_chunk_len: Độ dài tối đa của mỗi đoạn (số câu)
        window_size: Kích thước cửa sổ xem xét xu hướng tương đồng
        
    Returns:
        list: Danh sách các đoạn, mỗi đoạn là một list các chỉ số câu
    """
    n = len(sentences)
    segments = []  # Danh sách các đoạn kết quả
    current_threshold = initial_threshold
    processed = 0  # Số câu đã xử lý
    
    print(f"\n[START] Bắt đầu phân đoạn với ngưỡng ban đầu: {current_threshold:.4f}")
    
    while processed < n:
        segment_num = len(segments) + 1
        print(f"\n[SEGMENT] Đang tìm đoạn {segment_num} từ câu {processed} với ngưỡng: {current_threshold:.4f}")
        
        # Bắt đầu đoạn mới
        current_segment = [processed]  # Bắt đầu với câu đầu tiên chưa xử lý
        last_idx = processed
        
        # Mở rộng đoạn hiện tại cho đến khi gặp điểm ngắt hoặc hết câu
        while last_idx + 1 < n:
            next_idx = last_idx + 1
            sim = similarity_matrix[last_idx][next_idx]
            
            # Tính xu hướng tương đồng cục bộ (trung bình với các câu tiếp theo)
            avg_local_trend = calculate_local_trend(similarity_matrix, last_idx, window_size, n)
            
            # Kiểm tra các điều kiện theo thứ tự ưu tiên
            should_break, reason = check_break_conditions(
                sim, current_threshold, current_segment, 
                next_idx, min_chunk_len, max_chunk_len, avg_local_trend,
                similarity_matrix
            )
            
            if should_break:
                print(reason)
                break
                
            # Nếu không break, thêm câu vào đoạn hiện tại và cập nhật chỉ số
            add_reason = get_add_reason(sim, current_threshold, avg_local_trend, len(current_segment), min_chunk_len)
            current_segment.append(next_idx)
            last_idx = next_idx
            print(add_reason)
        
        # Thêm đoạn hiện tại vào kết quả
        segments.append(current_segment)
        processed = last_idx + 1
        
        print(f"[SUCCESS] Đã tạo đoạn {len(segments)} với {len(current_segment)} câu: {current_segment}")
        
        # Điều chỉnh ngưỡng cho đoạn tiếp theo nếu còn câu chưa xử lý
        if processed < n:
            old_threshold = current_threshold
            current_threshold = adjust_threshold(
                current_threshold, current_segment, 
                decay_factor, min_threshold, min_chunk_len, max_chunk_len
            )
            print(f"[ADAPT] Giảm ngưỡng từ {old_threshold:.4f} xuống {current_threshold:.4f} cho đoạn tiếp theo")
    
    # Hiển thị kết quả phân đoạn
    print(f"\n[RESULT] Đã phân thành {len(segments)} đoạn văn bản:")
    for i, segment in enumerate(segments):
        print(f"[SEGMENT] Đoạn {i+1}: {len(segment)} câu - {segment}")
    
    return segments

def calculate_local_trend(similarity_matrix, current_idx, window_size, n):
    """Tính xu hướng tương đồng cục bộ"""
    local_trend = 0
    valid_comparisons = 0
    
    for look_ahead in range(1, min(window_size + 1, n - current_idx)):
        if current_idx + look_ahead < n:
            local_trend += similarity_matrix[current_idx][current_idx + look_ahead]
            valid_comparisons += 1
    
    return local_trend / valid_comparisons if valid_comparisons > 0 else 0

def check_break_conditions(sim, threshold, current_segment, next_idx, min_chunk_len, max_chunk_len, avg_local_trend, similarity_matrix):
    """Kiểm tra các điều kiện để ngắt đoạn"""
    
    # 1. Kiểm tra độ dài tối đa
    if len(current_segment) >= max_chunk_len:
        return True, f"[BREAK] Điểm tách tại {current_segment[-1]}-{next_idx}: đoạn đạt độ dài tối đa ({max_chunk_len} câu)"
    
    # 2. Phát hiện thay đổi chủ đề mạnh
    if sim < threshold * 0.7 and len(current_segment) >= min_chunk_len:
        # Kiểm tra thêm câu tiếp theo để xác nhận thay đổi chủ đề
        topic_change = is_topic_change(similarity_matrix, current_segment[-1], next_idx)
        if topic_change:
            return True, f"[TOPIC] Phát hiện thay đổi chủ đề tại {current_segment[-1]}-{next_idx}: {sim:.4f} < {threshold*0.7:.4f}"
    
    # 3. Kiểm tra độ tương đồng xấp xỉ ngưỡng và xu hướng không tốt
    if (sim >= threshold * 0.85 and sim < threshold and 
        avg_local_trend < threshold * 0.9 and 
        len(current_segment) >= min_chunk_len):
        return True, f"[BREAK] Điểm tách tại {current_segment[-1]}-{next_idx}: {sim:.4f} < {threshold:.4f} (xu hướng không đủ tốt)"
    
    # 4. Kiểm tra nếu độ tương đồng thấp hơn ngưỡng và đoạn đã đủ dài
    if sim < threshold and len(current_segment) >= min_chunk_len:
        return True, f"[BREAK] Điểm tách tại {current_segment[-1]}-{next_idx}: {sim:.4f} < {threshold:.4f}"
    
    # Nếu không có điều kiện ngắt nào được thỏa mãn
    return False, ""

def is_topic_change(similarity_matrix, current_idx, next_idx):
    """Kiểm tra xem có phải thay đổi chủ đề hay không"""
    if next_idx + 1 >= len(similarity_matrix):
        return True
        
    current_next_sim = similarity_matrix[current_idx][next_idx]
    next_next_sim = similarity_matrix[next_idx][next_idx + 1]
    
    # Nếu câu tiếp theo có tương đồng cao hơn nhiều với câu sau nó
    # so với tương đồng với câu hiện tại -> có thay đổi chủ đề
    return next_next_sim > current_next_sim * 1.5

def get_add_reason(sim, threshold, avg_local_trend, segment_len, min_chunk_len):
    """Tạo lý do cho việc thêm câu vào đoạn"""
    if sim >= threshold:
        return f"[ADD] Thêm câu vào đoạn: {sim:.4f} >= {threshold:.4f}"
    elif sim >= threshold * 0.85 and avg_local_trend >= threshold * 0.9:
        return f"[TREND] Thêm câu dựa vào xu hướng tốt ({avg_local_trend:.4f}): {sim:.4f} gần với {threshold:.4f}"
    elif segment_len < min_chunk_len:
        return f"[FORCE] Buộc thêm câu để đạt độ dài tối thiểu, mặc dù {sim:.4f} < {threshold:.4f}"
    else:
        return f"[ADD] Thêm câu vào đoạn (trường hợp mặc định)"

def adjust_threshold(current_threshold, current_segment, decay_factor, min_threshold, min_chunk_len, max_chunk_len):
    """Điều chỉnh ngưỡng cho đoạn tiếp theo"""
    adaptive_decay = decay_factor
    
    # Điều chỉnh hệ số dựa trên chất lượng đoạn hiện tại
    if len(current_segment) <= min_chunk_len:
        # Đoạn quá ngắn, giảm ngưỡng mạnh hơn
        adaptive_decay *= 0.95
    elif len(current_segment) >= max_chunk_len:
        # Đoạn đạt độ dài tối đa, giảm ngưỡng ít hơn
        adaptive_decay *= 1.05
    
    # Tính ngưỡng mới và đảm bảo không dưới ngưỡng tối thiểu
    new_threshold = current_threshold * adaptive_decay
    return max(new_threshold, min_threshold)

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
    
def display_groups(groups, sentences):
    """
    Hiển thị các nhóm câu
    
    Args:
        groups: Danh sách các nhóm câu
        sentences: Danh sách các câu
    """
    for i, group in enumerate(groups):
        group_sentences = [sentences[idx] for idx in group]
        print(f"\nNhóm {i+1} ({len(group)} câu):")
        for s in group_sentences[:3]:  # Hiển thị tối đa 3 câu đầu tiên
            print(f" - {s[:100]}..." if len(s) > 100 else f" - {s}")
        if len(group_sentences) > 3:
            print(f" - ... và {len(group_sentences)-3} câu khác")

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
    """
    Lưu kết quả phân nhóm và trích xuất OIE vào database
    
    Args:
        query: Query hoặc mô tả về passage
        passage: Nội dung passage
        sentences: Danh sách các câu
        vectors: Danh sách các vector nhúng
        chunks: Danh sách các nhóm câu
        all_triples: Tất cả các triples được trích xuất
        oie_sentence_groups: Cấu trúc phân cấp triples theo nhóm câu
    """
    conn = connect_to_db()
    cursor = conn.cursor()
    
    # Chuyển đổi dữ liệu Python thành JSON để lưu trữ
    sentences_json = json.dumps([str(s) for s in sentences])
    
    # Đối với vectors, cần chuyển numpy arrays thành lists trước
    vectors_list = [v.tolist() for v in vectors]
    vectors_json = json.dumps(vectors_list)
    
    # Chuyển chunks thành JSON
    chunks_json = json.dumps([[int(idx) for idx in group] for group in chunks])
    
    # Chuyển OIE triples thành JSON
    triples_json = json.dumps(all_triples)
    oie_groups_json = json.dumps(oie_sentence_groups)
    
    try:
        cursor.execute(
            """
            INSERT INTO Semantic_Splitter 
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
                triples_json,
                oie_groups_json
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
    sentence_triples = [[] for _ in range(len(sentences))]  # Danh sách trống cho mỗi câu
    
    # Hiển thị tiến độ
    start_time = time.time()
    
    # Xử lý theo nhóm để có context tốt hơn
    for group_idx, group in enumerate(groups):
        print(f"  Đang trích xuất quan hệ cho nhóm {group_idx+1}/{len(groups)}...")
        
        # Trích xuất triples cho từng câu trong nhóm
        for i, sentence_idx in enumerate(group):
            sentence = sentences[sentence_idx]
            
            # Trích xuất triples từ câu
            triples = extract_triples_for_search(sentence)
            
            # Lưu trữ triples
            sentence_triples[sentence_idx] = triples
            all_triples.extend(triples)
    
    elapsed_time = time.time() - start_time
    print(f"\n[SUCCESS] Đã trích xuất {len(all_triples)} quan hệ trong {elapsed_time:.2f}s")
    
    return all_triples, sentence_triples
    
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
    
    # Bước 1: Tách câu với phương pháp tối ưu
    sentences = to_sentences(document, use_enhanced=True, use_spacy=True, min_words=3)
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
    initial_threshold = float(input(f"\nNhập threshold relationship ban đầu (0.0-1.0, đề xuất: {percentiles['75%']:.4f}): ") or percentiles['75%'])
    decay_factor = float(input("Nhập hệ số giảm threshold (0.7-0.95, đề xuất: 0.9): ") or "0.9")
    min_threshold = float(input(f"Nhập threshold tối thiểu (0.1-0.5, đề xuất: {percentiles['25%']:.4f}): ") or percentiles['25%'])
    min_chunk_len = int(input("Nhập độ dài tối thiểu của đoạn (1-5, đề xuất: 2): ") or "2")
    max_chunk_len = int(input("Nhập độ dài tối đa của đoạn (5-15, đề xuất: 8): ") or "8")
    
    # Bước 6: Phân nhóm câu với thuật toán phân đoạn tuần tự
    groups = semantic_sequential_spreading(sentences, sim_matrix, 
                                 initial_threshold, 
                                 decay_factor, 
                                 min_threshold, 
                                 min_chunk_len, 
                                 max_chunk_len)
    print(f"Đã phân thành {len(groups)} nhóm ")
    
    # Bước 7: Hiển thị các nhóm câu
    display_groups(groups, sentences)
    
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

def export_results_to_file(document, sentences, groups, sentence_triples, document_id):
    """
    Xuất kết quả phân nhóm và OIE ra file văn bản
    
    Args:
        document: Văn bản gốc
        sentences: Danh sách các câu
        groups: Các nhóm câu
        sentence_triples: Danh sách triples theo từng câu
        document_id: ID của tài liệu
    """
    # Chuẩn hóa document_id cho phù hợp với tên file
    if document_id is None:
        document_id = f"doc_{int(time.time())}"
    
    # Làm sạch document_id, loại bỏ các ký tự không hợp lệ
    safe_id = re.sub(r'[^a-zA-Z0-9_-]', '_', document_id.replace(' ', '_'))
    
    # Thêm thông tin thời gian và phương pháp
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    # Tạo tên file với nhiều thông tin hơn
    filename = generate_output_filename(document_id, prefix="splitter")
    
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(f"KẾT QUẢ PHÂN NHÓM TUẦN TỰ VÀ TRÍCH XUẤT QUAN HỆ (OIE)\n")
            f.write(f"Tài liệu: {document_id}\n")
            f.write(f"Thời gian: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write(f"Tổng số câu: {len(sentences)}\n")
            f.write(f"Số nhóm: {len(groups)}\n\n")
            
            # Đếm tổng số triples
            total_triples = sum(len(triples) for triples in sentence_triples)
            f.write(f"Tổng số quan hệ trích xuất: {total_triples}\n\n")
            
            for i, group in enumerate(groups):
                f.write(f"=== NHÓM {i+1} ({len(group)} câu) ===\n")
                
                # Tính số triples trong nhóm
                group_triples_count = sum(len(sentence_triples[idx]) for idx in group)
                f.write(f"Số quan hệ trong nhóm: {group_triples_count}\n\n")
                
                for idx in group:
                    f.write(f"[{idx}] {sentences[idx]}\n")
                    
                    # Ghi các triples của câu
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

def extract_triples(sentences):
    """
    Trích xuất quan hệ (subject, relation, object) từ danh sách câu
    
    Args:
        sentences: Danh sách các câu cần trích xuất
        
    Returns:
        list: Danh sách các triple từ tất cả câu
        list: Danh sách các triple theo từng câu
    """
    all_triples = []
    sentence_triples = []
    
    print(f"[INFO] Đang trích xuất OpenIE triples từ {len(sentences)} câu...")
    start_time = time.time()
    
    for i, sentence in enumerate(sentences):
        # Hiển thị tiến độ
        if i % 10 == 0 or i == len(sentences) - 1:
            elapsed = time.time() - start_time
            eta = (elapsed / (i + 1)) * (len(sentences) - i - 1) if i > 0 else 0
            print(f"  Đang xử lý câu {i+1}/{len(sentences)} - {(i+1)/len(sentences)*100:.1f}% - ETA: {eta:.1f}s", end="\r")
            
        # Trích xuất triples từ câu
        triples = extract_triples_for_search(sentence)
        
        # Lưu trữ triples
        sentence_triples.append(triples)
        all_triples.extend(triples)
    
    print(f"\n[SUCCESS] Đã trích xuất {len(all_triples)} quan hệ từ {len(sentences)} câu trong {time.time() - start_time:.1f}s")
    
    return all_triples, sentence_triples


if __name__ == "__main__":
    print("1. Xử lý passages từ tập dữ liệu")
    print("2. Phân tích tài liệu tự nhập")
    print("3. Phân tích tài liệu từ file")
    
    choice = input("\nLựa chọn của bạn (1/2/3): ")
    
    if choice == '1':
        # Đọc dataset
        doc = pd.read_csv('passages_1000.csv')
        passages = doc['passage_text'].tolist()
        
        # Lấy số lượng passages cần xử lý từ người dùng
        while True:
            try:
                num_passages = int(input("Nhập số passages cần xử lý (1-1000): "))
                if 1 <= num_passages <= 1000:
                    break
                else:
                    print("Số lượng phải nằm trong khoảng từ 1 đến 1000.")
            except ValueError:
                print("Vui lòng nhập một số nguyên.")
        
        # Giới hạn số lượng passages
        passages_to_process = passages[:min(num_passages, len(passages))]
        
        for i, passage in enumerate(passages_to_process):
            process_document(passage, f"Passage {i+1}", query=f"Query for passage {i+1}")
    
    elif choice == '2':
        # Nhập tài liệu trực tiếp
        print("\n=== NHẬP TÀI LIỆU TRỰC TIẾP ===")
        print("Nhập nội dung tài liệu (kết thúc bằng một dòng chỉ có '###'):")
        lines = []
        while True:
            line = input()
            if line == '###':
                break
            lines.append(line)
        
        if not lines:
            print("Không có nội dung được nhập, hủy phân tích.")
            exit()
        
        document = "\n".join(lines)
        query = input("Nhập query hoặc chủ đề tài liệu (có thể để trống): ")
        
        # Xử lý tài liệu nhập trực tiếp
        process_document(document, "Splitter_Tài_liệu_thủ_công", query=query, save_to_db=False)
    
    elif choice == '3':
        # Tải tài liệu từ file
        file_path = input("\nNhập đường dẫn đến file tài liệu: ")
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                document = file.read()
            print(f"Đã tải thành công tài liệu từ {file_path}")
            
            # Lấy tên file làm document_id
            import os
            document_id = os.path.splitext(os.path.basename(file_path))[0]
            
            query = input("Nhập query hoặc chủ đề tài liệu (có thể để trống): ")
            
            # Xử lý tài liệu từ file
            process_document(document, f"Splitter_{document_id}", query=query, save_to_db=False)
            
        except Exception as e:
            print(f"Lỗi khi đọc file: {e}")
    
    else:
        print("Lựa chọn không hợp lệ.")