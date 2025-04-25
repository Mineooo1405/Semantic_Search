import pandas as pd
from Tool.Database import connect_to_db
import json
import numpy as np
from dotenv import load_dotenv
import os
from Tool.Sentence_Detector import extract_and_simplify_sentences
from Tool.Sentence_Embedding import sentence_embedding
from Tool.OIE import extract_triples_for_search  # Thêm import OIE
import matplotlib.pyplot as plt
import seaborn as sns
import time
import pickle
import spacy
import re

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
        
        vectors.append(sentence_embedding(sentence))
    
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

def semantic_spreading_grouping(similarity_matrix, initial_threshold=0.75, decay_factor=0.9, min_threshold=0.2):
    """
    Thực hiện phân nhóm bằng giải thuật semantic spreading (greeding) với threshold giảm dần
    
    Args:
        similarity_matrix: Ma trận relationship
        initial_threshold: threshold  ban đầu
        decay_factor: Hệ số giảm threshold sau mỗi nhóm (0.9 = giảm 10%)
        min_threshold: threshold relationship tối thiểu
    
    Returns:
        list: Danh sách các nhóm, mỗi nhóm là một list các câu
    """
    n = len(similarity_matrix)
    grouped = [False] * n  # Đánh dấu câu đã được phân nhóm (mảng đánh dấu)
    groups = []  # Danh sách các nhóm
    current_threshold = initial_threshold
    
    print(f"\nBắt đầu phân nhóm với threshold ban đầu: {current_threshold:.4f}")
    
    # Lặp cho đến khi tất cả các câu được phân nhóm
    group_count = 0
    
    while False in grouped:
        group_count += 1
        print(f"\nĐang tìm nhóm {group_count} với threshold: {current_threshold:.4f}")
        
        # 1. Tìm và sắp xếp tất cả các cặp câu theo relationship giảm dần
        all_pairs = []
        for i in range(n):
            if not grouped[i]:  # Chỉ xét các câu chưa được phân nhóm
                for j in range(i+1, n):
                    if not grouped[j]:  # Chỉ xét các câu chưa được phân nhóm
                        sim = similarity_matrix[i][j]
                        if sim >= current_threshold:
                            all_pairs.append((i, j, sim))
        
        # Sắp xếp các cặp theo relationship giảm dần
        all_pairs.sort(key=lambda x: x[2], reverse=True)
        
        # Hiển thị thông tin về các cặp câu vượt threshold
        if all_pairs:
            print(f"  Tìm thấy {len(all_pairs)} cặp câu có relationship >= {current_threshold:.4f}")
            if len(all_pairs) > 0:
                i, j, sim = all_pairs[0]  # Cặp có relationship cao nhất
                print(f"  Cặp tốt nhất: Câu {i} và câu {j}: {sim:.4f}")
        else:
            print(f"  Không tìm thấy cặp câu nào có relationship >= {current_threshold:.4f}")
            
            # Nếu không tìm thấy cặp nào vượt threshold hiện tại, giảm threshold ngay lập tức
            new_threshold = current_threshold * decay_factor
            if new_threshold < min_threshold:
                # Nếu giảm threshold quá thấp, thêm mỗi câu chưa nhóm vào một nhóm riêng
                for i in range(n):
                    if not grouped[i]:
                        groups.append([i])
                        grouped[i] = True
                print(f"  threshold giảm xuống {new_threshold:.4f} < {min_threshold:.4f} (tối thiểu), mỗi câu còn lại là một nhóm")
                break
            else:
                current_threshold = new_threshold
                print(f"  Giảm threshold xuống {current_threshold:.4f} và thử lại")
                continue
        
        # 2. Chọn cặp tốt nhất làm anchor cho nhóm mới
        if all_pairs:
            i, j, _ = all_pairs[0]  # Lấy cặp có relationship cao nhất
            current_group = [i, j]
            grouped[i] = grouped[j] = True
            
            # 3. Mở rộng nhóm bằng cách lan truyền
            to_check = current_group.copy()
            
            while to_check:
                current = to_check.pop(0)
                
                # So sánh câu hiện tại với các câu chưa được phân nhóm
                for k in range(n):
                    if not grouped[k] and similarity_matrix[current][k] >= current_threshold:
                        current_group.append(k)
                        grouped[k] = True
                        to_check.append(k)
            
            # 4. Thêm nhóm vào danh sách các nhóm
            current_group.sort()
            groups.append(current_group)
            print(f"  Đã tạo nhóm {len(groups)} với {len(current_group)} câu: {current_group}")
            
            # 5. Giảm threshold cho nhóm tiếp theo
            current_threshold *= decay_factor
            if current_threshold < min_threshold:
                current_threshold = min_threshold
            print(f"  Giảm threshold xuống {current_threshold:.4f} cho nhóm tiếp theo")
            
        else:
            # Trường hợp không tìm thấy cặp nào, nhưng vẫn còn câu chưa được phân nhóm
            # (Chỉ xảy ra khi threshold đã giảm tới min_threshold)
            # Thêm câu đầu tiên chưa được phân nhóm vào một nhóm mới
            anchor = grouped.index(False)
            groups.append([anchor])
            grouped[anchor] = True
            print(f"  Tạo nhóm đơn lẻ cho câu {anchor}")
    
    # Hiển thị thống kê về các nhóm
    print(f"\nĐã phân thành {len(groups)} nhóm :")
    for i, group in enumerate(groups):
        print(f"  Nhóm {i+1}: {len(group)} câu - {group}")
    
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
        process_document(document, "Tài_liệu_thủ_công", query=query, save_to_db=False)
    
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
            process_document(document, document_id, query=query, save_to_db=False)
            
        except Exception as e:
            print(f"Lỗi khi đọc file: {e}")
    
    else:
        print("Lựa chọn không hợp lệ.")