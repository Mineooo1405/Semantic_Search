import pandas as pd
from Tool.Database import connect_to_db
import json
import numpy as np
from dotenv import load_dotenv
import os
from Tool.Sentence_Detector import sentence_detector
from Tool.Sentence_Embedding import sentence_embedding
import matplotlib.pyplot as plt
import seaborn as sns
import time
import pickle

load_dotenv()

def to_sentences(passage):
    return sentence_detector(passage)

def to_vectors(sentences, use_cache=True, cache_prefix="passage_vectors_"):
    """Chuyển đổi câu thành embedding vector với caching theo từng passage"""
    cache_file = f"{cache_prefix}{hash(str(sentences))}.pkl"
    
    # Kiểm tra xem có thể sử dụng cache không
    if use_cache and os.path.exists(cache_file):
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
    
    # Tính toán vector nhúng cho từng câu với hiển thị tiến trình
    for i, sentence in enumerate(sentences):
        if i % 5 == 0 or i == total - 1:
            elapsed = time.time() - start_time
            eta = (elapsed / (i + 1)) * (total - i - 1) if i > 0 else 0
            print(f"Đang tạo vector [{i+1}/{total}] - {(i+1)/total*100:.1f}% - ETA: {eta:.1f}s", end="\r")
        
        vectors.append(sentence_embedding(sentence))
    
    print("\nĐã hoàn thành việc tạo embedding vector!")
    
    # Lưu vào cache cho lần sau
    if use_cache:
        print("Đang lưu embedding vector vào cache...")
        os.makedirs(os.path.dirname(cache_file) if os.path.dirname(cache_file) else '.', exist_ok=True)
        with open(cache_file, 'wb') as f:
            pickle.dump(vectors, f)
        print("Đã lưu embedding vector vào cache!")
    
    return vectors

def create_semantic_matrix(vectors):
    """Tạo ma trận ngữ nghĩa dựa trên cosine similarity"""
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

def semantic_spreading_grouping(similarity_matrix, threshold=0.7):
    """
    Thực hiện phân nhóm bằng giải thuật semantic spreading grouping
    đã được cải tiến để hoạt động chính xác
    """
    n = len(similarity_matrix)
    grouped = [False] * n  # Đánh dấu câu đã được phân nhóm
    groups = []  # Danh sách các nhóm
    
    # In thông tin tổng quan về ma trận
    print("\nThông tin tổng quan về ma trận:")
    above_threshold_pairs = []
    for i in range(n):
        for j in range(i+1, n):  # Chỉ kiểm tra một nửa ma trận (loại bỏ các cặp trùng lặp)
            sim = similarity_matrix[i][j]
            if sim >= threshold:
                above_threshold_pairs.append((i, j, sim))
    
    # Sắp xếp các cặp theo độ tương đồng giảm dần
    above_threshold_pairs.sort(key=lambda x: x[2], reverse=True)
    
    # In ra các cặp câu có độ tương đồng vượt ngưỡng
    if above_threshold_pairs:
        print(f"Các cặp câu có độ tương đồng >= {threshold}:")
        for i, j, sim in above_threshold_pairs:
            print(f"  Câu {i} và câu {j}: {sim:.4f}")
    else:
        print(f"Không tìm thấy cặp câu nào có độ tương đồng >= {threshold}")
        # Nếu không có cặp nào vượt ngưỡng, giảm ngưỡng xuống để tìm được ít nhất vài cặp
        dynamic_threshold = threshold
        while not above_threshold_pairs and dynamic_threshold > 0.4:
            dynamic_threshold -= 0.1
            for i in range(n):
                for j in range(i+1, n):
                    sim = similarity_matrix[i][j]
                    if sim >= dynamic_threshold:
                        above_threshold_pairs.append((i, j, sim))
            
            if above_threshold_pairs:
                print(f"Đề xuất giảm threshold xuống {dynamic_threshold:.2f}. Các cặp câu tìm được:")
                for i, j, sim in above_threshold_pairs[:5]:  # Chỉ hiển thị 5 cặp đầu tiên
                    print(f"  Câu {i} và câu {j}: {sim:.4f}")
                break
    
    # Lặp cho đến khi tất cả các câu được phân nhóm
    while False in grouped:
        # Tìm câu chưa được phân nhóm đầu tiên làm anchor
        anchor = grouped.index(False)
        
        # Tạo nhóm mới với anchor
        current_group = [anchor]
        grouped[anchor] = True
        
        # Danh sách câu cần xét trong lần lan truyền này
        to_check = [anchor]
        
        # Lặp cho đến khi không mở rộng được nhóm nữa
        while to_check:
            current = to_check.pop(0)  # Lấy câu đầu tiên từ danh sách cần xét
            
            # So sánh câu hiện tại với các câu chưa được phân nhóm
            for j in range(n):
                if not grouped[j] and similarity_matrix[current][j] >= threshold:
                    current_group.append(j)  # Thêm vào nhóm hiện tại
                    grouped[j] = True  # Đánh dấu đã phân nhóm
                    to_check.append(j)  # Thêm vào danh sách cần xét
        
        # Sắp xếp nhóm theo thứ tự chỉ số tăng dần để dễ theo dõi
        current_group.sort()
        # Thêm nhóm hiện tại vào danh sách các nhóm
        groups.append(current_group)
    
    return groups

def visualize_similarity_matrix(matrix, groups=None, title="Semantic Similarity Matrix"):
    """
    Hiển thị ma trận tương đồng với phân nhóm
    
    Args:
        matrix: Ma trận tương đồng
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

def process_passage(passage, passage_id, query="", threshold=0.7, visualize=True):
    """Xử lý một đoạn văn: phát hiện câu, nhúng, phân nhóm và lưu kết quả"""
    print(f"\n--- Đang xử lý passage {passage_id} ---")
    
    # Tách câu
    sentences = to_sentences(passage)
    print(f"Đã tách thành {len(sentences)} câu")
    
    # Nếu không có câu nào, trả về kết quả rỗng
    if not sentences:
        print("Không có câu nào trong passage, bỏ qua")
        return [], [], []
    
    # Tạo vector nhúng
    vectors = to_vectors(sentences)
    
    # Tạo ma trận ngữ nghĩa
    sim_matrix = create_semantic_matrix(vectors)
    
    # Phân nhóm câu
    groups = semantic_spreading_grouping(sim_matrix, threshold)
    print(f"Đã phân thành {len(groups)} nhóm ngữ nghĩa")
    
    # Hiển thị kết quả
    for i, group in enumerate(groups):
        group_sentences = [sentences[idx] for idx in group]
        print(f"\nNhóm {i+1} ({len(group)} câu):")
        for s in group_sentences[:2]:  # Chỉ hiển thị 2 câu đầu tiên để ngắn gọn
            print(f" - {s[:100]}...")
        if len(group_sentences) > 2:
            print(f" - ... và {len(group_sentences)-2} câu khác")
    
    # Trực quan hóa ma trận
    if visualize:
        visualize_similarity_matrix(sim_matrix, groups, 
                                  title=f"Ma trận tương đồng ngữ nghĩa - Passage {passage_id}")
    
    # Lưu kết quả vào database
    save_to_database(query, passage, sentences, vectors, groups)
    
    return sentences, vectors, groups

if __name__ == "__main__":
    # Đọc dataset
    doc = pd.read_csv('passages_1000.csv')
    passages = doc['passage_text'].tolist()
    
    # Lấy số lượng passages cần xử lý từ người dùng
    num_passages = int(input("Nhập số passages cần xử lý (1-1000): "))
    threshold = float(input("Nhập ngưỡng tương đồng (0.0-1.0): "))
    
    # Giới hạn số lượng passages
    passages_to_process = passages[:min(num_passages, len(passages))]
    
    for i, passage in enumerate(passages_to_process):
        process_passage(passage, i+1, query=f"Query for passage {i+1}", threshold=threshold)

