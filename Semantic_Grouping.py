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
    print(f"\nĐã phân thành {len(groups)} nhóm ngữ nghĩa:")
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

def process_passage(passage, passage_id, query="", visualize=True, save_to_db=True):
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
    
    # Phân tích phân bố relationship và đưa ra đề xuất threshold
    percentiles = analyze_similarity_distribution(sim_matrix, sentences)
    
    # Yêu cầu người dùng chọn các tham số
    while True:
        try:
            initial_threshold = float(input("\nNhập threshold relationship ban đầu (0.0-1.0): "))
            if 0 <= initial_threshold <= 1:
                break
            else:
                print("threshold phải nằm trong khoảng từ 0 đến 1.")
        except ValueError:
            print("Vui lòng nhập một số thực.")
    
    while True:
        try:
            decay_factor = float(input("Nhập hệ số giảm threshold (0.7-0.95): "))
            if 0.7 <= decay_factor <= 0.95:
                break
            else:
                print("Hệ số giảm nên nằm trong khoảng từ 0.7 đến 0.95.")
        except ValueError:
            print("Vui lòng nhập một số thực.")
    
    while True:
        try:
            min_threshold = float(input("Nhập threshold tối thiểu (0.1-0.5): "))
            if 0.1 <= min_threshold <= 0.5:
                break
            else:
                print("threshold tối thiểu nên nằm trong khoảng từ 0.1 đến 0.5.")
        except ValueError:
            print("Vui lòng nhập một số thực.")
    
    # Phân nhóm câu với threshold giảm dần
    groups = semantic_spreading_grouping(sim_matrix, initial_threshold, decay_factor, min_threshold)
    print(f"Đã phân thành {len(groups)} nhóm ngữ nghĩa")
    
    # Hiển thị kết quả
    for i, group in enumerate(groups):
        group_sentences = [sentences[idx] for idx in group]
        print(f"\nNhóm {i+1} ({len(group)} câu):")
        for s in group_sentences[:3]:  # Hiển thị tối đa 3 câu đầu tiên
            print(f" - {s[:100]}..." if len(s) > 100 else f" - {s}")
        if len(group_sentences) > 3:
            print(f" - ... và {len(group_sentences)-3} câu khác")
    
    # Trực quan hóa ma trận
    if visualize:
        visualize_similarity_matrix(sim_matrix, groups, 
                                  title=f"Ma trận relationship ngữ nghĩa - Passage {passage_id}")
    
    # Lưu kết quả vào database nếu được yêu cầu
    if save_to_db:
        save_to_database(query, passage, sentences, vectors, groups)
    
    return sentences, vectors, groups

def test_manual_passage():
    """Cho phép người dùng nhập passage và kiểm thử thuật toán mà không lưu vào database"""
    print("\n=== TEST THỦ CÔNG VỚI PASSAGE TỰ NHẬP ===")
    print("Nhập đoạn văn (kết thúc bằng một dòng trống):")
    
    lines = []
    while True:
        line = input()
        if not line:
            break
        lines.append(line)
    
    if not lines:
        print("Không có nội dung được nhập, hủy test.")
        return
    
    passage = "\n".join(lines)
    query = input("Nhập query (có thể để trống): ")
    
    # Xử lý passage nhưng không lưu vào database
    process_passage(passage, "thủ công", query=query, save_to_db=False)

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

def process_document(document, document_id, query="", visualize=True, save_to_db=False):
    """Xử lý một tài liệu: phát hiện câu, nhúng, phân nhóm và lưu kết quả"""
    print(f"\n--- Đang xử lý tài liệu {document_id} ---")
    
    # Tách câu
    sentences = to_sentences(document)
    print(f"Đã tách thành {len(sentences)} câu")
    
    # Nếu không có câu nào, trả về kết quả rỗng
    if not sentences:
        print("Không có câu nào trong tài liệu, bỏ qua")
        return [], [], []
    
    # Tạo vector nhúng
    vectors = to_vectors(sentences)
    
    # Tạo ma trận ngữ nghĩa
    sim_matrix = create_semantic_matrix(vectors)
    
    # Phân tích phân bố relationship và đưa ra đề xuất threshold
    percentiles = analyze_similarity_distribution(sim_matrix, sentences)
    
    # Yêu cầu người dùng chọn các tham số
    while True:
        try:
            initial_threshold = float(input("\nNhập threshold relationship ban đầu (0.0-1.0): "))
            if 0 <= initial_threshold <= 1:
                break
            else:
                print("threshold phải nằm trong khoảng từ 0 đến 1.")
        except ValueError:
            print("Vui lòng nhập một số thực.")
    
    while True:
        try:
            decay_factor = float(input("Nhập hệ số giảm threshold (0.7-0.95): "))
            if 0.7 <= decay_factor <= 0.95:
                break
            else:
                print("Hệ số giảm nên nằm trong khoảng từ 0.7 đến 0.95.")
        except ValueError:
            print("Vui lòng nhập một số thực.")
    
    while True:
        try:
            min_threshold = float(input("Nhập threshold tối thiểu (0.1-0.5): "))
            if 0.1 <= min_threshold <= 0.5:
                break
            else:
                print("threshold tối thiểu nên nằm trong khoảng từ 0.1 đến 0.5.")
        except ValueError:
            print("Vui lòng nhập một số thực.")
    
    # Phân nhóm câu với threshold giảm dần
    groups = semantic_spreading_grouping(sim_matrix, initial_threshold, decay_factor, min_threshold)
    print(f"Đã phân thành {len(groups)} nhóm ngữ nghĩa")
    
    # Hiển thị kết quả
    for i, group in enumerate(groups):
        group_sentences = [sentences[idx] for idx in group]
        print(f"\nNhóm {i+1} ({len(group)} câu):")
        
        # Hiển thị các câu trong nhóm (giới hạn số câu hiển thị nếu quá nhiều)
        max_display = min(5, len(group_sentences))
        for s in group_sentences[:max_display]:
            print(f" - {s[:80]}..." if len(s) > 80 else f" - {s}")
        if len(group_sentences) > max_display:
            print(f" - ... và {len(group_sentences)-max_display} câu khác")
    
    # Trực quan hóa ma trận
    if visualize:
        visualize_similarity_matrix(sim_matrix, groups, 
                                  title=f"Ma trận relationship ngữ nghĩa - {document_id}")
    
    # Xuất kết quả ra file để người dùng có thể tham khảo chi tiết
    export_results_to_file(document, sentences, groups, document_id)
    
    # Lưu kết quả vào database nếu được yêu cầu
    if save_to_db:
        save_to_database(query, document, sentences, vectors, groups)
    
    return sentences, vectors, groups

def export_results_to_file(document, sentences, groups, document_id):
    """Xuất kết quả phân nhóm ra file văn bản"""
    filename = f"semantic_groups_{document_id.replace(' ','_')}.txt"
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(f"KẾT QUẢ PHÂN NHÓM NGỮ NGHĨA\n")
            f.write(f"Tài liệu: {document_id}\n")
            f.write(f"Thời gian: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write(f"Tổng số câu: {len(sentences)}\n")
            f.write(f"Số nhóm ngữ nghĩa: {len(groups)}\n\n")
            
            for i, group in enumerate(groups):
                f.write(f"NHÓM {i+1} ({len(group)} câu):\n")
                for idx in group:
                    f.write(f"[{idx}] {sentences[idx]}\n")
                f.write("\n")
        
        print(f"\nĐã xuất kết quả chi tiết ra file: {filename}")
    except Exception as e:
        print(f"Lỗi khi xuất kết quả ra file: {e}")

# Thay đổi main để sử dụng hàm mới
if __name__ == "__main__":
    print("1. Xử lý passages từ tập dữ liệu")
    print("2. Phân tích tài liệu tự nhập hoặc từ file")
    
    choice = input("\nLựa chọn của bạn (1/2): ")
    
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
            process_passage(passage, i+1, query=f"Query for passage {i+1}")
    
    elif choice == '2':
        test_document()
    
    else:
        print("Lựa chọn không hợp lệ.")


