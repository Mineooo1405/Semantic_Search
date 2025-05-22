import pandas as pd
from Tool.Database import connect_to_db # Keep for potential future use, but not for chunking function
from dotenv import load_dotenv
from Tool.Sentence_Detector import extract_and_simplify_sentences
# Use the standard embedding function
from Tool.Sentence_Embedding import sentence_embedding as embed_text_list # Assuming this handles model loading
from Tool.OIE import extract_relations_from_paragraph # MODIFIED: Changed import
import os # Keep for potential future use
import json # Keep for potential future use
import time # Keep for potential future use
import re # Keep for potential future use
import nltk # For sentence tokenization
import pickle # Keep for potential future use
from psycopg2 import extras as psycopg2_extras # Keep for potential future use
from pgvector.psycopg2 import register_vector # Keep for potential future use
from typing import List, Tuple, Dict, Union, Optional # Added Optional
import gc # <-- IMPORT GARBAGE COLLECTOR
from sklearn.metrics.pairwise import cosine_similarity # <-- IMPORT cosine_similarity
import numpy as np # Ensure numpy is imported
import matplotlib.pyplot as plt # For visualization
import seaborn as sns # For visualization
import hashlib # Added for generate_output_filename fallback
# import traceback # Optional: for detailed error logging if needed

load_dotenv()

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

# --- Helper function for batched embedding ---
def embed_sentences_in_batches(sentences: List[str], model_name: str, batch_size: int = 32, device: Optional[object] = None) -> Optional[np.ndarray]: # Added device
    """Nhúng danh sách câu theo lô để giảm sử dụng bộ nhớ."""
    if not sentences:
        return None
    all_embeddings = []
    print(f"  Embedding {len(sentences)} sentences in batches of {batch_size} on device {device}...") # Add print
    try:
        num_batches = (len(sentences) + batch_size - 1) // batch_size
        for i in range(0, len(sentences), batch_size):
            batch_num = (i // batch_size) + 1
            print(f"    Embedding batch {batch_num}/{num_batches}...", end='\r')
            batch = sentences[i:i+batch_size]
            # Gọi hàm nhúng gốc, truyền model_name và device nếu cần
            batch_embeddings = embed_text_list(batch, model_name_or_path=model_name, device=device) # Pass model_name and device
            if batch_embeddings is not None:
                 # Chuyển đổi sang numpy array nếu cần và giải phóng tensor gốc (nếu có)
                 if hasattr(batch_embeddings, 'cpu'): # Check if it's a PyTorch tensor
                     batch_embeddings = batch_embeddings.cpu().numpy()
                 elif hasattr(batch_embeddings, 'numpy'): # Check if it's a TensorFlow tensor
                     batch_embeddings = batch_embeddings.numpy()

                 all_embeddings.append(np.array(batch_embeddings))
            # Giải phóng bộ nhớ batch trung gian (quan trọng)
            del batch
            del batch_embeddings
            gc.collect() # Gọi GC sau mỗi batch

        print(f"    Finished embedding {len(sentences)} sentences.         ") # Clear progress line

        if not all_embeddings:
            return None

        embeddings_array = np.vstack(all_embeddings)
        del all_embeddings # Giải phóng danh sách các batch embeddings
        gc.collect()
        return embeddings_array

    except Exception as e:
        print(f"\nError during batched embedding: {e}")
        # Giải phóng bộ nhớ nếu có lỗi
        if 'all_embeddings' in locals(): del all_embeddings
        gc.collect()
        return None

# --- Modified create_semantic_matrix ---
def create_semantic_matrix(sentences: List[str], model_name: str, batch_size_embed: int = 32, device: Optional[object] = None) -> Optional[np.ndarray]: # Added device
    """
    Tạo ma trận tương đồng ngữ nghĩa giữa các câu, sử dụng embedding theo lô.
    """
    if len(sentences) < 2:
        return None

    # Nhúng câu theo lô
    embeddings = embed_sentences_in_batches(sentences, model_name=model_name, batch_size=batch_size_embed, device=device) # Pass device

    if embeddings is None or embeddings.shape[0] != len(sentences):
        print("  Error: Embedding failed or mismatch in number of embeddings.")
        if 'embeddings' in locals() and embeddings is not None: del embeddings # Giải phóng nếu có lỗi
        gc.collect()
        return None

    # Tính toán ma trận tương đồng cosine
    print(f"  Calculating similarity matrix for {len(sentences)} sentences...")
    try:
        # Sử dụng hàm tối ưu của sklearn
        sim_matrix = cosine_similarity(embeddings)
        print("  Similarity matrix calculation complete.")
        # Giải phóng bộ nhớ embeddings sau khi tính xong ma trận
        del embeddings
        gc.collect()
        return sim_matrix
    except Exception as e:
        print(f"  Error calculating similarity matrix: {e}")
        if 'embeddings' in locals() and embeddings is not None: del embeddings # Giải phóng nếu có lỗi
        gc.collect()
        return None

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
                **{f'p{p}': original_max for p in [10, 25, 50, 75, 80, 85, 90, 95]}}
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

def display_oie_triples(groups: List[List[int]], 
                        sentences: List[str], 
                        oie_results_by_group: List[List[Dict[str, str]]], 
                        max_display_triples_per_group: int = 5):
    """
    Hiển thị các triples OIE được trích xuất cho mỗi nhóm câu.
    
    Args:
        groups: Danh sách các nhóm (mỗi nhóm là danh sách chỉ số câu).
        sentences: Danh sách tất cả các câu.
        oie_results_by_group: Danh sách các kết quả OIE (mỗi phần tử là list các triple dict cho một nhóm).
        max_display_triples_per_group: Số triples tối đa hiển thị cho mỗi nhóm.
    """
    print("\\n=== TRÍCH XUẤT QUAN HỆ THEO NHÓM (OIE Results per Group) ===")
    
    if not oie_results_by_group:
        print("  Không có dữ liệu OIE để hiển thị.")
        return

    for i, group_indices in enumerate(groups):
        group_text_preview_list = [sentences[idx] for idx in group_indices if idx < len(sentences)]
        group_text_preview = " ".join(group_text_preview_list)[:100] + "..."
        print(f"\\nNhóm {i+1} (Văn bản nhóm: \"{group_text_preview}\"):") # Corrected syntax error
        
        current_group_oie_triples = oie_results_by_group[i] if i < len(oie_results_by_group) else []
        
        if current_group_oie_triples:
            print(f"  Tìm thấy {len(current_group_oie_triples)} triple(s) cho nhóm này:")
            for triple_idx, triple_dict in enumerate(current_group_oie_triples[:max_display_triples_per_group]):
                subj = triple_dict.get('subject', 'N/A')
                rel = triple_dict.get('relation', 'N/A')
                obj = triple_dict.get('object', 'N/A')
                print(f"    - ({subj}, {rel}, {obj})")

            if len(current_group_oie_triples) > max_display_triples_per_group:
                print(f"    - ... và {len(current_group_oie_triples) - max_display_triples_per_group} triple(s) khác.")
        else:
            print("  Không có triples nào được trích xuất cho nhóm này hoặc có lỗi xảy ra.")

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
        tuple: (sentences, vectors, groups, all_triples_raw)
    """
    print(f"\\n--- Đang xử lý {document_id} ---")
    
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
    all_triples_raw = [] 
    oie_results_by_group = [] # List of lists of OIE dicts, one inner list per group

    if extract_oie:
        print("\\nExtracting OIE for groups using extract_relations_from_paragraph...")
        for i, group_indices in enumerate(groups):
            group_sentences_list = [sentences[idx] for idx in group_indices if idx < len(sentences)]
            group_text = " ".join(group_sentences_list)
            
            current_group_oie_triples_for_this_group = [] # Renamed to avoid confusion
            if group_text.strip(): # Ensure text is not empty before calling OIE
                try:
                    # Ensure extract_relations_from_paragraph is available in this scope
                    # It should be imported at the top of the file: from Tool.OIE import extract_relations_from_paragraph
                    current_group_oie_triples_for_this_group = extract_relations_from_paragraph(group_text)
                    all_triples_raw.extend(current_group_oie_triples_for_this_group)
                except Exception as e:
                    print(f"Lỗi trong quá trình trích xuất OIE cho nhóm {i+1}: {e}")
                    # import traceback # Consider importing traceback at the top if this is used
                    # traceback.print_exc()
            oie_results_by_group.append(current_group_oie_triples_for_this_group) 
        print(f"\\n[SUCCESS] Đã trích xuất {len(all_triples_raw)} quan hệ từ tất cả các nhóm.")
    
    # Bước 9: Hiển thị OIE triples theo nhóm
    if extract_oie and all_triples_raw: # Check all_triples_raw
        # Call the updated display_oie_triples
        display_oie_triples(groups, sentences, oie_results_by_group)
    
    # Bước 10: Trực quan hóa ma trận
    if visualize:
        visualize_similarity_matrix(sim_matrix, groups, 
                                  title=f"Ma trận similarity - {document_id}")
    
    # Bước 11: Xuất kết quả ra file
    # Call the updated export_results_to_file
    export_results_to_file(document, sentences, groups, oie_results_by_group if extract_oie else None, document_id)
    
    # Bước 12: Lưu kết quả vào database nếu được yêu cầu
    if save_to_db:
        if extract_oie:
            # oie_results_by_group is already in the correct format for oie_sentence_groups
            save_to_database_with_oie(query, document, sentences, vectors, groups, all_triples_raw, oie_results_by_group)
        else:
            save_to_database(query, document, sentences, vectors, groups)
    
    return sentences, vectors, groups, all_triples_raw # Return all_triples_raw

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

def export_results_to_file(document: str, 
                           sentences: List[str], 
                           groups: List[List[int]], 
                           oie_results_by_group: Optional[List[List[Dict[str, str]]]] = None, 
                           document_id: Optional[str] = None):
    """
    Xuất kết quả phân nhóm và OIE (theo nhóm) ra file văn bản.
    """
    if document_id is None:
        # Fallback if document_id is not provided
        # Ensure hashlib is imported: import hashlib
        # Ensure time is imported: import time
        document_id = hashlib.md5(document[:1000].encode('utf-8')).hexdigest()[:8] 
        
    filename = generate_output_filename(document_id, prefix="grouping_analysis")
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(f"=== KẾT QUẢ PHÂN NHÓM NGỮ NGHĨA ===\\n")
            f.write(f"Tài liệu ID: {document_id}\\n")
            f.write(f"Thời gian xuất: {time.strftime('%Y-%m-%d %H:%M:%S')}\\n") # Ensure time is imported
            
            f.write("\\n--- VĂN BẢN GỐC ---\\n")
            f.write(document + "\\n")
            
            f.write("\\n--- CÁC CÂU ĐÃ TÁCH ---\\n")
            for i, sentence_text in enumerate(sentences): # Renamed sentence to sentence_text for clarity
                f.write(f"{i+1}. {sentence_text}\\n")
            
            f.write("\\n--- CÁC NHÓM NGỮ NGHĨA ---\\n")
            for i, group_indices in enumerate(groups):
                f.write(f"\\nNhóm {i+1} (gồm {len(group_indices)} câu):\\n")
                for idx in group_indices:
                    if idx < len(sentences): # Check index bounds
                        f.write(f"  - {sentences[idx]}\\n")
            
            # Ghi thông tin OIE nếu có
            if oie_results_by_group:
                f.write("\\n\\n=== OPEN INFORMATION EXTRACTION (OIE Results per Group) ===\\n")
                for i, group_oie_triples in enumerate(oie_results_by_group):
                    current_group_indices = groups[i] if i < len(groups) else []
                    group_sentences_list = [sentences[idx] for idx in current_group_indices if idx < len(sentences)]
                    group_text_preview = " ".join(group_sentences_list)[:70] + "...";
                    f.write(f"\\n--- Nhóm {i+1} (Văn bản nhóm: \"{group_text_preview}\") ---\\n") # Corrected syntax error

                    if group_oie_triples:
                        f.write(f"  Tìm thấy {len(group_oie_triples)} triple(s):\\n")
                        for triple_dict in group_oie_triples:
                            # Ensure format_oie_triples_to_string is available and works with a single triple dict in a list
                            formatted_triple_str = format_oie_triples_to_string([triple_dict]).strip()
                            f.write(f"    - {formatted_triple_str}\\n")
                    else:
                        f.write("  (Không có triples cho nhóm này hoặc có lỗi xảy ra)\\n")
                        
        print(f"Đã xuất kết quả phân tích chi tiết ra file: {filename}")
    except Exception as e:
        print(f"Lỗi khi xuất file: {e}")
        # import traceback # Consider importing traceback at the top
        # traceback.print_exc()
    # Ensure the try block has a corresponding except or finally, which it does now.

# --- ADD THE NEW CHUNKING FUNCTION ---
def semantic_chunk_passage_from_grouping_logic(
    passage_id: str,
    passage_text: str,
    model_name: str,
    initial_threshold: Union[str, float] = 'auto',
    decay_factor: float = 0.85,
    min_threshold: Union[str, float] = 'auto',
    initial_percentile: int = 95,
    min_percentile: int = 10,
    embedding_batch_size: int = 16, # Added batch size for embedding
    include_oie: bool = False, # Added include_oie flag
    device: Optional[object] = None, # ADDED DEVICE PARAMETER
    **kwargs # Catch unused args like simplify if passed
) -> List[Tuple[str, str, Optional[str]]]: # Return type includes optional OIE string
    """
    Phân đoạn văn bản dựa trên logic nhóm ngữ nghĩa, có tùy chọn OIE.
    """
    print(f"\nProcessing passage ID: {passage_id} with Semantic Grouping...")
    print(f"  Initial Threshold: {initial_threshold}, Decay: {decay_factor}, Min Threshold: {min_threshold}")
    if initial_threshold == 'auto': print(f"  Initial Percentile: {initial_percentile}")
    if min_threshold == 'auto': print(f"  Min Percentile: {min_percentile}")
    print(f"  Embedding Batch Size: {embedding_batch_size}")
    print(f"  Include OIE: {include_oie}")
    print(f"  Device: {device}") # ADDED FOR LOGGING
    
    # Bước 1: Trích xuất câu
    sentences = extract_and_simplify_sentences(passage_text, simplify=False)
    
    if not sentences or len(sentences) < 2:
        print("  Not enough sentences to perform grouping. Returning single chunk.")
        oie_string_single_chunk = None
        if include_oie and passage_text.strip():
            try:
                relations = extract_relations_from_paragraph(passage_text, use_enhanced_settings=True)
                if relations:
                    oie_string_single_chunk = format_oie_triples_to_string(relations)
            except Exception as e_oie_single:
                print(f"  Error during OIE for single chunk: {e_oie_single}")
        return [(f"{passage_id}_group_0", passage_text, oie_string_single_chunk)]
    
    # Bước 2: Tạo ma trận tương đồng ngữ nghĩa (sử dụng batch embedding)
    sim_matrix = create_semantic_matrix(sentences, model_name=model_name, batch_size_embed=embedding_batch_size, device=device) # PASS DEVICE
    if sim_matrix is None:
        print("  Failed to create similarity matrix. Returning single chunk.")
        oie_string_fail_matrix = None
        if include_oie and passage_text.strip():
            try:
                relations = extract_relations_from_paragraph(passage_text, use_enhanced_settings=True)
                if relations:
                    oie_string_fail_matrix = format_oie_triples_to_string(relations)
            except Exception as e_oie_fail_matrix:
                print(f"  Error during OIE for single chunk (matrix fail): {e_oie_fail_matrix}")
        return [(f"{passage_id}_group_0", passage_text, oie_string_fail_matrix)]
    
    # Bước 3: Xác định ngưỡng tự động nếu cần
    current_initial_threshold = initial_threshold
    current_min_threshold = min_threshold
    
    if initial_threshold == 'auto' or min_threshold == 'auto':
        upper_triangle_indices = np.triu_indices_from(sim_matrix, k=1)
        similarities = sim_matrix[upper_triangle_indices]
        
        if len(similarities) == 0:
            print("  Not enough similarity values for auto threshold. Returning single chunk.")
            oie_string_auto_thresh_fail = None
            if include_oie and passage_text.strip():
                try:
                    relations = extract_relations_from_paragraph(passage_text, use_enhanced_settings=True)
                    if relations:
                        oie_string_auto_thresh_fail = format_oie_triples_to_string(relations)
                except Exception as e_oie_auto_thresh_fail:
                    print(f"  Error during OIE for single chunk (auto thresh fail): {e_oie_auto_thresh_fail}")
            return [(f"{passage_id}_group_0", passage_text, oie_string_auto_thresh_fail)]
        
        if initial_threshold == 'auto':
            current_initial_threshold = np.percentile(similarities, initial_percentile)
            print(f"  Auto Initial Threshold ({initial_percentile}th percentile): {current_initial_threshold:.4f}")
        if min_threshold == 'auto':
            current_min_threshold = np.percentile(similarities, min_percentile)
            print(f"  Auto Min Threshold ({min_percentile}th percentile): {current_min_threshold:.4f}")
    
    try:
        final_initial_threshold = float(current_initial_threshold)
        final_min_threshold = float(current_min_threshold)
    except ValueError:
        print("  Error converting auto thresholds to float. Returning single chunk.")
        oie_string_float_conv_fail = None
        if include_oie and passage_text.strip():
            try:
                relations = extract_relations_from_paragraph(passage_text, use_enhanced_settings=True)
                if relations:
                    oie_string_float_conv_fail = format_oie_triples_to_string(relations)
            except Exception as e_oie_float_conv_fail:
                print(f"  Error during OIE for single chunk (float conv fail): {e_oie_float_conv_fail}")
        return [(f"{passage_id}_group_0", passage_text, oie_string_float_conv_fail)]
    
    # Bước 4: Nhóm câu
    groups = []
    current_group = [0]
    threshold = final_initial_threshold
    
    for i in range(len(sentences) - 1):
        similarity = sim_matrix[i, i+1]
        if similarity >= threshold:
            current_group.append(i+1)
        else:
            groups.append(current_group)
            current_group = [i+1]
            threshold = max(final_min_threshold, threshold * decay_factor)
    groups.append(current_group)
    
    # Bước 5: Tạo chunks từ các nhóm và tùy chọn thêm OIE
    final_chunks = []
    for i, group_indices in enumerate(groups):
        chunk_sentences = [sentences[idx] for idx in group_indices]
        chunk_text = " ".join(chunk_sentences)
        
        oie_string_for_chunk = None
        if include_oie and chunk_text.strip():
            try:
                relations = extract_relations_from_paragraph(chunk_text, use_enhanced_settings=True)
                if relations:
                    oie_string_for_chunk = format_oie_triples_to_string(relations)
            except Exception as e_oie_group:
                print(f"  Error during OIE extraction for group {i}: {e_oie_group}")
        
        chunk_id = f"{passage_id}_group_{i}"
        final_chunks.append((chunk_id, chunk_text, oie_string_for_chunk))
    
    del sentences
    del sim_matrix
    if 'similarities' in locals(): del similarities
    del groups
    gc.collect()
    
    print(f"  Passage {passage_id} chunked into {len(final_chunks)} groups.")
    return final_chunks

# --- Helper function to format OIE triples (needed locally) ---
def format_oie_triples_to_string(triples_list: List[Dict[str, str]]) -> str:
    if not triples_list:
        return ""
    formatted_triples = []
    for triple in triples_list:
        s = str(triple.get('subject', '')).replace('\\t', ' ').replace('\\n', ' ').strip()
        r = str(triple.get('relation', '')).replace('\\t', ' ').replace('\\n', ' ').strip()
        o = str(triple.get('object', '')).replace('\\t', ' ').replace('\\n', ' ').strip()
        if s and r and o: # Only include complete triples
            formatted_triples.append(f"({s}; {r}; {o})")
    if not formatted_triples:
        return ""
    return " [OIE_TRIPLES] " + " | ".join(formatted_triples) + " [/OIE_TRIPLES]"

# --- Main function for testing (if needed) ---
if __name__ == '__main__':
    # Ví dụ kiểm thử
    sample_passage = "The quick brown fox jumps over the lazy dog. The dog barks loudly. A cat watches from a distance."
    passage_id = "sample_001"
    model_name_test = "sentence-transformers/all-MiniLM-L6-v2" # Thay bằng model bạn dùng

    print("Testing Semantic Grouping without OIE...")
    chunks_no_oie = semantic_chunk_passage_from_grouping_logic(
        passage_id, sample_passage, model_name_test,
        initial_threshold=0.6, decay_factor=0.9, min_threshold=0.3,
        embedding_batch_size=2, include_oie=False
    )
    for cid, text, oie_str in chunks_no_oie:
        print(f"  Chunk ID: {cid}")
        print(f"  Text: {text}")
        assert oie_str is None, "OIE string should be None when include_oie=False"
    print("Test without OIE completed.")

    print("\nTesting Semantic Grouping with OIE...")
    # Giả lập rằng Tool.OIE.extract_relations_from_paragraph đã được import và hoạt động
    # Bạn cần đảm bảo CORENLP_HOME được thiết lập đúng cách cho OIE hoạt động
    try:
        chunks_with_oie = semantic_chunk_passage_from_grouping_logic(
            passage_id, sample_passage, model_name_test,
            initial_threshold=0.6, decay_factor=0.9, min_threshold=0.3,
            embedding_batch_size=2, include_oie=True
        )
        for cid, text, oie_str in chunks_with_oie:
            print(f"  Chunk ID: {cid}")
            print(f"  Text: {text}")
            print(f"  OIE: {oie_str}")
            if oie_str is not None:
                assert "[OIE_TRIPLES]" in oie_str, "OIE string format error"
        print("Test with OIE completed.")
    except Exception as e:
        print(f"Error during OIE test: {e}. Ensure Stanford CoreNLP is set up correctly.")

    # Test với ngưỡng tự động
    print("\nTesting Semantic Grouping with AUTO thresholds and OIE...")
    try:
        chunks_auto_oie = semantic_chunk_passage_from_grouping_logic(
            passage_id, sample_passage, model_name_test,
            initial_threshold='auto', decay_factor=0.8, min_threshold='auto',
            initial_percentile=90, min_percentile=20,
            embedding_batch_size=2, include_oie=True
        )
        for cid, text, oie_str in chunks_auto_oie:
            print(f"  Chunk ID: {cid}")
            print(f"  Text: {text}")
            print(f"  OIE: {oie_str}")
        print("Test with AUTO thresholds and OIE completed.")
    except Exception as e:
        print(f"Error during AUTO OIE test: {e}. Ensure Stanford CoreNLP is set up correctly.")

    # Test với một câu duy nhất
    single_sentence_passage = "This is a single sentence."
    print("\nTesting with a single sentence...")
    chunks_single = semantic_chunk_passage_from_grouping_logic(
        "single_001", single_sentence_passage, model_name_test, include_oie=True
    )
    assert len(chunks_single) == 1, "Single sentence should result in one chunk"
    print(f"  Chunk Text: {chunks_single[0][1]}")
    print(f"  OIE: {chunks_single[0][2]}")
    print("Test with single sentence completed.")

    # Test với không có câu nào
    empty_passage = "    "
    print("\nTesting with an empty passage...")
    chunks_empty = semantic_chunk_passage_from_grouping_logic(
        "empty_001", empty_passage, model_name_test, include_oie=True
    )
    assert len(chunks_empty) == 1, "Empty passage should result in one chunk"
    assert chunks_empty[0][1].strip() == "", "Chunk text for empty passage should be empty"
    print("Test with empty passage completed.")

    print("\nAll tests completed.")

