import os
import re
import nltk
import time
import json
import pandas as pd
from typing import List, Dict, Union, Optional, Callable
from dotenv import load_dotenv
from Tool.Sentence_Detector import sentence_detector, enhanced_sentence_detector, sentence_splitter_for_oie, spacy_sentence_splitter
from Tool.Database import connect_to_db
from Tool.OIE import extract_triples_for_search

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
    _, sub_sentences = spacy_sentence_splitter(passage)
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

#=============== XỬ LÝ VĂN BẢN ===================

def process_document(document, chunk_size=1000, chunk_overlap=200, document_id="Unknown", export_file=True):
    """
    Xử lý văn bản: tách thành các chunk theo câu và trích xuất quan hệ OpenIE
    
    Args:
        document: Văn bản cần xử lý
        chunk_size: Kích thước tối đa của mỗi chunk (ký tự) 
        chunk_overlap: Độ chồng lấp giữa các chunk liên tiếp
        document_id: ID của văn bản để sử dụng khi lưu file
        export_file: Có xuất kết quả ra file hay không
        
    Returns:
        dict: Kết quả xử lý bao gồm chunks, sentences, sentence_groups, và triples
    """
    print(f"\n[PROCESS] Đang xử lý văn bản {document_id}")
    print(f"[INFO] Kích thước văn bản: {len(document)} ký tự")
    print(f"[INFO] Kích thước chunk tối đa: {chunk_size}")
    print(f"[INFO] Độ chồng lấp: {chunk_overlap}")
    
    start_time = time.time()
    
    # Chia văn bản thành các chunk dựa trên câu
    chunks, sentences, sentence_groups = split_by_sentence(document, chunk_size, chunk_overlap)
    
    # Trích xuất OpenIE triples từ từng câu
    print(f"[INFO] Đang trích xuất quan hệ OpenIE từ {len(sentences)} câu...")
    
    # Cấu trúc để lưu trữ OIE triples
    all_triples = []  # Danh sách phẳng tất cả các triples
    oie_sentence_groups = []  # Cấu trúc phân cấp [sentence_group][sentence_idx][triple]
    
    # Trích xuất triples cho mỗi nhóm câu
    for group in sentence_groups:
        group_triples = []
        for sentence_idx in group:
            sentence = sentences[sentence_idx]
            # Trích xuất triples từ một câu
            sentence_triples = extract_triples_for_search(sentence)
            group_triples.append(sentence_triples)
            all_triples.extend(sentence_triples)
        oie_sentence_groups.append(group_triples)
    
    elapsed_time = time.time() - start_time
    
    # Hiển thị kết quả
    print(f"[SUCCESS] Đã xử lý trong {elapsed_time:.2f}s")
    print(f"[SUCCESS] Đã tách thành {len(chunks)} chunk")
    print(f"[SUCCESS] Đã trích xuất {len(all_triples)} quan hệ")
    
    # Hiển thị chi tiết các chunk
    display_chunks(chunks)
    
    # Xuất kết quả ra file
    if export_file:
        export_results_to_file(document, chunks, sentences, sentence_groups, all_triples, oie_sentence_groups, document_id)
    
    # Đóng gói kết quả
    results = {
        'chunks': chunks,
        'sentences': sentences,
        'sentence_groups': sentence_groups,
        'all_triples': all_triples,
        'oie_sentence_groups': oie_sentence_groups
    }
    
    return results

def display_chunks(chunks, max_preview=70):
    """
    Hiển thị danh sách các chunk
    
    Args:
        chunks: Danh sách các chunk
        max_preview: Độ dài tối đa để hiển thị preview của mỗi chunk
    """
    print("\n[RESULT] Chi tiết các chunk:")
    
    for i, chunk in enumerate(chunks):
        # Chuẩn bị preview: thay thế newline với khoảng trắng
        preview = chunk.replace('\n', ' ').strip()
        if len(preview) > max_preview:
            preview = preview[:max_preview] + "..."
            
        print(f"[CHUNK {i+1}] ({len(chunk)} ký tự): {preview}")

def export_results_to_file(document, chunks, sentences, sentence_groups, all_triples, oie_sentence_groups, document_id):
    """
    Xuất các kết quả ra file văn bản
    
    Args:
        document: Văn bản gốc
        chunks: Danh sách các chunk
        sentences: Danh sách các câu gốc
        sentence_groups: Các nhóm câu theo chunk
        all_triples: Tất cả các triples được trích xuất
        oie_sentence_groups: Cấu trúc phân cấp các triples theo nhóm câu
        document_id: ID của văn bản
    """
    filename = f"text_chunks_{document_id.replace(' ', '_')}.txt"
    
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(f"KẾT QUẢ CHIA VĂN BẢN THÀNH CHUNKS VÀ OIE\n")
            f.write(f"Thời gian: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Văn bản: {document_id}\n")
            f.write(f"Kích thước văn bản: {len(document)} ký tự\n")
            f.write(f"Số câu: {len(sentences)}\n")
            f.write(f"Số chunk: {len(chunks)}\n")
            f.write(f"Số quan hệ OIE: {len(all_triples)}\n\n")
            
            for i, chunk in enumerate(chunks):
                f.write(f"=== CHUNK {i+1} ({len(chunk)} ký tự) ===\n")
                f.write(chunk)
                f.write("\n\n")
                
                # Hiển thị các câu trong chunk
                f.write(f"Các câu trong chunk {i+1}:\n")
                for idx in sentence_groups[i]:
                    f.write(f"  [{idx}] {sentences[idx]}\n")
                f.write("\n")
                
                # Hiển thị các triples từ các câu trong chunk
                f.write(f"Các quan hệ OIE trong chunk {i+1}:\n")
                for j, sentence_idx in enumerate(sentence_groups[i]):
                    f.write(f"  Câu {sentence_idx}: {sentences[sentence_idx]}\n")
                    for triple in oie_sentence_groups[i][j]:
                        f.write(f"    - {triple['subject']} | {triple['relation']} | {triple['object']}\n")
                f.write("\n")
                
                f.write("="*50 + "\n\n")
                
        print(f"[INFO] Đã xuất kết quả ra file: {filename}")
    except Exception as e:
        print(f"[ERROR] Lỗi khi xuất ra file: {e}")

def save_to_database(document, results, document_id="Unknown"):
    """
    Lưu văn bản, các chunk và triples OIE vào database
    
    Args:
        document: Văn bản gốc
        results: Kết quả xử lý từ hàm process_document
        document_id: ID của văn bản
        
    Returns:
        int: ID của bản ghi được thêm vào database
    """
    conn = connect_to_db()
    cursor = conn.cursor()
    
    # Lấy dữ liệu từ kết quả
    chunks = results['chunks']
    sentences = results['sentences']
    sentence_groups = results['sentence_groups']
    all_triples = results['all_triples']
    oie_sentence_groups = results['oie_sentence_groups']
    
    # Chuyển đổi thành định dạng JSON để lưu vào database
    sentences_json = json.dumps(sentences)
    
    # Tạo đối tượng chunking chứa cả chunks và metadata
    chunking_data = {
        "chunks": chunks,
        "metadata": {
            "method": "sentence",
            "chunk_count": len(chunks),
            "sentence_groups": sentence_groups
        }
    }
    chunking_json = json.dumps(chunking_data)
    
    # JSON cho OpenIE triples
    all_triples_json = json.dumps(all_triples)
    oie_sentence_groups_json = json.dumps(oie_sentence_groups)
    
    try:
        cursor.execute(
            """
            INSERT INTO Text_Splitter 
            (query, Original_Paragraph, Sentences, Chunking, OIE_Triples, OIE_Sentence_Groups)
            VALUES (%s, %s, %s, %s, %s, %s)
            RETURNING id
            """,
            (
                document_id,  # Sử dụng document_id làm query
                document,     # Văn bản gốc
                sentences_json,  # Danh sách các câu
                chunking_json,   # Kết quả phân đoạn và metadata
                all_triples_json,  # Tất cả các OIE triples
                oie_sentence_groups_json  # Cấu trúc phân cấp các triples
            )
        )
        
        inserted_id = cursor.fetchone()[0]
        conn.commit()
        print(f"[DB] Đã lưu kết quả vào database với ID: {inserted_id}")
        return inserted_id
        
    except Exception as e:
        conn.rollback()
        print(f"[ERROR] Lỗi khi lưu vào database: {e}")
        return None
    finally:
        cursor.close()
        conn.close()

#=============== CHỨC NĂNG XỬ LÝ DATASET ===================

def process_dataset(file_path="passages_1000.csv", num_passages=10, save_to_db=True):
    """
    Xử lý một tập dữ liệu passages từ file CSV
    
    Args:
        file_path: Đường dẫn đến file CSV
        num_passages: Số lượng passages muốn xử lý
        save_to_db: Có lưu kết quả vào database không
        
    Returns:
        list: Danh sách các ID đã được lưu trong database
    """
    print(f"\n[DATASET] Đang xử lý dataset từ file {file_path}")
    
    try:
        # Đọc file CSV
        df = pd.read_csv(file_path)
        passages = df['passage_text'].tolist()
        
        # Giới hạn số lượng
        if num_passages > len(passages):
            num_passages = len(passages)
            print(f"[INFO] Dataset chỉ có {len(passages)} passages, sẽ xử lý tất cả")
        else:
            passages = passages[:num_passages]
            print(f"[INFO] Sẽ xử lý {num_passages} passages đầu tiên")
        
        # Xử lý từng passage
        db_ids = []
        for i, passage in enumerate(passages):
            print(f"\n[PASSAGE {i+1}/{num_passages}] Đang xử lý...")
            
            # Tạo ID cho passage
            passage_id = f"passage_{i+1}"
            
            # Xử lý passage
            results = process_document(
                document=passage,
                chunk_size=1000,  # Giá trị mặc định
                chunk_overlap=200,  # Giá trị mặc định
                document_id=passage_id,
                export_file=(i < 5)  # Chỉ xuất 5 passage đầu tiên ra file
            )
            
            # Lưu vào database
            if save_to_db:
                db_id = save_to_database(passage, results, passage_id)
                if db_id:
                    db_ids.append(db_id)
            
            # In tiến độ
            print(f"[PROGRESS] Đã xử lý {i+1}/{num_passages} passages ({(i+1)/num_passages*100:.1f}%)")
        
        print(f"[SUCCESS] Đã xử lý xong {num_passages} passages")
        if save_to_db:
            print(f"[DB] Đã lưu {len(db_ids)} passages vào database")
        
        return db_ids
        
    except Exception as e:
        print(f"[ERROR] Lỗi khi xử lý dataset: {e}")
        return []

#=============== CHỨC NĂNG KIỂM THỬ ===================

def test_with_manual_input():
    """
    Cho phép người dùng nhập văn bản trực tiếp và kiểm thử thuật toán
    """
    print("\n[TEST] TEST THỦ CÔNG VỚI VĂN BẢN TỰ NHẬP")
    print("[INFO] Nhập văn bản (kết thúc bằng \"###\" trên một dòng riêng):")
    
    lines = []
    while True:
        line = input()
        if line.strip() == "###":
            break
        lines.append(line)
    
    if not lines:
        print("[ERROR] Không có nội dung được nhập, hủy test")
        return
    
    text = "\n".join(lines)
    document_id = f"Manual_{int(time.time())}"
    
    # Lấy các tham số từ người dùng
    chunk_size, chunk_overlap = get_user_parameters()
    
    # Xử lý văn bản
    results = process_document(
        document=text, 
        chunk_size=chunk_size, 
        chunk_overlap=chunk_overlap,
        document_id=document_id
    )
    
    # Hỏi người dùng có muốn lưu vào database không
    save_to_db = input("\n[INPUT] Bạn có muốn lưu kết quả vào database không? (y/n): ").lower() == 'y'
    if save_to_db:
        save_to_database(text, results, document_id)

def test_with_file():
    """
    Tải văn bản từ file và kiểm thử thuật toán
    """
    print("\n[FILE] TẢI VĂN BẢN TỪ FILE")
    file_path = input("[INPUT] Nhập đường dẫn đến file văn bản: ")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
        
        document_id = os.path.basename(file_path)
        print(f"[INFO] Đã tải thành công văn bản từ {file_path}")
        print(f"[INFO] Kích thước: {len(text)} ký tự")
        
        # Lấy các tham số từ người dùng
        chunk_size, chunk_overlap = get_user_parameters()
        
        # Xử lý văn bản
        results = process_document(
            document=text, 
            chunk_size=chunk_size, 
            chunk_overlap=chunk_overlap,
            document_id=document_id
        )
        
        # Hỏi người dùng có muốn lưu vào database không
        save_to_db = input("\n[INPUT] Bạn có muốn lưu kết quả vào database không? (y/n): ").lower() == 'y'
        if save_to_db:
            save_to_database(text, results, document_id)
            
    except Exception as e:
        print(f"[ERROR] Lỗi khi đọc file: {e}")

def get_user_parameters():
    """
    Lấy các tham số từ người dùng
    
    Returns:
        tuple: (chunk_size, chunk_overlap)
    """
    # Lấy kích thước chunk
    while True:
        try:
            chunk_size = int(input(f"[INPUT] Nhập kích thước tối đa cho mỗi chunk (100-10000, mặc định: 1000): ") or "1000")
            if 100 <= chunk_size <= 10000:
                break
            else:
                print("[ERROR] Kích thước phải nằm trong khoảng từ 100 đến 10000")
        except ValueError:
            print("[ERROR] Vui lòng nhập một số nguyên")
    
    # Lấy độ chồng lấp
    while True:
        try:
            default_overlap = min(200, int(chunk_size * 0.2))  # Mặc định là 20% của chunk_size
            chunk_overlap = int(input(f"[INPUT] Nhập độ chồng lấp giữa các chunk (0-{chunk_size//2}, mặc định: {default_overlap}): ") or str(default_overlap))
            if 0 <= chunk_overlap <= chunk_size // 2:
                break
            else:
                print(f"[ERROR] Độ chồng lấp phải nằm trong khoảng từ 0 đến {chunk_size//2}")
        except ValueError:
            print("[ERROR] Vui lòng nhập một số nguyên")
    
    return chunk_size, chunk_overlap

def process_dataset_ui():
    """
    Giao diện người dùng để xử lý dataset
    """
    print("\n[DATASET] XỬ LÝ TẬP DỮ LIỆU")
    
    # Hỏi đường dẫn file dataset
    default_path = "passages_1000.csv"
    file_path = input(f"[INPUT] Nhập đường dẫn đến file CSV (mặc định: {default_path}): ") or default_path
    
    # Kiểm tra file tồn tại
    if not os.path.exists(file_path):
        print(f"[ERROR] File {file_path} không tồn tại")
        return
    
    # Hỏi số lượng passages cần xử lý
    while True:
        try:
            num_passages = int(input("[INPUT] Nhập số lượng passages cần xử lý (mặc định: 10): ") or "10")
            if num_passages > 0:
                break
            else:
                print("[ERROR] Số lượng phải lớn hơn 0")
        except ValueError:
            print("[ERROR] Vui lòng nhập một số nguyên")
    
    # Hỏi có lưu vào database không
    save_to_db = input("[INPUT] Lưu kết quả vào database? (y/n, mặc định: y): ").lower() != 'n'
    
    # Xử lý dataset
    process_dataset(file_path, num_passages, save_to_db)

#=============== MAIN ===================

if __name__ == "__main__":
    print("[TITLE] CHIA VĂN BẢN THÀNH CÁC CÂU VÀ TRÍCH XUẤT QUAN HỆ OIE")
    print("[MENU] 1. Nhập văn bản thủ công")
    print("[MENU] 2. Tải văn bản từ file")
    print("[MENU] 3. Xử lý dataset CSV")
    
    choice = input("\n[INPUT] Lựa chọn của bạn (1/2/3): ")
    
    if choice == '1':
        test_with_manual_input()
    elif choice == '2':
        test_with_file()
    elif choice == '3':
        process_dataset_ui()
    else:
        print("[ERROR] Lựa chọn không hợp lệ")