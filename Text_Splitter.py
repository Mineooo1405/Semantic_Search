import os
import re
import nltk
import time
import json
from typing import List, Dict, Union, Optional, Callable
from dotenv import load_dotenv
from Tool.Sentence_Detector import sentence_detector
from Tool.Database import connect_to_db

# Download NLTK data for sentence tokenization if not already present
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

load_dotenv()

#=============== CÁC HÀM TIỆN ÍCH CƠ BẢN ====================

def to_sentences(passage):
    """
    Tách đoạn văn thành các câu riêng biệt
    
    Args:
        passage: Đoạn văn bản cần tách câu
        
    Returns:
        list: Danh sách các câu
    """
    return sentence_detector(passage)

def calculate_chunk_stats(chunks):
    """
    Tính toán thống kê về các chunk được tạo ra
    
    Args:
        chunks: Danh sách các chunk
        
    Returns:
        dict: Thống kê về các chunk
    """
    stats = {
        'count': len(chunks),
        'avg_length': sum(len(c) for c in chunks) / len(chunks) if chunks else 0,
        'min_length': min(len(c) for c in chunks) if chunks else 0,
        'max_length': max(len(c) for c in chunks) if chunks else 0
    }
    return stats

#=============== THUẬT TOÁN CHIA VĂN BẢN ===================

def split_by_character(text, chunk_size=1000, chunk_overlap=200):
    """
    Chia văn bản thành các đoạn dựa trên số ký tự
    
    Args:
        text: Văn bản cần chia
        chunk_size: Kích thước tối đa của mỗi chunk (ký tự)
        chunk_overlap: Độ chồng lấp giữa các chunk liên tiếp
        
    Returns:
        list: Danh sách các chunk văn bản
    """
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        # Tính vị trí kết thúc của chunk hiện tại
        end = min(start + chunk_size, len(text))
        
        # Nếu không phải chunk cuối và còn văn bản
        if end < len(text):
            # Tìm vị trí xuống dòng, dấu chấm hoặc khoảng trắng cuối cùng trong chunk
            last_newline = text.rfind('\n', start, end)
            last_period = text.rfind('. ', start, end)
            last_space = text.rfind(' ', start, end)
            
            # Chọn điểm ngắt phù hợp nhất
            if last_newline > start + chunk_size * 0.7:
                end = last_newline + 1  # +1 để bao gồm cả ký tự xuống dòng
            elif last_period > start + chunk_size * 0.7:
                end = last_period + 2  # +2 để bao gồm cả dấu chấm và khoảng trắng
            elif last_space > start:
                end = last_space + 1  # +1 để bao gồm cả khoảng trắng
        
        # Thêm chunk vào kết quả
        chunks.append(text[start:end])
        
        # Tính vị trí bắt đầu của chunk tiếp theo, có tính đến overlap
        start = max(start, end - chunk_overlap)
    
    # Hiển thị thống kê
    stats = calculate_chunk_stats(chunks)
    print(f"[INFO] Chia theo ký tự: tạo {stats['count']} chunk")
    print(f"[INFO] Độ dài trung bình: {stats['avg_length']:.1f} ký tự")
    
    return chunks

def split_by_sentence(text, chunk_size=1000, chunk_overlap=200):
    """
    Chia văn bản thành các đoạn dựa trên ranh giới câu
    
    Args:
        text: Văn bản cần chia
        chunk_size: Kích thước tối đa của mỗi chunk (ký tự)
        chunk_overlap: Số câu chồng lấp giữa các chunk liên tiếp
        
    Returns:
        list: Danh sách các chunk văn bản
    """
    # Tách câu
    sentences = to_sentences(text)
    print(f"[INFO] Đã tách thành {len(sentences)} câu")
    
    if not sentences:
        return []
    
    if len(text) <= chunk_size:
        return [text]
    
    # Tính kích thước của các câu
    sentence_lengths = [len(s) for s in sentences]
    
    chunks = []
    current_chunk = []
    current_size = 0
    
    for i, sentence in enumerate(sentences):
        sentence_len = sentence_lengths[i]
        
        # Nếu câu hiện tại quá dài, chia nhỏ câu đó
        if sentence_len > chunk_size:
            # Xử lý chunk hiện tại trước
            if current_chunk:
                chunks.append(' '.join(current_chunk))
                current_chunk = []
                current_size = 0
            
            # Chia câu dài thành các phần nhỏ hơn
            sub_chunks = split_by_character(sentence, chunk_size, chunk_overlap)
            chunks.extend(sub_chunks)
            continue
        
        # Nếu thêm câu hiện tại vào chunk sẽ vượt quá kích thước tối đa
        if current_size + sentence_len + 1 > chunk_size and current_chunk:  # +1 cho khoảng trắng
            chunks.append(' '.join(current_chunk))
            
            # Tính toán số câu cần giữ lại cho overlap
            if chunk_overlap > 0:
                # Số câu chồng lấp dựa trên kích thước
                overlap_size = 0
                overlap_sentences = []
                
                for s in reversed(current_chunk):
                    s_len = len(s)
                    if overlap_size + s_len > chunk_overlap:
                        break
                    overlap_sentences.insert(0, s)
                    overlap_size += s_len + 1  # +1 cho khoảng trắng
                
                current_chunk = overlap_sentences
                current_size = overlap_size
            else:
                current_chunk = []
                current_size = 0
        
        # Thêm câu hiện tại vào chunk
        current_chunk.append(sentence)
        current_size += sentence_len + 1  # +1 cho khoảng trắng
    
    # Thêm chunk cuối cùng nếu có
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    # Hiển thị thống kê
    stats = calculate_chunk_stats(chunks)
    print(f"[INFO] Chia theo câu: tạo {stats['count']} chunk")
    print(f"[INFO] Độ dài trung bình: {stats['avg_length']:.1f} ký tự")
    
    return chunks

def split_by_paragraph(text, chunk_size=1000, chunk_overlap=200):
    """
    Chia văn bản thành các đoạn dựa trên đoạn văn
    
    Args:
        text: Văn bản cần chia
        chunk_size: Kích thước tối đa của mỗi chunk (ký tự)
        chunk_overlap: Độ chồng lấp giữa các chunk liên tiếp
        
    Returns:
        list: Danh sách các chunk văn bản
    """
    if len(text) <= chunk_size:
        return [text]
    
    # Chia thành các đoạn văn dựa trên dòng trống
    paragraphs = re.split(r'\n\s*\n', text)
    print(f"[INFO] Đã tách thành {len(paragraphs)} đoạn văn")
    
    chunks = []
    current_chunk = []
    current_size = 0
    
    for paragraph in paragraphs:
        paragraph = paragraph.strip()
        if not paragraph:
            continue
            
        paragraph_len = len(paragraph)
        
        # Nếu đoạn văn hiện tại quá dài, chia nhỏ đoạn đó
        if paragraph_len > chunk_size:
            # Xử lý chunk hiện tại trước
            if current_chunk:
                chunks.append('\n\n'.join(current_chunk))
                current_chunk = []
                current_size = 0
            
            # Chia đoạn văn dài thành các phần nhỏ hơn
            sub_chunks = split_by_sentence(paragraph, chunk_size, chunk_overlap)
            chunks.extend(sub_chunks)
            continue
        
        # Nếu thêm đoạn văn hiện tại vào chunk sẽ vượt quá kích thước tối đa
        if current_size + paragraph_len + 4 > chunk_size and current_chunk:  # +4 cho "\n\n"
            chunks.append('\n\n'.join(current_chunk))
            current_chunk = []
            current_size = 0
        
        # Thêm đoạn văn hiện tại vào chunk
        current_chunk.append(paragraph)
        current_size += paragraph_len + 4  # +4 cho "\n\n"
    
    # Thêm chunk cuối cùng nếu có
    if current_chunk:
        chunks.append('\n\n'.join(current_chunk))
    
    # Hiển thị thống kê
    stats = calculate_chunk_stats(chunks)
    print(f"[INFO] Chia theo đoạn văn: tạo {stats['count']} chunk")
    print(f"[INFO] Độ dài trung bình: {stats['avg_length']:.1f} ký tự")
    
    return chunks

def split_recursively(text, chunk_size=1000, chunk_overlap=200):
    """
    Chia văn bản đệ quy theo thứ tự ưu tiên: đoạn văn -> câu -> ký tự
    
    Args:
        text: Văn bản cần chia
        chunk_size: Kích thước tối đa của mỗi chunk (ký tự)
        chunk_overlap: Độ chồng lấp giữa các chunk liên tiếp
        
    Returns:
        list: Danh sách các chunk văn bản
    """
    if len(text) <= chunk_size:
        return [text]
    
    # Thử chia theo đoạn văn
    if '\n\n' in text:
        chunks = split_by_paragraph(text, chunk_size, chunk_overlap)
        if len(chunks) > 1:
            return chunks
    
    # Thử chia theo câu
    chunks = split_by_sentence(text, chunk_size, chunk_overlap)
    if len(chunks) > 1:
        return chunks
    
    # Chia theo ký tự (cách cuối cùng)
    return split_by_character(text, chunk_size, chunk_overlap)

#=============== XỬ LÝ VĂN BẢN ===================

def process_document(document, method="sentence", chunk_size=1000, chunk_overlap=200, document_id="Unknown", export_file=True):
    """
    Xử lý văn bản: tách thành các chunk theo phương pháp đã chọn
    
    Args:
        document: Văn bản cần xử lý
        method: Phương pháp chia ("character", "sentence", "paragraph", "recursive")
        chunk_size: Kích thước tối đa của mỗi chunk (ký tự)
        chunk_overlap: Độ chồng lấp giữa các chunk liên tiếp
        document_id: ID của văn bản để sử dụng khi lưu file
        export_file: Có xuất kết quả ra file hay không
        
    Returns:
        list: Danh sách các chunk văn bản
    """
    print(f"\n[PROCESS] Đang xử lý văn bản {document_id}")
    print(f"[INFO] Kích thước văn bản: {len(document)} ký tự")
    print(f"[INFO] Phương pháp chia: {method}")
    print(f"[INFO] Kích thước chunk tối đa: {chunk_size}")
    print(f"[INFO] Độ chồng lấp: {chunk_overlap}")
    
    start_time = time.time()
    
    # Chọn thuật toán phù hợp
    if method == "character":
        chunks = split_by_character(document, chunk_size, chunk_overlap)
    elif method == "paragraph":
        chunks = split_by_paragraph(document, chunk_size, chunk_overlap)
    elif method == "recursive":
        chunks = split_recursively(document, chunk_size, chunk_overlap)
    else:  # Mặc định là "sentence"
        chunks = split_by_sentence(document, chunk_size, chunk_overlap)
    
    elapsed_time = time.time() - start_time
    
    # Hiển thị kết quả
    print(f"[SUCCESS] Đã chia thành {len(chunks)} chunk trong {elapsed_time:.2f}s")
    display_chunks(chunks)
    
    # Xuất kết quả ra file
    if export_file:
        export_chunks_to_file(document, chunks, document_id)
    
    return chunks

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

def export_chunks_to_file(document, chunks, document_id):
    """
    Xuất các chunk ra file văn bản
    
    Args:
        document: Văn bản gốc
        chunks: Danh sách các chunk
        document_id: ID của văn bản
    """
    filename = f"text_chunks_{document_id.replace(' ', '_')}.txt"
    
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(f"KẾT QUẢ CHIA VĂN BẢN THÀNH CHUNKS\n")
            f.write(f"Thời gian: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Văn bản: {document_id}\n")
            f.write(f"Kích thước văn bản: {len(document)} ký tự\n")
            f.write(f"Số chunk: {len(chunks)}\n\n")
            
            for i, chunk in enumerate(chunks):
                f.write(f"=== CHUNK {i+1} ({len(chunk)} ký tự) ===\n")
                f.write(chunk)
                f.write("\n\n" + "="*50 + "\n\n")
                
        print(f"[INFO] Đã xuất kết quả ra file: {filename}")
    except Exception as e:
        print(f"[ERROR] Lỗi khi xuất ra file: {e}")

def save_to_database(document, chunks, method, chunk_size, chunk_overlap, document_id="Unknown"):
    """
    Lưu văn bản và các chunk vào database
    
    Args:
        document: Văn bản gốc
        chunks: Danh sách các chunk
        method: Phương pháp chia
        chunk_size: Kích thước tối đa của mỗi chunk
        chunk_overlap: Độ chồng lấp giữa các chunk
        document_id: ID của văn bản
        
    Returns:
        int: ID của bản ghi được thêm vào database
    """
    conn = connect_to_db()
    cursor = conn.cursor()
    
    # Tách văn bản gốc thành các câu
    sentences = to_sentences(document)
    
    # Chuyển đổi thành định dạng JSON để lưu vào database
    sentences_json = json.dumps(sentences)
    
    # Tạo đối tượng chunking chứa cả chunks và metadata
    chunking_data = {
        "chunks": chunks,
        "metadata": {
            "method": method,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "chunk_count": len(chunks)
        }
    }
    chunking_json = json.dumps(chunking_data)
    
    try:
        cursor.execute(
            """
            INSERT INTO Text_Splitter 
            (query, Original_Paragraph, Sentences, Chunking)
            VALUES (%s, %s, %s, %s)
            RETURNING id
            """,
            (
                document_id,  # Sử dụng document_id làm query
                document,     # Văn bản gốc
                sentences_json,  # Danh sách các câu
                chunking_json    # Kết quả phân đoạn và metadata
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
    method, chunk_size, chunk_overlap = get_user_parameters()
    
    # Xử lý văn bản
    chunks = process_document(
        text, 
        method=method, 
        chunk_size=chunk_size, 
        chunk_overlap=chunk_overlap,
        document_id=document_id
    )
    
    # Hỏi người dùng có muốn lưu vào database không
    save_to_db = input("\n[INPUT] Bạn có muốn lưu kết quả vào database không? (y/n): ").lower() == 'y'
    if save_to_db:
        save_to_database(text, chunks, method, chunk_size, chunk_overlap, document_id)

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
        method, chunk_size, chunk_overlap = get_user_parameters()
        
        # Xử lý văn bản
        chunks = process_document(
            text, 
            method=method, 
            chunk_size=chunk_size, 
            chunk_overlap=chunk_overlap,
            document_id=document_id
        )
        
        # Hỏi người dùng có muốn lưu vào database không
        save_to_db = input("\n[INPUT] Bạn có muốn lưu kết quả vào database không? (y/n): ").lower() == 'y'
        if save_to_db:
            save_to_database(text, chunks, method, chunk_size, chunk_overlap, document_id)
            
    except Exception as e:
        print(f"[ERROR] Lỗi khi đọc file: {e}")

def get_user_parameters():
    """
    Lấy các tham số từ người dùng
    
    Returns:
        tuple: (method, chunk_size, chunk_overlap)
    """
    print("\n[INPUT] Chọn phương pháp chia văn bản:")
    print("1. Chia theo ký tự (character)")
    print("2. Chia theo câu (sentence - mặc định)")
    print("3. Chia theo đoạn văn (paragraph)")
    print("4. Chia đệ quy (recursive)")
    
    choice = input("[INPUT] Lựa chọn của bạn (1/2/3/4): ")
    
    # Xác định phương pháp
    method_map = {
        '1': 'character',
        '2': 'sentence',
        '3': 'paragraph',
        '4': 'recursive',
    }
    method = method_map.get(choice, 'sentence')  # Mặc định là sentence
    
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
    
    return method, chunk_size, chunk_overlap

#=============== MAIN ===================

if __name__ == "__main__":
    print("[TITLE] CHIA VĂN BẢN THÀNH CÁC CHUNKS DỰA TRÊN KÍCH THƯỚC")
    print("[MENU] 1. Nhập văn bản thủ công")
    print("[MENU] 2. Tải văn bản từ file")
    print("[MENU] 3. Xử lý tập dữ liệu")
    
    choice = input("\n[INPUT] Lựa chọn của bạn (1/2/3): ")
    
    if choice == '1':
        test_with_manual_input()
    elif choice == '2':
        test_with_file()
    elif choice == '3':
        # Triển khai chức năng xử lý tập dữ liệu ở đây
        print("[FEATURE] Chức năng đang được phát triển")
    else:
        print("[ERROR] Lựa chọn không hợp lệ")