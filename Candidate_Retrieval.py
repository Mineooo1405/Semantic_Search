import psycopg2
from pgvector.psycopg2 import register_vector
import numpy as np
from Tool.Sentence_Embedding import sentence_embedding as embed_text_list # Import hàm embedding
from Tool.Database import connect_to_db # Import hàm kết nối DB
import time
import re
from Semantic_Grouping import process_and_store_grouped_chunks
import pandas as pd

try:
    with open("Tool/input.txt", "r", encoding="utf-8") as file:
        full_text = file.read()
    # Tách thành các đoạn văn dựa trên một hoặc nhiều dòng trống
    # Loại bỏ các khoảng trắng thừa ở đầu/cuối mỗi đoạn và các đoạn rỗng
    passages = [p.strip() for p in re.split(r'\n\s*\n', full_text) if p.strip()]
    print(f"Đã trích xuất {len(passages)} đoạn văn từ Tool/input.txt") # Sửa thông báo
except FileNotFoundError:
    print("Lỗi: Không tìm thấy file Tool/input.txt")
    passages = [] # Khởi tạo list rỗng nếu lỗi
except Exception as e:
    print(f"Lỗi khi đọc file Tool/input.txt: {e}")
    passages = []

# Chỉ chạy xử lý nếu đọc được passages
if passages:
    
    try:
        conn_del = connect_to_db()
        cursor_del = conn_del.cursor()
        cursor_del.execute("DROP TABLE IF EXISTS semantic_grouping_chunks_vector;")
        conn_del.commit()
        print("Đã xóa bảng semantic_grouping_chunks_vector cũ (nếu tồn tại).")
        cursor_del.close()
        conn_del.close()
    except Exception as del_err:
        print(f"Lỗi khi xóa bảng cũ: {del_err}")

    process_and_store_grouped_chunks(
        passages,
        initial_threshold='auto',       
        decay_factor=0.7,              
        min_threshold='auto',         
        auto_percentiles=(95, 10),      
        model_name="thenlper/gte-large",
        create_index=True
    )
else:
    print("Không có đoạn văn nào để xử lý.")

#process_and_store_grouped_chunks(passages, 0.6, 0.7, 0.1,"thenlper/gte-large",True)

def find_similar_chunks(query_text: str, top_k: int = 5, model_name: str = "thenlper/gte-large", table_name: str = "semantic_grouping_chunks_vector"):
    """
    Tìm kiếm các chunk tương đồng nhất với query_text trong database sử dụng pgvector.

    Args:
        query_text (str): Câu truy vấn của người dùng.
        top_k (int): Số lượng chunk tương đồng nhất cần trả về.
        model_name (str): Tên mô hình embedding đã sử dụng để tạo vector cho chunk và query.
        table_name (str): Tên bảng chứa chunk và vector (đã dùng pgvector).

    Returns:
        list: Danh sách các tuple, mỗi tuple chứa (chunk_id, chunk_text, distance).
              Trả về list rỗng nếu có lỗi hoặc không tìm thấy kết quả.
    """
    print(f"\n--- Bắt đầu Candidate Retrieval cho query: '{query_text[:50]}...' ---")
    print(f"Sử dụng model: {model_name}, tìm top {top_k} chunks từ bảng '{table_name}'")
    start_time = time.time()

    conn = None
    cursor = None
    results = []

    try:
        # 1. Embed Query
        print("  Đang tạo embedding cho query...")
        query_vector_np = embed_text_list([query_text], model_name=model_name)
        if query_vector_np is None or len(query_vector_np) == 0:
            print("  Lỗi: Không thể tạo embedding cho query.")
            return []
        query_vector = query_vector_np[0]
        print(f"  Kích thước vector query: {query_vector.shape}")

        # 2. Kết nối DB và Đăng ký Vector
        print("  Đang kết nối database...")
        conn = connect_to_db()
        if conn is None:
            print("  Lỗi: Không thể kết nối database.")
            return []
        register_vector(conn)
        cursor = conn.cursor()
        print("  Kết nối database và đăng ký pgvector thành công.")

        # 3. Xây dựng và Thực thi Câu lệnh SQL
        # Sử dụng toán tử <=> (Cosine Distance)
        # Lưu ý: Khoảng cách Cosine càng nhỏ càng tốt (0 là giống hệt, 1 là khác biệt, 2 là đối nghịch)
        sql_query = f"""
        SELECT chunk_id, chunk_text, chunk_vector <=> %s AS distance
        FROM {table_name}
        ORDER BY distance ASC -- Sắp xếp theo khoảng cách tăng dần
        LIMIT %s;
        """
        print(f"  Đang thực thi truy vấn tìm kiếm vector...")
        query_start_time = time.time()
        # Truyền NumPy array trực tiếp làm tham số
        cursor.execute(sql_query, (query_vector, top_k))
        db_results = cursor.fetchall()
        query_duration = time.time() - query_start_time
        print(f"  Truy vấn database hoàn thành trong {query_duration:.4f}s. Tìm thấy {len(db_results)} kết quả.")

        # 4. Xử lý Kết quả
        if db_results:
            results = [(row[0], row[1], row[2]) for row in db_results] # (chunk_id, chunk_text, distance)

    except psycopg2.errors.UndefinedTable:
        print(f"  Lỗi: Bảng '{table_name}' không tồn tại. Hãy chạy bước lưu trữ chunk trước.")
        if conn: conn.rollback()
    except psycopg2.errors.UndefinedFunction:
         print(f"  Lỗi: Toán tử <=> hoặc kiểu vector không được định nghĩa. Đảm bảo extension pgvector đã được cài đặt và kích hoạt (`CREATE EXTENSION vector;`).")
         if conn: conn.rollback()
    except psycopg2.errors.InvalidTextRepresentation as e:
         print(f"  Lỗi: Định dạng vector không hợp lệ khi truy vấn. Chi tiết: {e}")
         if conn: conn.rollback()
    except (Exception, psycopg2.DatabaseError) as error:
        print(f"  Lỗi trong quá trình tìm kiếm: {error}")
        if conn: conn.rollback() # Rollback nếu có lỗi
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()
            print("  Đã đóng kết nối database.")

    total_duration = time.time() - start_time
    print(f"--- Candidate Retrieval hoàn thành trong {total_duration:.2f}s ---")
    return results

# --- Ví dụ sử dụng ---
if __name__ == "__main__":
    test_query = "What is the significance of the bluestones at Stonehenge?"
    # test_query = "Tell me about the construction phases."
    # test_query = "Who were the Druids and what was their connection?"

    # Gọi hàm tìm kiếm
    similar_chunks = find_similar_chunks(
        query_text=test_query,
        top_k=5,
        model_name="thenlper/gte-large", # Đảm bảo dùng đúng model đã lưu vector
        table_name="semantic_grouping_chunks_vector" # Đảm bảo đúng tên bảng
    )

    # In kết quả
    if similar_chunks:
        print("\n=== KẾT QUẢ TÌM KIẾM (TOP 5) ===")
        for i, (chunk_id, chunk_text, distance) in enumerate(similar_chunks):
            print(f"\n{i+1}. Chunk ID: {chunk_id} (Distance: {distance:.4f})")
            print(f"   Text: {chunk_text[:300]}...") # In 300 ký tự đầu
    else:
        print("\nKhông tìm thấy chunk tương đồng hoặc có lỗi xảy ra.")