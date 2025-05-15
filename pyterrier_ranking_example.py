# pyterrier_ranking_example.py
import pandas as pd
import pyterrier as pt
import os

# --- Khởi tạo PyTerrier ---
# Đảm bảo bạn đã cài đặt PyTerrier và các tiện ích transformers:
# pip install python-terrier pyterrier_transformers
# Yêu cầu Java JDK để chạy đầy đủ. Xem tài liệu PyTerrier để cài đặt.
# Việc khởi tạo có thể tải các thành phần cần thiết.
if not pt.started():
    # Thử khởi tạo không yêu cầu Terrier backend cho ví dụ re-ranking này
    # Nếu bạn cần các chức năng Terrier đầy đủ (như BM25), cần đảm bảo Terrier được cài đặt
    try:
        pt.init(boot_packages=["com.github.terrierteam:terrier-prf:-SNAPSHOT"])
    except Exception as e:
        print(f"PyTerrier init failed (might need Java/Terrier backend): {e}")
        print("Proceeding with transformer re-ranking only.")
        # Một số chức năng có thể không hoạt động nếu không có Terrier backend

# --- Dữ liệu mẫu ---
# Giả sử chúng ta có một truy vấn và một danh sách các tài liệu ứng viên cần xếp hạng
queries = pd.DataFrame([["q1", "what is semantic search?"]], columns=["qid", "query"])

# DataFrame chứa các tài liệu ứng viên (thường được lấy từ bước retrieval trước đó)
# Cần các cột 'qid', 'docno' (ID duy nhất của tài liệu), 'text'
docs_df = pd.DataFrame([
    ["d1", "Semantic search seeks to understand the intent behind a query."],
    ["d2", "Traditional keyword search matches exact words."],
    ["d3", "Vector databases store numerical representations of text."],
    ["d4", "Learning to rank models optimize document ordering."],
], columns=["docno", "text"])

# Giả lập kết quả từ bước retrieval ban đầu (ví dụ: BM25 hoặc bi-encoder)
# Thường sẽ có 'score' và 'rank' ban đầu, nhưng ở đây ta chỉ cần qid, query, docno, text
# để đưa vào re-ranker. CrossRanker sẽ tự tính score mới.
# Tạo input cho re-ranker bằng cách kết hợp query và docs
# (Trong pipeline thực tế, bạn sẽ nối retriever >> re-ranker)
initial_candidates = []
for i, q_row in queries.iterrows():
    for j, d_row in docs_df.iterrows():
        initial_candidates.append({
            "qid": q_row["qid"],
            "query": q_row["query"],
            "docno": d_row["docno"],
            "text": d_row["text"]
        })
initial_candidates_df = pd.DataFrame(initial_candidates)
print("--- Initial Candidates ---")
print(initial_candidates_df)

# --- Mô hình Re-ranking ---
# Sử dụng Cross-Encoder từ pyterrier_transformers (dựa trên Hugging Face)
# Chọn một mô hình cross-encoder đã huấn luyện trước (ví dụ: cho MS MARCO)
# Cần cài đặt: pip install torch # Hoặc tensorflow tùy mô hình
try:
    # pyterrier_transformers có thể cần được import tường minh
    # import pyterrier_transformers
    cross_encoder_model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    # CrossRanker nhận DataFrame với qid, query, docno, text và trả về DataFrame với score
    cross_ranker = pt.text.CrossRanker(cross_encoder_model_name, batch_size=4)
    print(f"\n--- Initialized CrossRanker with {cross_encoder_model_name} ---")

    # Áp dụng re-ranker trên các ứng viên ban đầu
    # Trong pipeline đầy đủ: final_results = retriever >> cross_ranker
    # Ở đây ta áp dụng trực tiếp lên DataFrame ứng viên
    reranked_results = cross_ranker(initial_candidates_df)

    # Sắp xếp kết quả theo điểm số mới từ cao đến thấp
    reranked_results_sorted = reranked_results.sort_values(by=['qid', 'score'], ascending=[True, False])

    print("\n--- Re-ranked Results (Sorted by Score) ---")
    print(reranked_results_sorted[['qid', 'query', 'docno', 'score', 'text']])

except ImportError:
    print("\nError: pyterrier_transformers not found or dependencies missing.")
    print("Please install it: pip install pyterrier_transformers")
except Exception as e:
    print(f"\nAn error occurred during re-ranking: {e}")
    print("Ensure the transformer model name is correct and libraries (torch/tf) are installed.")

print("\nPyTerrier example finished.") 