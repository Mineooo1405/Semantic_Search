# filepath: d:\SemanticSearch\train_model.py
import matchzoo as mz
import torch # Hoặc tensorflow tùy backend
import os
import pandas as pd
from matchzoo.engine.base_metric import Accuracy

# --- 1. Định nghĩa Tham số ---
TRAIN_DATA_PATH = "D:/SemanticSearch/TrainingData_MatchZoo_BEIR_Semantic/msmarco_semantic_train_triplets.tsv"
MODEL_OUTPUT_DIR = "D:/SemanticSearch/TrainedModels/my_semantic_model"
PRETRAINED_MODEL_NAME = "bert-base-uncased" # Hoặc model bạn muốn fine-tune (phải khớp với embedding model nếu có thể)
BATCH_SIZE = 16 # Điều chỉnh tùy theo GPU memory
EPOCHS = 3      # Số epochs huấn luyện
LEARNING_RATE = 2e-5

os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)

# --- 2. Chuẩn bị Dữ liệu ---
# MatchZoo thường cần định nghĩa cách đọc dữ liệu
# Tạo một DataPack từ file TSV (cần kiểm tra cách MatchZoo xử lý TSV trực tiếp hoặc cần chuyển đổi)

# Cách 1: Đọc thủ công và tạo DataPack (linh hoạt hơn)
print("Loading training data...")
train_data_list = []
# Đọc giới hạn dòng để test nhanh (bỏ limit khi chạy thật)
# limit = 1000
with open(TRAIN_DATA_PATH, 'r', encoding='utf-8') as f:
    for i, line in enumerate(f):
        # if i >= limit: break # Bỏ comment để giới hạn test
        parts = line.strip().split('\t')
        if len(parts) == 3:
            # MatchZoo thường cần 'text_left', 'text_right', và 'label'
            # Với triplet loss, ta cần (anchor, pos, neg)
            # Cần xem model/task của MatchZoo xử lý input triplet thế nào
            # Giả định model nhận anchor, pos, neg làm input riêng
             train_data_list.append({
                 'text_anchor': parts[0],
                 'text_positive': parts[1],
                 'text_negative': parts[2],
                 'id_anchor': f'q_{i}', # ID giả
                 'id_positive': f'p_{i}',
                 'id_negative': f'n_{i}'
             })
        # else: print(f"Skipping invalid line {i+1}") # Debug

print(f"Loaded {len(train_data_list)} triplets.")
train_pack = mz.DataPack(data=pd.DataFrame(train_data_list)) # Chuyển thành DataFrame cho DataPack

# --- 3. Tiền xử lý (Sử dụng Preprocessor của model) ---
# Sử dụng preprocessor phù hợp với model Transformer
preprocessor = mz.preprocessors.BertPreprocessor(model_name=PRETRAINED_MODEL_NAME)

print("Preprocessing data...")
# Quan trọng: Đảm bảo các cột text được xử lý
# Cần điều chỉnh tên cột nếu khác 'text_anchor', 'text_positive', 'text_negative'
train_processed = preprocessor.fit_transform(train_pack, verbose=1)
# Lưu preprocessor để sử dụng sau này khi inference
preprocessor.save(os.path.join(MODEL_OUTPUT_DIR, 'preprocessor.dill'))

# --- 4. Định nghĩa Mô hình Bi-Encoder ---
# Sử dụng một kiến trúc Bi-Encoder phù hợp. Ví dụ với BERT.
# MatchZoo có thể có các lớp tiện ích hoặc bạn cần tự định nghĩa
# Ví dụ: Giả sử có một lớp `BertBiEncoderForTripletLoss`
# Cần kiểm tra tài liệu MatchZoo cho cách triển khai Bi-Encoder + Triplet Loss

# Giả định: Tạo task Ranking với TripletLoss
ranking_task = mz.tasks.Ranking(
    losses=mz.losses.TripletLoss(margin=1.0) # Margin là siêu tham số
    # metrics=['map', 'ndcg@3', 'ndcg@5'] # Metrics thường dùng cho đánh giá, không phải loss huấn luyện trực tiếp với triplet
)

# Chọn model Bi-Encoder (ví dụ: sử dụng BERT làm encoder)
# MatchZoo có thể yêu cầu bạn định nghĩa cách 2 encoder chia sẻ trọng số hoặc không
# Đây là phần phức tạp và phụ thuộc nhiều vào MatchZoo API
# Ví dụ giả định:
model = mz.models.Bert(
    task=ranking_task,
    pretrained_model_name=PRETRAINED_MODEL_NAME
    # MatchZoo có thể có cách khác để định nghĩa Bi-Encoder rõ ràng hơn
)
# Cần đảm bảo model này được thiết kế để xử lý 3 inputs (anchor, pos, neg)
# và tính toán Triplet Loss

# --- 5. Huấn luyện ---
# Tạo dataset và dataloader
train_generator = mz.dataloader.TripletGenerator(
    inputs=train_processed,
    batch_size=BATCH_SIZE,
    shuffle=True
)

# Cấu hình Trainer
trainer = mz.trainers.Trainer(
    model=model,
    optimizer=torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE),
    trainloader=train_generator,
    epochs=EPOCHS,
    # validation_loader=val_generator, # Cần tạo validation set nếu có
    # metrics=ranking_task.metrics, # Metrics để theo dõi
    save_dir=MODEL_OUTPUT_DIR,
    save_all=True, # Lưu model tốt nhất và checkpoint cuối
    verbose=1
)

print("Starting training...")
trainer.run()

print("Training finished.")
print(f"Model and preprocessor saved in: {MODEL_OUTPUT_DIR}")

# --- 6. (Tùy chọn) Lưu Model đã Huấn luyện ---
# Trainer thường tự động lưu model tốt nhất
# Bạn có thể load lại model tốt nhất nếu cần
# model.load_state_dict(torch.load(os.path.join(MODEL_OUTPUT_DIR, 'model.pt'))) # Ví dụ