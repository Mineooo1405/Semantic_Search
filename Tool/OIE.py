import os
import hashlib
from dotenv import load_dotenv
from openie import StanfordOpenIE
import re
import json
import diskcache

OIE_CACHE = diskcache.Cache('./oie_cache')  # Initialize cache in a directory

corenlp_path = r'D:/SemanticSearch/CoreNLP' # Ví dụ đường dẫn
if 'CORENLP_HOME' not in os.environ and os.path.exists(corenlp_path):
    os.environ['CORENLP_HOME'] = corenlp_path
    print(f"[INFO] Đã đặt CORENLP_HOME='{corenlp_path}'")
elif 'CORENLP_HOME' not in os.environ:
    print("[WARNING] Biến môi trường CORENLP_HOME chưa được đặt và đường dẫn mặc định không tồn tại. OpenIE có thể không hoạt động.")
load_dotenv()

_CLIENT_STANDARD = None
_CLIENT_ENHANCED = None

STANDARD_PROPERTIES = {
    'annotators': 'tokenize,pos,lemma,depparse,natlog,coref,openie',
    'openie.affinity_probability_cap': 0.8,
    'openie.triple.strict': False,
    'openie.max_entailments_per_clause': 3,
    'openie.resolve_coref': True,
    'timeout': 120000 # 2 phút
}

ENHANCED_PROPERTIES = {
    'annotators': 'tokenize,pos,lemma,depparse,natlog,coref,openie',
    'outputFormat': 'json', # Đảm bảo output là JSON để parse
    'openie.affinity_probability_cap': 0.5,
    'openie.triple.strict': False,
    'openie.max_entailments_per_clause': 5,
    'openie.resolve_coref': True,
    'openie.min_relation_length': 1,
    'timeout': 150000 # 2.5 phút
}

def get_stanford_oie_client(properties, enhanced_setting: bool):
    global _CLIENT_STANDARD, _CLIENT_ENHANCED
    
    if 'CORENLP_HOME' not in os.environ:
        corenlp_path_default = r'D:/SemanticSearch/CoreNLP' # Cập nhật đường dẫn này
        if os.path.exists(corenlp_path_default):
            os.environ['CORENLP_HOME'] = corenlp_path_default
            print(f"[INFO] Đã đặt CORENLP_HOME='{corenlp_path_default}'")
        else:
            print("[ERROR] CORENLP_HOME chưa được đặt và đường dẫn mặc định không tồn tại. StanfordOpenIE sẽ không hoạt động.")
            return None

    client_to_use = _CLIENT_ENHANCED if enhanced_setting else _CLIENT_STANDARD
    props_to_use = properties if properties is not None else (ENHANCED_PROPERTIES if enhanced_setting else STANDARD_PROPERTIES)

    if client_to_use is None:
        try:
            print(f"[INFO] Khởi tạo StanfordOpenIE client (enhanced={enhanced_setting})...")
            client_to_use = StanfordOpenIE(properties=props_to_use)
            if enhanced_setting:
                _CLIENT_ENHANCED = client_to_use
            else:
                _CLIENT_STANDARD = client_to_use
            print("[INFO] StanfordOpenIE client khởi tạo thành công.")
        except Exception as e:
            print(f"[ERROR] Lỗi khởi tạo StanfordOpenIE client (enhanced={enhanced_setting}): {e}")
            return None
    return client_to_use

def extract_triples(text, properties=None, enhanced=False):
    """
    Trích xuất quan hệ (subject, relation, object) từ văn bản sử dụng Stanford OpenIE
    và sử dụng caching.
    """
    # --- Start Caching Logic ---
    cache_key_parts = [text]
    if properties:
        cache_key_parts.append(json.dumps(properties, sort_keys=True)) # Đảm bảo thứ tự key nhất quán
    cache_key_parts.append(str(enhanced))
    cache_key = hashlib.md5("||".join(cache_key_parts).encode('utf-8')).hexdigest()

    cached_result = OIE_CACHE.get(cache_key)
    if cached_result is not None:
        # print(f"[DEBUG] OIE Cache hit for key: {cache_key}")
        return cached_result
    # --- End Caching Logic ---

    client = get_stanford_oie_client(properties, enhanced)
    if not client:
        return []
    
    current_properties = ENHANCED_PROPERTIES if enhanced else STANDARD_PROPERTIES
    if properties: 
         current_properties = properties

    triples = [] # Di chuyển khởi tạo ra ngoài try-except để return nếu có lỗi
    try:
        final_props_for_annotate = current_properties.copy()
        final_props_for_annotate['outputFormat'] = 'json'

        annotated_text = client.client.annotate(text, properties=final_props_for_annotate)
        
        if isinstance(annotated_text, str):
            try:
                result_json = json.loads(annotated_text)
            except json.JSONDecodeError as je:
                print(f"[ERROR] Lỗi parse JSON từ OpenIE: {je}")
                print(f"[DEBUG] Raw output: {annotated_text[:500]}")
                OIE_CACHE.set(cache_key, []) # Cache kết quả rỗng nếu lỗi parse
                return []
        elif isinstance(annotated_text, dict):
             result_json = annotated_text
        else:
            print(f"[ERROR] Định dạng kết quả không mong muốn từ OpenIE: {type(annotated_text)}")
            OIE_CACHE.set(cache_key, []) # Cache kết quả rỗng
            return []

        if 'sentences' in result_json:
            for sentence_data in result_json['sentences']:
                if 'openie' in sentence_data:
                    for triple_data in sentence_data['openie']: # Đổi tên biến để tránh nhầm lẫn
                        triples.append({
                            'subject': triple_data['subject'],
                            'relation': triple_data['relation'],
                            'object': triple_data['object']
                        })
        else:
            print("[WARNING] Không tìm thấy key 'sentences' trong kết quả OpenIE.")
        
        OIE_CACHE.set(cache_key, triples) # Lưu kết quả thành công vào cache
        return triples
    except Exception as e:
        print(f"[ERROR] Lỗi khi trích xuất quan hệ: {e}")
        import traceback
        traceback.print_exc()
        OIE_CACHE.set(cache_key, []) # Cache kết quả rỗng nếu có lỗi nghiêm trọng
        return []

def extract_triples_for_search(text, properties=None):
    """
    Trích xuất quan hệ với cấu hình tối ưu cho tìm kiếm (độ bao phủ cao)
    
    Args:
        text: Văn bản cần trích xuất quan hệ
        properties: Tùy chọn ghi đè cấu hình mặc định
        
    Returns:
        list: Danh sách các triple {subject, relation, object}
    """
    # Sử dụng hàm extract_triples với tham số enhanced=True
    return extract_triples(text, properties, enhanced=True)

def extract_triples_with_context(text, context=None, properties=None):
    """
    Trích xuất quan hệ với ngữ cảnh nhóm để tăng độ chính xác
    
    Args:
        text: Câu cần trích xuất quan hệ
        context: Ngữ cảnh nhóm (các câu liên quan)
        properties: Tùy chọn cấu hình
        
    Returns:
        list: Danh sách các triple đã được lọc và cải thiện
    """
    # Nếu không có ngữ cảnh, trích xuất trực tiếp
    if context is None or text == context:
        return extract_triples_for_search(text, properties)
    
    # Xử lý trích xuất với ngữ cảnh
    # Cách 1: Trích xuất từ câu gốc trước
    direct_triples = extract_triples_for_search(text, properties)
    
    # Cách 2: Nếu không đủ triples, thử với ngữ cảnh
    if len(direct_triples) < 2:
        # Tạo văn bản ngữ cảnh ngắn gọn
        context_text = context[:1000]  # Giới hạn kích thước ngữ cảnh
        context_triples = extract_triples_for_search(context_text, properties)
        
        # Lọc các triples liên quan đến câu gốc
        relevant_triples = []
        text_keywords = extract_keywords(text)
        
        for triple in context_triples:
            # Kiểm tra xem triple có liên quan đến từ khóa trong văn bản không
            if any(keyword in triple['subject'].lower() for keyword in text_keywords) or \
               any(keyword in triple['object'].lower() for keyword in text_keywords):
                relevant_triples.append(triple)
        
        # Kết hợp triples trực tiếp và triples từ ngữ cảnh
        combined_triples = direct_triples + relevant_triples
        # Loại bỏ trùng lặp
        return filter_duplicate_triples(combined_triples)
    
    return direct_triples

def filter_low_quality_triples(triples, min_subject_len=1, min_relation_len=2, min_object_len=1):
    """
    Lọc bỏ các triples chất lượng thấp
    
    Args:
        triples: Danh sách các triple cần lọc
        min_subject_len: Độ dài tối thiểu của subject (số từ)
        min_relation_len: Độ dài tối thiểu của relation (số từ)
        min_object_len: Độ dài tối thiểu của object (số từ)
        
    Returns:
        list: Danh sách triple đã được lọc
    """
    filtered = []
    
    for triple in triples:
        # Kiểm tra độ dài các thành phần
        subject_words = len(triple['subject'].split())
        relation_words = len(triple['relation'].split())
        object_words = len(triple['object'].split())
        
        if subject_words < min_subject_len or relation_words < min_relation_len or object_words < min_object_len:
            continue
            
        # Kiểm tra lỗi cú pháp phổ biến
        if any(bad in triple['relation'].lower() for bad in ['can can', 'are are', 'of are', 'is is']):
            continue
            
        # Kiểm tra lỗi dấu câu
        if triple['subject'].count('(') != triple['subject'].count(')') or \
           triple['relation'].count('(') != triple['relation'].count(')') or \
           triple['object'].count('(') != triple['object'].count(')'):
            continue
        
        # Thêm triple hợp lệ
        filtered.append(triple)
    
    return filtered

def filter_duplicate_triples(triples):
    """
    Loại bỏ các triples trùng lặp hoặc gần giống nhau
    
    Args:
        triples: Danh sách các triple cần lọc
        
    Returns:
        list: Danh sách triple không trùng lặp
    """
    if not triples:
        return []
        
    # Tạo một bản sao để không làm thay đổi danh sách gốc
    unique_triples = []
    seen = set()
    
    for triple in triples:
        # Chuẩn hóa để so sánh: loại bỏ khoảng trắng thừa, chuyển về chữ thường
        normalized_triple = (
            re.sub(r'\s+', ' ', triple['subject'].lower().strip()),
            re.sub(r'\s+', ' ', triple['relation'].lower().strip()),
            re.sub(r'\s+', ' ', triple['object'].lower().strip())
        )
        
        # Kiểm tra trùng lặp
        if normalized_triple not in seen:
            seen.add(normalized_triple)
            unique_triples.append(triple)
    
    return unique_triples

def extract_keywords(text, min_word_len=4, max_keywords=10):
    """
    Trích xuất các từ khóa từ văn bản
    
    Args:
        text: Văn bản cần trích xuất từ khóa
        min_word_len: Độ dài tối thiểu của từ khóa
        max_keywords: Số từ khóa tối đa
        
    Returns:
        list: Danh sách các từ khóa
    """
    # Chuyển về chữ thường
    text = text.lower()
    
    # Loại bỏ dấu câu và số
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\d+', ' ', text)
    
    # Tách từ
    words = text.split()
    
    # Loại bỏ từ ngắn và từ stop word
    stop_words = {'a', 'an', 'the', 'and', 'or', 'but', 'is', 'are', 'was', 'were', 
                 'in', 'on', 'at', 'to', 'for', 'with', 'by', 'about', 'like', 
                 'from', 'of', 'as', 'that', 'this', 'these', 'those', 'it', 'its'}
    
    content_words = [word for word in words if len(word) >= min_word_len and word not in stop_words]
    
    # Tính tần suất
    freq = {}
    for word in content_words:
        freq[word] = freq.get(word, 0) + 1
    
    # Sắp xếp theo tần suất giảm dần
    sorted_words = sorted(freq.items(), key=lambda x: x[1], reverse=True)
    
    # Trả về danh sách từ khóa phổ biến nhất
    return [word for word, count in sorted_words[:max_keywords]]

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
    
    import time
    # Hiển thị tiến độ
    start_time = time.time()
    
    # Xử lý theo nhóm để có context tốt hơn
    for group_idx, group in enumerate(groups):
        print(f"  Đang trích xuất quan hệ cho nhóm {group_idx+1}/{len(groups)}...")
        
        # Tạo ngữ cảnh nhóm
        group_context = " ".join([sentences[idx] for idx in group])
        
        # Trích xuất triples cho từng câu trong nhóm
        for i, sentence_idx in enumerate(group):
            sentence = sentences[sentence_idx]
            
            # Trích xuất triples từ câu với ngữ cảnh nhóm
            triples = extract_triples_with_context(sentence, group_context)
            
            # Lọc bỏ triples chất lượng thấp
            filtered_triples = filter_low_quality_triples(triples)
            
            # Lưu trữ triples
            sentence_triples[sentence_idx] = filtered_triples
            all_triples.extend(filtered_triples)
    
    # Loại bỏ trùng lặp ở mức toàn cục
    all_triples = filter_duplicate_triples(all_triples)
    
    elapsed_time = time.time() - start_time
    print(f"\n[SUCCESS] Đã trích xuất {len(all_triples)} quan hệ trong {elapsed_time:.2f}s")
    
    return all_triples, sentence_triples

def generate_graph(text, output_path="graph.png", properties=None):
    """
    Tạo biểu đồ trực quan từ các triples trong văn bản
    
    Args:
        text: Văn bản cần trực quan hóa
        output_path: Đường dẫn file ảnh
        properties: Tùy chọn cấu hình
        
    Returns:
        bool: True nếu tạo biểu đồ thành công, False nếu thất bại
    """
    # Trích xuất triples trước
    triples = extract_triples(text, properties)
    
    # Nếu không có triples, không thể tạo biểu đồ
    if not triples:
        print("[ERROR] Không thể tạo biểu đồ do không tìm thấy quan hệ")
        return False
    
    try:
        client = StanfordOpenIE(properties=properties or {})
        with client:
            # Phương pháp generate_graphviz_graph sử dụng client và triples
            client.generate_graphviz_graph(text, output_path)
            print(f"[INFO] Đã tạo biểu đồ tại: {output_path}")
            return True
    except Exception as e:
        print(f"[ERROR] Lỗi khi tạo biểu đồ: {e}")
        return False

def close_oie_clients():
    """Đóng các client OpenIE nếu chúng đã được mở."""
    global _CLIENT_STANDARD, _CLIENT_ENHANCED
    if _CLIENT_STANDARD:
        try:
            _CLIENT_STANDARD.stop() # Hoặc .close(), tùy thuộc vào API của thư viện
            print("[INFO] StanfordOpenIE standard client stopped.")
        except Exception as e:
            print(f"[ERROR] Lỗi khi đóng standard client: {e}")
        _CLIENT_STANDARD = None
    if _CLIENT_ENHANCED:
        try:
            _CLIENT_ENHANCED.stop() # Hoặc .close()
            print("[INFO] StanfordOpenIE enhanced client stopped.")
        except Exception as e:
            print(f"[ERROR] Lỗi khi đóng enhanced client: {e}")
        _CLIENT_ENHANCED = None

# Nên gọi close_oie_clients() khi ứng dụng của bạn kết thúc
# Ví dụ: import atexit; atexit.register(close_oie_clients)