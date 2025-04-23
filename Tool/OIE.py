import os
from dotenv import load_dotenv
from openie import StanfordOpenIE
import re

load_dotenv()

def extract_triples(text, properties=None, enhanced=False):
    """
    Trích xuất quan hệ (subject, relation, object) từ văn bản sử dụng Stanford OpenIE
    
    Args:
        text: Văn bản cần trích xuất quan hệ
        properties: Tùy chọn cấu hình cho OpenIE
        enhanced: Sử dụng cấu hình nâng cao cho tìm kiếm (tăng độ bao phủ)
        
    Returns:
        list: Danh sách các triple {subject, relation, object}
    """
    # Cấu hình mặc định tiêu chuẩn
    standard_properties = {
        'annotators': 'tokenize,pos,lemma,depparse,natlog,coref,openie',
        'openie.affinity_probability_cap': 0.8,
        'openie.triple.strict': False,
        'openie.max_entailments_per_clause': 3,
        'openie.resolve_coref': True,
        'timeout': 120000
    }
    
    # Cấu hình nâng cao cho tìm kiếm (tăng độ bao phủ)
    enhanced_properties = {
        'annotators': 'tokenize,pos,lemma,depparse,natlog,coref,openie',
        'outputFormat': 'json',
        'openie.affinity_probability_cap': 0.5,  # Tăng độ phủ
        'openie.triple.strict': False,           # Chấp nhận cấu trúc linh hoạt 
        'openie.max_entailments_per_clause': 5,  # Tăng số quan hệ trích xuất
        'openie.resolve_coref': True,            # Bật xử lý đồng tham chiếu
        'openie.min_relation_length': 1,         # Chấp nhận quan hệ ngắn
        'timeout': 150000                        # Tăng timeout
    }
    
    # Sử dụng cấu hình được cung cấp, hoặc chọn theo tham số enhanced
    if properties is None:
        properties = enhanced_properties if enhanced else standard_properties
    
    # Khởi tạo client
    client = StanfordOpenIE(properties=properties)
    
    try:
        with client:
            direct_props = {
                'annotators': 'tokenize,pos,lemma,depparse,natlog,coref,openie',
                'outputFormat': 'json'
            }
            result = client.client.annotate(text, properties=direct_props)
            triples = []
            for sentence in result['sentences']:
                if 'openie' in sentence:
                    for triple in sentence['openie']:
                        triples.append({
                            'subject': triple['subject'],
                            'relation': triple['relation'],
                            'object': triple['object']
                        })
            return triples
    except Exception as e:
        print(f"[ERROR] Lỗi khi trích xuất quan hệ: {e}")
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
