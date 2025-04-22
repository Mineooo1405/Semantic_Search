import os
from dotenv import load_dotenv
from openie import StanfordOpenIE

load_dotenv()

def extract_triples(text, properties=None):

    if properties is None:
        properties = {
            'annotators': 'tokenize,pos,lemma,depparse,natlog,coref,openie',
            'openie.affinity_probability_cap': 0.8,
            'openie.triple.strict': False,
            'openie.max_entailments_per_clause': 3,
            'openie.resolve_coref': True,
            'timeout': 120000
        }
    
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

    if properties is None:
        properties = {
            'annotators': 'tokenize,pos,lemma,depparse,natlog,coref,openie',
            'outputFormat': 'json',
            # Tăng độ phủ để bắt được nhiều quan hệ có ích
            'openie.affinity_probability_cap': 0.5,  
            # Chấp nhận cả triples có cấu trúc không quá nghiêm ngặt
            'openie.triple.strict': False,  
            # Trích xuất 2-3 quan hệ từ mỗi mệnh đề để có cả quan hệ đơn giản và phức tạp
            'openie.max_entailments_per_clause': 5,  
            # Bật xử lý đồng tham chiếu để liên kết các thực thể trong ngữ cảnh
            'openie.resolve_coref': True,  
            # Giảm độ dài tối thiểu của quan hệ để bắt được cả quan hệ ngắn
            'openie.min_relation_length': 1,
            # Tăng timeout để xử lý văn bản dài
            'timeout': 150000
        }
    
    # Gọi hàm extract_triples với cấu hình tối ưu
    return extract_triples(text, properties)

def generate_graph(text, output_path="graph.png", properties=None):
    """
    Tạo biểu đồ trực quan từ các triples trong văn bản
    
    Args:
        text: Văn bản cần trực quan hóa
        output_path: Đường dẫn file ảnh
        properties: Tùy chọn cấu hình
    """
    # Cấu hình mặc định
    if properties is None:
        properties = {
            'annotators': 'tokenize,pos,lemma,depparse,natlog,coref,openie',
            'openie.affinity_probability_cap': 0.8,
            'openie.triple.strict': False,
            'openie.max_entailments_per_clause': 3,
            'openie.resolve_coref': True,
            'timeout': 120000
        }
    
    # Trích xuất triples trước
    triples = extract_triples(text, properties)
    
    # Nếu không có triples, không thể tạo biểu đồ
    if not triples:
        print("[ERROR] Không thể tạo biểu đồ do không tìm thấy quan hệ")
        return False
    
    try:
        client = StanfordOpenIE(properties=properties)
        with client:
            # Phương pháp generate_graphviz_graph sử dụng client và triples
            client.generate_graphviz_graph(text, output_path)
            print(f"[INFO] Đã tạo biểu đồ tại: {output_path}")
            return True
    except Exception as e:
        print(f"[ERROR] Lỗi khi tạo biểu đồ: {e}")
        return False
