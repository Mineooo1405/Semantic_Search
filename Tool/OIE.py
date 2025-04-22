import os
from dotenv import load_dotenv
from openie import StanfordOpenIE

# Tải biến môi trường
load_dotenv()

def extract_triples(text, properties=None):
    """
    Trích xuất quan hệ (subject, relation, object) từ văn bản
    
    Args:
        text: Văn bản cần trích xuất
        properties: Tùy chọn cấu hình OpenIE
        
    Returns:
        List các dict với 'subject', 'relation', 'object'
    """
    # Cấu hình mặc định tối ưu
    if properties is None:
        properties = {
            'openie.affinity_probability_cap': 0.8,
            'openie.triple.strict': False,
            'openie.max_entailments_per_clause': 3,
            'openie.resolve_coref': True,
            'timeout': 120000
        }
    
    try:
        with StanfordOpenIE(properties=properties) as client:
            triples = client.annotate(text)
        return triples
    except Exception as e:
        print(f"[ERROR] Lỗi khi trích xuất: {e}")
        return []

def generate_graph(text, output_path="graph.png"):
    """
    Tạo biểu đồ trực quan từ các triples trong văn bản
    
    Args:
        text: Văn bản cần trực quan hóa
        output_path: Đường dẫn file ảnh
    """
    with StanfordOpenIE() as client:
        client.generate_graphviz_graph(text, output_path)
        print(f"[INFO] Đã tạo biểu đồ tại: {output_path}")