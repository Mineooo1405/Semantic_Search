from nltk.tokenize import sent_tokenize
import re
import nltk

# Cách import và load spaCy đúng
import spacy

# Tải mô hình transformer
try:
    print("[INFO] Đang tải mô hình transformer (có thể mất vài phút đầu tiên)...")
    nlp = spacy.load("en_core_web_trf")
    print("[INFO] Đã tải mô hình transformer thành công")
    USING_TRANSFORMER = True
except OSError:
    print("[WARNING] Không tìm thấy mô hình 'en_core_web_trf'")
    print("[INFO] Tải mô hình cơ bản...")
    try:
        nlp = spacy.load("en_core_web_sm")
        print("[INFO] Đã tải mô hình cơ bản thành công")
        USING_TRANSFORMER = False
    except OSError:
        print("[ERROR] Không tìm thấy mô hình spaCy nào")
        print("[INFO] Vui lòng cài đặt: python -m spacy download en_core_web_trf")
        nlp = None
        USING_TRANSFORMER = False

def sentence_detector(text):
    """
    Phương pháp cơ bản để tách văn bản thành các câu
    
    Args:
        text (str): Văn bản cần tách
        
    Returns:
        list: Danh sách các câu
    """
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
 
    return sent_tokenize(text)

def split_by_delimiter(sentence):
    """
    Tách câu theo dấu câu ngắt mềm (phẩy, chấm phẩy, hai chấm)
    
    Args:
        sentence (str): Câu cần tách
        
    Returns:
        list: Danh sách các cụm sau khi tách
    """
    # Tách theo dấu phẩy, chấm phẩy, dấu hai chấm nhưng giữ ngữ cảnh
    # Không tách nếu dưới 5 từ tránh tạo cụm vô nghĩa
    if len(sentence.split()) < 5:
        return [sentence]
    
    # Tách câu có các cụm rõ ràng (chấm phẩy, hai chấm)
    if ';' in sentence or ':' in sentence:
        subsentences = re.split(r'(?<=[;:])\s+', sentence)
        result = []
        for sub in subsentences:
            if len(sub.split()) >= 3:  # Chỉ giữ cụm có ít nhất 3 từ
                result.append(sub.strip())
            else:
                # Nếu quá ngắn, ghép vào cụm trước đó
                if result:
                    result[-1] = result[-1] + " " + sub.strip()
                else:
                    result.append(sub.strip())
        return result
    
    # Tách câu dài theo dấu phẩy
    if len(sentence.split()) > 12 and ',' in sentence:
        subsentences = re.split(r'(?<=,)\s+', sentence)
        result = []
        
        for i, sub in enumerate(subsentences):
            if len(sub.split()) >= 4:  # Cụm tương đối đầy đủ
                result.append(sub.strip())
            else:
                # Ghép cụm ngắn với cụm liền trước hoặc sau
                if result:
                    result[-1] = result[-1] + " " + sub.strip()
                else:
                    # Nếu là cụm đầu tiên quá ngắn, ghép với cụm tiếp theo
                    if i+1 < len(subsentences):
                        result.append(sub.strip() + " " + subsentences[i+1].strip())
                        subsentences[i+1] = ""  # Đánh dấu đã xử lý
                    else:
                        result.append(sub.strip())
        
        # Lọc bỏ cụm rỗng
        result = [r for r in result if r]
        return result
    
    # Nếu không thỏa điều kiện nào, trả về nguyên câu
    return [sentence]

def split_by_conjunction(sentence):
    """
    Tách câu theo liên từ phổ biến
    
    Args:
        sentence (str): Câu cần tách
        
    Returns:
        list: Danh sách các cụm sau khi tách
    """
    # Chỉ tách câu dài và có liên từ
    if len(sentence.split()) < 10:
        return [sentence]
    
    # Các mẫu có liên từ phổ biến kèm dấu phẩy
    patterns = [
        r',\s+(and|but|or)\s+',  # ", and ", ", but ", ", or "
        r'\s+(because|although|though|while|whereas|since)\s+',  # "because", "although"...
        r',\s+(which|who|whom|where|when)\s+'  # ", which ", ", who "...
    ]
    
    # Tìm tất cả vị trí tách tiềm năng
    split_positions = []
    for pattern in patterns:
        for match in re.finditer(pattern, sentence):
            split_positions.append((match.start(), match.end(), match.group()))
    
    # Nếu không tìm thấy điểm tách
    if not split_positions:
        return [sentence]
    
    # Sắp xếp theo vị trí xuất hiện
    split_positions.sort()
    
    # Tách câu thành các phần
    result = []
    last_end = 0
    
    for start, end, connector in split_positions:
        # Từ vị trí cuối cùng đến liên từ hiện tại 
        part = sentence[last_end:start].strip()
        
        # Chỉ thêm phần có ý nghĩa (ít nhất 3 từ)
        if len(part.split()) >= 3:
            result.append(part)
            
            # Cập nhật vị trí cuối
            last_end = start + 1  # Giữ lại dấu phẩy nếu có
    
    # Thêm phần cuối cùng
    final_part = sentence[last_end:].strip()
    if len(final_part.split()) >= 3:
        result.append(final_part)
    elif result:  # Nếu phần cuối quá ngắn, ghép vào phần trước
        result[-1] = result[-1] + " " + final_part
    
    # Nếu kết quả trống, trả về câu gốc
    if not result:
        return [sentence]
        
    return result

def process_parenthetical(sentence):
    """
    Xử lý các cụm trong ngoặc đơn và ngoặc ngang
    
    Args:
        sentence (str): Câu cần xử lý
        
    Returns:
        list: Danh sách các cụm chính và cụm trong ngoặc
    """
    # Tìm các cụm trong ngoặc
    paren_pattern = r'\(([^)]+)\)'      # (text inside parentheses)
    dash_pattern = r'\s-\s([^-]+)\s-\s'  # - text between dashes -
    
    parentheticals = []
    clean_sentence = sentence
    
    # Xử lý ngoặc đơn
    for match in re.finditer(paren_pattern, sentence):
        content = match.group(1)
        if len(content.split()) >= 3:  # Chỉ tách nếu có ít nhất 3 từ
            parentheticals.append(content)
            # Xóa nội dung ngoặc khỏi câu chính
            clean_sentence = clean_sentence.replace(match.group(0), "")
    
    # Xử lý ngoặc ngang
    for match in re.finditer(dash_pattern, sentence):
        content = match.group(1)
        if len(content.split()) >= 3:  # Chỉ tách nếu có ít nhất 3 từ
            parentheticals.append(content)
            # Xóa nội dung ngoặc khỏi câu chính
            clean_sentence = clean_sentence.replace(match.group(0), " ")
    
    # Trả về câu đã được làm sạch và các cụm trong ngoặc
    result = [clean_sentence.strip()]
    result.extend(parentheticals)
    
    return [s for s in result if s and len(s.split()) >= 3]

def enhanced_sentence_detector(text, aggressive=True):
    """
    Phương pháp nâng cao để tách văn bản thành câu và tiếp tục chia nhỏ các câu phức tạp
    
    Args:
        text (str): Văn bản cần tách
        aggressive (bool): Mức độ chia nhỏ tích cực
        
    Returns:
        list: Danh sách các câu và cụm câu
    """
    # Tách thành câu cơ bản
    sentences = sentence_detector(text)
    
    # Đối với mỗi câu, tiếp tục chia nhỏ hơn
    refined_sentences = []
    
    for sentence in sentences:
        # Chỉ xử lý câu đủ dài
        if len(sentence.split()) < 5:
            refined_sentences.append(sentence)
            continue
        
        # 1. Xử lý các cụm trong ngoặc
        parenthetical_parts = process_parenthetical(sentence)
        
        # Với mỗi phần sau khi xử lý ngoặc
        for part in parenthetical_parts:
            if len(part.split()) < 5:
                refined_sentences.append(part)
                continue
                
            # 2. Tách theo dấu câu
            delimiter_parts = split_by_delimiter(part)
            
            # 3. Nếu cần chia tích cực hơn, tách theo liên từ
            if aggressive:
                conjunction_parts = []
                for d_part in delimiter_parts:
                    # Chỉ chia tiếp các cụm dài
                    if len(d_part.split()) > 10:
                        conj_subparts = split_by_conjunction(d_part)
                        conjunction_parts.extend(conj_subparts)
                    else:
                        conjunction_parts.append(d_part)
                refined_sentences.extend(conjunction_parts)
            else:
                refined_sentences.extend(delimiter_parts)
    
    # Loại bỏ trùng lặp và chuẩn hóa
    result = []
    for s in refined_sentences:
        s = s.strip()
        if s and s not in result:
            result.append(s)
    
    return result

def sentence_splitter_for_oie(text):
    """
    Pipeline hoàn chỉnh để tách văn bản tối ưu cho OpenIE
    
    Args:
        text (str): Văn bản cần xử lý
        
    Returns:
        tuple: (câu gốc, cụm câu đã chia nhỏ)
    """
    # Tách câu cơ bản
    original_sentences = sentence_detector(text)
    
    # Chia nhỏ câu tích cực để phục vụ OIE
    sub_sentences = enhanced_sentence_detector(text, aggressive=True)
    
    return original_sentences, sub_sentences

# Demo chức năng
if __name__ == "__main__":
    example = "Pascal Hardy, an engineer and sustainable development consultant, began experimenting with vertical farming and aeroponic growing towers - as the soil-free plastic tubes are known - on his Paris apartment block roof five years ago."
    
    print("=== TÁCH CÂU CƠ BẢN ===")
    basic_sentences = sentence_detector(example)
    for i, s in enumerate(basic_sentences):
        print(f"{i+1}. {s}")
    
    print("\n=== TÁCH CÂU NÂNG CAO (TỐI ƯU CHO OIE) ===")
    enhanced_sentences = enhanced_sentence_detector(example)
    for i, s in enumerate(enhanced_sentences):
        print(f"{i+1}. {s}")
        
def spacy_sentence_splitter(text, split_clauses=True):
    """
    Tách câu sử dụng spaCy với mô hình transformer
    
    Args:
        text: Văn bản cần tách
        split_clauses: Có chia nhỏ mệnh đề hay không
    
    Returns:
        tuple: (original_sentences, sub_sentences)
    """
    # Nếu không có spaCy, dùng phương pháp dự phòng
    if nlp is None:
        original_sentences = sentence_detector(text)
        enhanced = enhanced_sentence_detector(text, aggressive=split_clauses)
        return original_sentences, enhanced
    
    # Xử lý văn bản với spaCy
    doc = nlp(text)
    
    # Danh sách câu gốc
    original_sentences = [sent.text for sent in doc.sents]
    
    # Nếu không cần chia nhỏ mệnh đề, trả về câu gốc
    if not split_clauses:
        return original_sentences, original_sentences
    
    # Danh sách kết quả sau khi chia nhỏ
    sub_sentences = []
    
    # Thêm câu gốc trước
    sub_sentences.extend(original_sentences)
    
    # Cho mỗi câu
    for sent in doc.sents:
        if len(sent) <= 10:  # Bỏ qua câu ngắn
            continue
            
        # 1. Tách mệnh đề chính/phụ dựa trên phân tích phụ thuộc
        clause_heads = [token for token in sent 
                       if token.dep_ in ('ROOT', 'ccomp', 'xcomp', 'advcl', 'acl', 'relcl') 
                       and token.pos_ == 'VERB']
        
        for head in clause_heads:
            # Bỏ qua nếu là động từ trong câu gốc
            if head.dep_ == 'ROOT':
                continue
                
            # Xây dựng mệnh đề từ gốc
            clause_tokens = []
            
            # Tìm chủ ngữ của mệnh đề
            subjects = [child for child in head.children 
                       if child.dep_ in ('nsubj', 'nsubjpass')]
            
            # Thêm chủ ngữ vào mệnh đề
            for subj in subjects:
                clause_tokens.append(subj)
                # Thêm các từ phụ thuộc vào chủ ngữ
                clause_tokens.extend(list(subj.subtree))
            
            # Thêm động từ
            clause_tokens.append(head)
            
            # Thêm các từ phụ thuộc vào động từ (trừ liên từ và dấu)
            for child in head.children:
                if child.dep_ not in ('punct', 'cc', 'mark') and child not in subjects:
                    clause_tokens.extend(list(child.subtree))
            
            # Sắp xếp token theo vị trí trong câu
            clause_tokens = sorted(set(clause_tokens), key=lambda x: x.i)
            
            # Tạo chuỗi từ tokens
            clause_text = ' '.join(t.text for t in clause_tokens).strip()
            
            # Chỉ thêm mệnh đề có ít nhất 3 từ và không trùng lặp
            min_words = 3
            if len(clause_text.split()) >= min_words and clause_text not in sub_sentences:
                sub_sentences.append(clause_text)
        
        # 2. Xử lý cụm trong ngoặc
        for token in sent:
            if token.is_bracket or token.text in ['-', '(', '[', '{']:
                # Tìm dấu đóng ngoặc tương ứng
                stack = [token]
                content_tokens = []
                
                for t in sent[token.i+1:]:
                    if (token.text == '(' and t.text == ')') or \
                       (token.text == '[' and t.text == ']') or \
                       (token.text == '{' and t.text == '}') or \
                       (token.text == '-' and t.text == '-'):
                        # Tạo chuỗi từ nội dung trong ngoặc
                        if content_tokens:
                            content_text = ' '.join(t.text for t in content_tokens).strip()
                            if len(content_text.split()) >= min_words and content_text not in sub_sentences:
                                sub_sentences.append(content_text)
                        break
                    else:
                        content_tokens.append(t)
    
    # Sắp xếp theo độ dài giảm dần để đảm bảo câu dài trước
    sub_sentences = sorted(set(sub_sentences), key=len, reverse=True)
    
    return original_sentences, sub_sentences

# Hàm main để kiểm tra khi chạy trực tiếp
if __name__ == "__main__":
    test_text = "Pascal Hardy, an engineer and sustainable development consultant, began experimenting with vertical farming and aeroponic growing towers - as the soil-free plastic tubes are known - on his Paris apartment block roof five years ago."
    
    print("\n=== Test with spaCy" + (" Transformer" if USING_TRANSFORMER else " Basic") + " Model ===")
    original, sub_sentences = spacy_sentence_splitter(test_text)
    
    print("\nOriginal sentences:")
    for i, s in enumerate(original):
        print(f"{i+1}. {s}")
    
    print("\nSub-sentences:")
    for i, s in enumerate(sub_sentences):
        print(f"{i+1}. {s}")