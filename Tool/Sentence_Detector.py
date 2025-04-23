from nltk.tokenize import sent_tokenize
import re
import nltk

# Cách import và load spaCy đúng
import spacy

# Tải mô hình transformer
try:
    print("[INFO] Đang tải mô hình transformer")
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
    
    # THÊM: Kiểm tra xem dấu phẩy có phân tách chủ ngữ-vị ngữ không
    if ',' in sentence and len(sentence.split(',')) == 2:
        parts = sentence.split(',')
        # Kiểm tra phần đầu có phải là chủ ngữ (không chứa động từ chính)
        # Và phần sau có phải là vị ngữ (bắt đầu với động từ)
        first_part = parts[0].strip()
        second_part = parts[1].strip()
        
        common_verbs = ['is', 'are', 'was', 'were', 'be', 'being', 'been',
                      'have', 'has', 'had', 'do', 'does', 'did',
                      'can', 'could', 'may', 'might', 'shall', 'should',
                      'will', 'would', 'must', 'begin', 'began', 'begun']
        
        # Nếu phần đầu không chứa động từ phổ biến và phần sau bắt đầu bằng động từ
        has_verb_in_first = any(f" {v} " in f" {first_part} " for v in common_verbs)
        starts_with_verb = any(second_part.lower().startswith(v) for v in common_verbs)
        
        if not has_verb_in_first and starts_with_verb:
            # Đây có thể là cặp chủ ngữ-vị ngữ, giữ nguyên
            return [sentence]
    
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
    
    # Cập nhật mẫu regex để phát hiện chính xác cụm trong dấu gạch ngang
    dash_pattern = r'\s+[-—–]\s+([^-—–]+?)\s+[-—–]\s+'
    
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
            # Xóa nội dung ngoặc khỏi câu chính, giữ lại khoảng trắng
            clean_sentence = clean_sentence.replace(match.group(0), " ")
    
    # Trả về câu đã được làm sạch và các cụm trong ngoặc
    result = [clean_sentence.strip()]
    result.extend(parentheticals)
    
    return [s for s in result if s and len(s.split()) >= 3]

def enhanced_sentence_detector(text, aggressive=True):
    # Bảo vệ từ ghép có gạch nối
    protected_text, replacements = protect_hyphenated_words(text)
    
    # Tách câu cơ bản trước
    sentences = sentence_detector(protected_text)
    
    # Đối với mỗi câu, tiếp tục chia nhỏ hơn
    refined_sentences = []
    
    for sentence in sentences:
        
        # 1. Xử lý các cụm trong ngoặc
        parenthetical_parts = process_parenthetical(sentence)
        
        # Với mỗi phần sau khi xử lý ngoặc
        for part in parenthetical_parts:
    
            # 2. Tách theo dấu câu
            delimiter_parts = split_by_delimiter(part)
            
            # 3. Nếu cần chia tích cực hơn, tách theo liên từ
            if aggressive:
                conjunction_parts = []
                for d_part in delimiter_parts:
                    if len(d_part.split()) > 5: 
                        conj_subparts = split_by_conjunction(d_part)
                        conjunction_parts.extend(conj_subparts)
                    else:
                        conjunction_parts.append(d_part)
                refined_sentences.extend(conjunction_parts)
            else:
                refined_sentences.extend(delimiter_parts)

    result = []
    for s in refined_sentences:
        s = s.strip()
        if s and s not in result:
            result.append(s)
    
    # Khôi phục từ ghép trong kết quả
    result = [restore_hyphenated_words(s, replacements) for s in result]
    return result

def sentence_splitter_for_oie(text):
    
    # Tách câu cơ bản
    original_sentences = sentence_detector(text)
    
    # Chia nhỏ câu tích cực để phục vụ OIE
    sub_sentences = enhanced_sentence_detector(text, aggressive=True)
    
    return original_sentences, sub_sentences
        
def spacy_sentence_splitter(text, split_clauses=True):

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
        # Bỏ điều kiện kiểm tra độ dài câu để bao gồm cả câu ngắn
            
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
            
            # Giảm giới hạn từ tối thiểu từ 3 xuống 2
            min_words = 2  # Giảm từ 3 xuống 2
            if len(clause_text.split()) >= min_words and clause_text not in sub_sentences:
                sub_sentences.append(clause_text)
        
        # 2. Xử lý cụm trong ngoặc và dấu gạch ngang
        for sent in doc.sents:
            # Phát hiện cụm có dấu gạch ngang theo cách mới
            dash_spans = []
            
            # Tìm các đoạn văn nằm giữa cặp dấu gạch ngang (có khoảng trắng xung quanh)
            text = sent.text
            
            # Điều chỉnh regex cho dấu gạch ngang để phân biệt với từ ghép
            # Sử dụng negative lookbehind và lookahead để tránh khớp với từ ghép có gạch nối
            dash_pattern = r'(?<!\S)[-—–]\s+(.+?)\s+[-—–](?!\S)'
            
            for match in re.finditer(dash_pattern, text):
                content = match.group(1)
                # Kiểm tra thêm để đảm bảo không phải là phần của từ ghép
                if ('-' not in content or ' ' in content) and len(content.split()) >= 2:
                    if content not in sub_sentences:
                        sub_sentences.append(content)
            
            # Xử lý các loại ngoặc (giữ nguyên phần này)
            for token in sent:
                if token.is_bracket or token.text in ['(', '[', '{']:
                    # Tìm dấu đóng ngoặc tương ứng
                    stack = [token]
                    content_tokens = []
                    
                    for t in sent[token.i+1:]:
                        if (token.text == '(' and t.text == ')') or \
                           (token.text == '[' and t.text == ']') or \
                           (token.text == '{' and t.text == '}'):
                            # Tạo chuỗi từ nội dung trong ngoặc
                            if content_tokens:
                                content_text = ' '.join(t.text for t in content_tokens).strip()
                                if len(content_text.split()) >= min_words and content_text not in sub_sentences:
                                    sub_sentences.append(content_text)
                            break
                        else:
                            content_tokens.append(t)
    
    # Lọc trùng lặp nhưng giữ nguyên thứ tự xuất hiện
    seen = set()
    filtered_sentences = []
    for s in sub_sentences:
        if s not in seen:
            seen.add(s)
            filtered_sentences.append(s)
    
    sub_sentences = filtered_sentences
    return original_sentences, sub_sentences

def protect_hyphenated_words(text):
    """
    Bảo vệ từ ghép có gạch nối bằng cách tạm thời thay thế gạch nối bằng ký tự đặc biệt
    
    Args:
        text: Văn bản cần xử lý
        
    Returns:
        tuple: (processed_text, replacements)
    """
    # Mẫu regex cho từ ghép có gạch nối: từ-từ
    hyphen_pattern = r'(\w+)-(\w+)'
    
    # Danh sách các từ cần khôi phục
    replacements = {}
    
    # Hàm để thay thế từng từ ghép
    def replace_match(match):
        full_word = match.group(0)
        placeholder = f"__HYPHEN_{len(replacements)}__"
        replacements[placeholder] = full_word
        return placeholder
    
    # Thay thế tất cả từ ghép có gạch nối
    processed_text = re.sub(hyphen_pattern, replace_match, text)
    
    return processed_text, replacements

def restore_hyphenated_words(text, replacements):
    """
    Khôi phục từ ghép có gạch nối từ ký tự đặc biệt
    
    Args:
        text: Văn bản đã được xử lý
        replacements: Dict ánh xạ từ placeholder đến từ gốc
        
    Returns:
        str: Văn bản đã được khôi phục
    """
    restored_text = text
    for placeholder, original in replacements.items():
        restored_text = restored_text.replace(placeholder, original)
    
    return restored_text

def filter_redundant_sentences(sentences):
    """Lọc bỏ các câu dư thừa (là tập con của câu khác)"""
    filtered = []
    
    # Sắp xếp câu theo độ dài giảm dần để ưu tiên câu dài hơn
    sorted_sentences = sorted(sentences, key=len, reverse=True)
    
    for i, sentence1 in enumerate(sorted_sentences):
        is_subset = False
        # Loại bỏ dấu câu khi so sánh để tìm tập con chính xác hơn 
        clean_s1 = re.sub(r'[,.;:!?\'"-]', '', sentence1.lower()).strip()
        
        # Kiểm tra xem câu này có phải là tập con của câu nào khác không
        for j, sentence2 in enumerate(sorted_sentences):
            if i == j:
                continue
                
            # Loại bỏ dấu câu khi so sánh
            clean_s2 = re.sub(r'[,.;:!?\'"-]', '', sentence2.lower()).strip()
            
            # Kiểm tra nếu là tập con nghiêm ngặt
            if clean_s1 in clean_s2:
                # Giảm ngưỡng xuống 0.6 (60%) để lọc nhiều hơn 
                if len(clean_s1) / len(clean_s2) < 0.6:
                    is_subset = True
                    break
        
        if not is_subset:
            filtered.append(sentence1)
    
    # Khôi phục lại thứ tự tương đối ban đầu
    result = []
    for s in sentences:
        if s in filtered and s not in result:
            result.append(s)
    
    return result

def to_sentences(passage, use_enhanced=True, use_spacy=True):
    # Tải mô hình transformer
    try:
        print("[INFO] Đang tải mô hình transformer")
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
        
    result_sentences = []
    
    try:
        # Bước 1: Đầu tiên tách bằng enhanced_sentence_detector
        if use_enhanced:
            initial_sentences = enhanced_sentence_detector(passage, aggressive=True)
            print(f"[INFO] Đã tách thành {len(initial_sentences)} câu với detector nâng cao")
        else:
            initial_sentences = sentence_detector(passage)
            print(f"[INFO] Đã tách thành {len(initial_sentences)} câu với detector cơ bản")
        
        # Cập nhật kết quả ban đầu
        result_sentences = initial_sentences
        
        # Bước 2: Tiếp tục tách chi tiết hơn bằng spaCy nếu được yêu cầu
        if use_spacy and nlp is not None:
            final_sentences = []
            
            # Xử lý từng câu đã được tách ở bước 1
            for sentence in initial_sentences:
                # Bao gồm cả câu ngắn (loại bỏ giới hạn từ)
                if not sentence.strip():
                    continue
                
                # Dùng spaCy để tách chi tiết hơn
                _, sub_sentences = spacy_sentence_splitter(sentence)
                final_sentences.extend(sub_sentences)
            
            # Loại bỏ trùng lặp nhưng giữ nguyên thứ tự xuất hiện ban đầu
            seen = set()
            result_sentences = []
            for s in final_sentences:
                if s.strip() and s not in seen:
                    seen.add(s)
                    result_sentences.append(s)
            
            print(f"[INFO] Đã tiếp tục tách thành {len(result_sentences)} câu con với spaCy")
    
    except Exception as e:
        print(f"[WARNING] Lỗi khi sử dụng phương pháp tách câu: {e}")
        print("[INFO] Sử dụng phương pháp cơ bản làm dự phòng")
        result_sentences = sentence_detector(passage)
    
    # Lọc bỏ các câu trống và câu dư thừa
    result_sentences = [s for s in result_sentences if s.strip()]
    result_sentences = filter_redundant_sentences(result_sentences)

    return result_sentences

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
    
    # Test hàm to_sentences mới
    print("\n=== Test with Combined Approach ===")
    combined_sentences = to_sentences(test_text, use_enhanced=True, use_spacy=True)
    
    print("\nCombined sentences:")
    for i, s in enumerate(combined_sentences):
        print(f"{i+1}. {s}")
    
    # Test với một ví dụ phức tạp hơn
    complex_text = "The equipment weighs little, can be installed on almost any flat surface, and doesn't depend on a supply of electricity. Instead it is powered by the process that produced it: the natural growth of plants. Hardy's company, Agripolis, has installed systems at 75 different Parisian sites, reducing the distance food travels to reach the city's markets and restaurants."
    
    print("\n=== Test Complex Example ===")
    complex_sentences = to_sentences(complex_text)
    
    print("\nCombined sentences (complex):")
    for i, s in enumerate(complex_sentences):
        print(f"{i+1}. {s}")