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

def filter_subset_sentences(sentences):
    """Lọc bỏ câu là tập con nghiêm ngặt của câu khác"""
    filtered = []
    
    # Sắp xếp theo độ dài giảm dần
    sorted_sentences = sorted(sentences, key=len, reverse=True)
    
    for i, sentence1 in enumerate(sorted_sentences):
        is_subset = False
        clean_s1 = re.sub(r'[,.;:!?\'"-]', '', sentence1.lower()).strip()
        
        for j, sentence2 in enumerate(sorted_sentences):
            if i == j:
                continue
            
            clean_s2 = re.sub(r'[,.;:!?\'"-]', '', sentence2.lower()).strip()
            
            # Kiểm tra là tập con và độ dài chênh lệch > 30%
            if clean_s1 in clean_s2 and len(clean_s1) / len(clean_s2) < 0.7:
                is_subset = True
                break
        
        if not is_subset:
            filtered.append(sentence1)
    
    return filtered

def filter_sentences_for_oie(sentences, min_words=3):
    """Lọc các câu phù hợp cho việc trích xuất OIE"""
    filtered = []
    
    for sentence in sentences:
        # Loại bỏ câu quá ngắn
        if len(sentence.split()) < min_words:
            continue
            
        # Loại bỏ các fragment không hoàn chỉnh
        if re.match(r'^(and|or|but|because|if|when|while|since|although|though)\s', sentence.lower()):
            # Fragment bắt đầu bằng liên từ
            continue
            
        # Kiểm tra cấu trúc chủ ngữ-vị ngữ (câu nên có ít nhất 1 động từ)
        words = sentence.lower().split()
        if not any(verb in words for verb in ['is', 'are', 'was', 'were', 'am', 'be', 'been', 'being',
                                           'have', 'has', 'had', 'do', 'does', 'did', 
                                           'can', 'could', 'will', 'would', 'may', 'might', 'shall', 'should',
                                           'must']):
            # Thử kiểm tra động từ thông thường với 'ed', 'ing' kết thúc
            if not any(w.endswith(('ed', 'ing')) for w in words):
                continue
        
        # Loại bỏ câu quá dài (có thể khó trích xuất OIE chính xác)
        if len(sentence.split()) > 40:
            # Có thể bổ sung phương pháp tách câu phức tạp
            pass
            
        filtered.append(sentence)
    
    # Lọc bỏ các câu là tập con của nhau
    return filter_subset_sentences(filtered)

def to_sentences(passage, use_enhanced=True, use_spacy=True, min_words=3):
    """
    Tách đoạn văn thành các câu tối ưu cho OIE
    
    Args:
        passage: Đoạn văn bản cần tách
        use_enhanced: Sử dụng phương pháp tách câu nâng cao
        use_spacy: Sử dụng spaCy cho việc tách câu chi tiết
        min_words: Số từ tối thiểu trong một câu
        
    Returns:
        list: Danh sách các câu đã được tối ưu cho OIE
    """
    result_sentences = []
    
    try:
        # Bước 1: Tiền xử lý đoạn văn - gộp dòng và chuẩn hóa dấu câu
        passage = re.sub(r'\s+', ' ', passage)
        passage = re.sub(r'\.{2,}', '.', passage)  # Thay ... bằng .
        
        # Bảo vệ từ ghép có gạch nối
        protected_text, replacements = protect_hyphenated_words(passage)
        
        # Bước 2: Tách câu cơ bản (giữ nguyên câu dài)
        if use_enhanced:
            initial_sentences = enhanced_sentence_detector(protected_text, aggressive=False)
            print(f"[INFO] Đã tách thành {len(initial_sentences)} câu với detector nâng cao")
        else:
            initial_sentences = sentence_detector(protected_text)
            print(f"[INFO] Đã tách thành {len(initial_sentences)} câu với detector cơ bản")
            
        # Khôi phục từ ghép
        initial_sentences = [restore_hyphenated_words(s, replacements) for s in initial_sentences]
        
        # Bước 3: Phát hiện và kết hợp các câu fragment
        combined_sentences = combine_sentence_fragments(initial_sentences)
        print(f"[INFO] Đã kết hợp thành {len(combined_sentences)} câu hoàn chỉnh")
        
        # Bước 4: Với mỗi câu dài, tách thành các mệnh đề có ý nghĩa
        refined_sentences = []
        for sentence in combined_sentences:
            # Với câu ngắn, giữ nguyên
            if len(sentence.split()) < 15:
                refined_sentences.append(sentence)
                continue
                
            # Với câu dài, sử dụng spaCy để tách có kiểm soát (nếu được yêu cầu)
            if use_spacy and nlp is not None:
                # Tách câu với spaCy, giữ cấu trúc ngữ pháp
                try:
                    sub_sentences = extract_meaningful_clauses(sentence)
                    # Thêm cả câu gốc và câu con
                    refined_sentences.append(sentence)  # Giữ câu gốc
                    refined_sentences.extend(sub_sentences)  # Thêm các mệnh đề con
                except Exception as e:
                    print(f"[WARNING] Lỗi khi tách mệnh đề: {e}")
                    refined_sentences.append(sentence)
            else:
                refined_sentences.append(sentence)
        
        # Bước 5: Lọc câu dư thừa và vô nghĩa
        filtered_sentences = filter_sentences_for_oie(refined_sentences, min_words)
        
        # Loại bỏ trùng lặp nhưng giữ nguyên thứ tự
        seen = set()
        result_sentences = []
        for s in filtered_sentences:
            if s not in seen:
                seen.add(s)
                result_sentences.append(s)
                
        print(f"[INFO] Kết quả cuối cùng: {len(result_sentences)} câu tối ưu cho OIE")
    
    except Exception as e:
        print(f"[WARNING] Lỗi khi tách câu: {e}")
        print("[INFO] Sử dụng phương pháp cơ bản làm dự phòng")
        result_sentences = sentence_detector(passage)
    
    return result_sentences

def combine_sentence_fragments(sentences):
    """Kết hợp các fragments thành câu hoàn chỉnh"""
    if not sentences:
        return []
        
    results = []
    current_sentence = sentences[0]
    
    for i in range(1, len(sentences)):
        current = current_sentence.strip()
        next_s = sentences[i].strip()
        
        # Kiểm tra fragment không hoàn chỉnh
        is_fragment = (
            # Kết thúc bởi dấu phẩy
            current.endswith(',') or
            # Không có động từ chính
            not any(word in current.lower() for word in ['is', 'are', 'was', 'were', 'have', 'has', 'had', 'do', 'does', 'did']) or
            # Fragment điển hình
            re.match(r'^(in|on|at|by|with|as|if|when|but)\s', next_s) or
            # Câu quá ngắn (< 5 từ) và kết thúc bằng dấu phẩy
            (len(current.split()) < 5 and current.endswith(','))
        )
        
        if is_fragment:
            # Kết hợp với câu tiếp theo
            current_sentence = f"{current} {next_s}"
        else:
            # Hoàn thành một câu, bắt đầu câu mới
            results.append(current_sentence)
            current_sentence = next_s
    
    # Thêm câu cuối
    if current_sentence:
        results.append(current_sentence)
        
    return results

def extract_meaningful_clauses(sentence):
    """Trích xuất các mệnh đề có ý nghĩa từ câu phức tạp"""
    doc = nlp(sentence)
    clauses = []
    
    # Tìm các mệnh đề chính và phụ
    for token in doc:
        # Tìm vị ngữ chính
        if token.dep_ in ('ROOT', 'ccomp', 'xcomp', 'advcl', 'relcl') and token.pos_ == 'VERB':
            # Xây dựng mệnh đề
            clause_tokens = list(token.subtree)
            
            # Sắp xếp theo vị trí trong câu
            clause_tokens.sort(key=lambda t: t.i)
            
            # Tạo chuỗi mệnh đề
            clause_text = ' '.join(t.text for t in clause_tokens)
            
            # Kiểm tra xem mệnh đề có ý nghĩa không (đủ dài)
            if len(clause_text.split()) >= 4:
                clauses.append(clause_text)
    
    # Tìm cụm danh ngữ quan trọng (có thể chứa thông tin)
    for chunk in doc.noun_chunks:
        if len(chunk.text.split()) >= 4 and any(token.pos_ == 'VERB' for token in chunk.root.children):
            if chunk.text not in clauses:
                clauses.append(chunk.text)
                
    return clauses

def extract_triples_with_context(sentence, context=None):
    """
    Trích xuất các triple (subject, relation, object) từ một câu với ngữ cảnh
    
    Args:
        sentence: Câu cần trích xuất
        context: Ngữ cảnh bổ sung (mặc định là None)
        
    Returns:
        list: Danh sách các triple dạng {'subject': '...', 'relation': '...', 'object': '...'}
    """
    # Nếu không có spaCy, trả về danh sách rỗng
    if nlp is None:
        return []
    
    triples = []
    
    # Phân tích câu với spaCy
    doc = nlp(sentence)
    
    # Tìm các cấu trúc subject-verb-object
    for token in doc:
        # Tìm các động từ chính
        if token.pos_ == 'VERB':
            # Tìm chủ ngữ
            subjects = [subj for subj in token.children if subj.dep_ in ('nsubj', 'nsubjpass')]
            
            # Tìm tân ngữ
            objects = []
            for child in token.children:
                if child.dep_ in ('dobj', 'pobj', 'attr', 'acomp'):
                    objects.append(child)
                    # Mở rộng tân ngữ với các từ phụ thuộc
                    objects.extend([gc for gc in child.children if gc.dep_ in ('conj', 'appos')])
            
            # Nếu tìm thấy cả chủ ngữ và tân ngữ
            for subj in subjects:
                # Mở rộng chủ ngữ
                subj_span = [subj] + list(subj.subtree)
                subj_span = sorted(set(subj_span), key=lambda x: x.i)
                subject_text = ' '.join(t.text for t in subj_span if t.dep_ not in ('punct', 'cc'))
                
                # Xây dựng quan hệ (vị ngữ)
                relation_tokens = [token]
                for child in token.children:
                    if child.dep_ in ('aux', 'auxpass', 'neg', 'prep', 'prt', 'advmod') and child.i > token.i:
                        relation_tokens.append(child)
                relation_tokens = sorted(relation_tokens, key=lambda x: x.i)
                relation_text = ' '.join(t.text for t in relation_tokens)
                
                # Xử lý mỗi tân ngữ
                for obj in objects:
                    # Mở rộng tân ngữ
                    obj_span = [obj] + list(obj.subtree)
                    obj_span = sorted(set(obj_span), key=lambda x: x.i)
                    object_text = ' '.join(t.text for t in obj_span if t.dep_ not in ('punct', 'cc'))
                    
                    # Tạo và thêm triple
                    triple = {
                        'subject': subject_text.strip(),
                        'relation': relation_text.strip(),
                        'object': object_text.strip()
                    }
                    
                    triples.append(triple)
    
    return triples

def process_oie_in_groups(sentences, groups):
    """Trích xuất OIE cho các câu trong nhóm, tận dụng ngữ cảnh chung"""
    all_triples = []
    sentence_triples = [[] for _ in range(len(sentences))]
    
    # Xử lý theo nhóm
    for group_idx, group in enumerate(groups):
        print(f"  Đang trích xuất quan hệ cho nhóm {group_idx+1}/{len(groups)}...")
        
        # Tạo ngữ cảnh chung cho nhóm
        group_context = " ".join([sentences[idx] for idx in group])
        
        # Trích xuất triples cho từng câu với ngữ cảnh tốt hơn
        for i, sentence_idx in enumerate(group):
            sentence = sentences[sentence_idx]
            
            # Thêm tham số ngữ cảnh khi gọi hàm trích xuất
            triples = extract_triples_with_context(sentence, group_context)
            
            # Lọc bỏ triples chất lượng thấp
            filtered_triples = filter_low_quality_triples(triples)
            
            # Lưu trữ triples
            sentence_triples[sentence_idx] = filtered_triples
            all_triples.extend(filtered_triples)
    
    return all_triples, sentence_triples

def clause_has_svo(clause_tokens):
    """
    Kiểm tra xem một mệnh đề có cấu trúc Chủ ngữ-Động từ-Tân ngữ hay không
    
    Args:
        clause_tokens: Danh sách các token trong mệnh đề
        
    Returns:
        bool: True nếu có cấu trúc SVO, False nếu không
    """
    has_subject = any(token.dep_ in ('nsubj', 'nsubjpass') for token in clause_tokens)
    has_verb = any(token.pos_ == 'VERB' for token in clause_tokens)
    has_object = any(token.dep_ in ('dobj', 'pobj', 'attr', 'acomp') for token in clause_tokens)
    
    return has_subject and has_verb

def filter_low_quality_triples(triples):
    """Lọc bỏ OIE triples chất lượng thấp"""
    filtered = []
    
    for triple in triples:
        # Kiểm tra subject
        if not triple['subject'] or len(triple['subject'].split()) > 8:
            continue
            
        # Kiểm tra relation
        if not triple['relation'] or any(bad in triple['relation'] for bad in ['can can', 'are are', 'of are']):
            continue
            
        # Kiểm tra object
        if not triple['object']:
            continue
            
        # Thêm triple hợp lệ
        filtered.append(triple)
    
    return filtered

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
        
    print("\n=== NHẬP TÀI LIỆU TRỰC TIẾP ===")
    print("Nhập nội dung tài liệu (kết thúc bằng một dòng chỉ có '###'):")
    lines = []
    while True:
        line = input()
        if line == '###':
            break
        lines.append(line)
        
    if not lines:
        print("Không có nội dung được nhập, hủy phân tích.")
        exit()
        
    document = "\n".join(lines)
    
    # Test với một ví dụ phức tạp hơn
    complex_text = document
    
    print("\n=== Test Complex Example ===")
    complex_sentences = to_sentences(complex_text)
    
    print("\nCombined sentences (complex):")
    for i, s in enumerate(complex_sentences):
        print(f"{i+1}. {s}")