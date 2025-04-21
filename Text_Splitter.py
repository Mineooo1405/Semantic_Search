import re
from typing import List, Dict, Union, Optional, Callable
import nltk
from nltk.tokenize import sent_tokenize

# Download NLTK data for sentence tokenization if not already present
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

class TextSplitter:
    """
    Công cụ chia nhỏ văn bản thành các đoạn nhỏ (chunks) dựa trên độ dài tối đa.
    Hỗ trợ nhiều chiến lược phân chia văn bản khác nhau.
    """
    
    def __init__(self, 
                 chunk_size: int = 1000, 
                 chunk_overlap: int = 200,
                 separator: str = "\n",
                 keep_separator: bool = False,
                 length_function: Callable = len):
        """
        Khởi tạo TextSplitter
        
        Args:
            chunk_size: Độ dài tối đa của mỗi chunk (ký tự)
            chunk_overlap: Độ dài phần chồng lấp giữa các chunk liên tiếp
            separator: Ký tự phân cách sử dụng khi tìm vị trí cắt hợp lý
            keep_separator: Giữ lại separator ở đầu chunk mới hay không
            length_function: Hàm dùng để đo độ dài của văn bản
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separator = separator
        self.keep_separator = keep_separator
        self.length_function = length_function
    
    def split_text(self, text: str) -> List[str]:
        """
        Chia văn bản thành các chunk dựa trên separator và độ dài tối đa
        
        Args:
            text: Văn bản cần chia
            
        Returns:
            List[str]: Danh sách các chunk văn bản
        """
        # Nếu văn bản đủ nhỏ, trả về luôn
        if self.length_function(text) <= self.chunk_size:
            return [text]
        
        # Chia văn bản thành các phần nhỏ theo separator
        splits = self._split_by_separator(text, self.separator)
        
        # Gộp các phần nhỏ thành các chunk có độ dài phù hợp
        return self._merge_splits(splits)
    
    def _split_by_separator(self, text: str, separator: str) -> List[str]:
        """
        Chia văn bản thành các phần nhỏ dựa trên separator
        
        Args:
            text: Văn bản cần chia
            separator: Ký tự hoặc chuỗi phân cách
            
        Returns:
            List[str]: Danh sách các phần văn bản sau khi chia
        """
        # Sử dụng regex để chia khi separator có thể là multi-character
        if separator:
            # Nếu không giữ separator, không cần thêm vào kết quả
            if not self.keep_separator:
                return re.split(f"({re.escape(separator)})", text)
            # Nếu giữ separator, thêm vào kết quả
            else:
                parts = []
                splits = re.split(f"({re.escape(separator)})", text)
                # Ghép separator vào phần sau nó
                for i in range(0, len(splits) - 1, 2):
                    if i + 1 < len(splits):
                        parts.append(splits[i] + splits[i + 1])
                    else:
                        parts.append(splits[i])
                if len(splits) % 2 == 1:
                    parts.append(splits[-1])
                return parts
        
        # Nếu không có separator, trả về text nguyên bản
        return [text]
    
    def _merge_splits(self, splits: List[str]) -> List[str]:
        """
        Gộp các phần nhỏ thành các chunk có độ dài phù hợp
        
        Args:
            splits: Danh sách các phần văn bản sau khi chia
            
        Returns:
            List[str]: Danh sách các chunk có độ dài phù hợp
        """
        chunks = []
        current_chunk = []
        current_length = 0
        
        for split in splits:
            split_length = self.length_function(split)
            
            # Nếu phần hiện tại quá lớn, chia nhỏ hơn nữa
            if split_length > self.chunk_size:
                # Xử lý chunk hiện tại trước
                if current_chunk:
                    chunks.append("".join(current_chunk))
                    current_chunk = []
                    current_length = 0
                
                # Chia nhỏ phần lớn thành các đoạn nhỏ hơn
                local_chunks = []
                for i in range(0, len(split), self.chunk_size - self.chunk_overlap):
                    end = min(i + self.chunk_size, len(split))
                    local_chunks.append(split[i:end])
                    if end == len(split):
                        break
                
                # Thêm tất cả trừ đoạn cuối vào chunks
                chunks.extend(local_chunks[:-1])
                
                # Thêm đoạn cuối vào chunk hiện tại
                if local_chunks:
                    current_chunk = [local_chunks[-1]]
                    current_length = self.length_function(local_chunks[-1])
            
            # Nếu thêm phần hiện tại vẫn nằm trong giới hạn
            elif current_length + split_length <= self.chunk_size:
                current_chunk.append(split)
                current_length += split_length
            
            # Nếu thêm vào sẽ vượt quá giới hạn
            else:
                chunks.append("".join(current_chunk))
                current_chunk = [split]
                current_length = split_length
        
        # Xử lý chunk cuối cùng nếu có
        if current_chunk:
            chunks.append("".join(current_chunk))
        
        return chunks


class CharacterTextSplitter(TextSplitter):
    """Chia văn bản theo số ký tự tối đa"""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        super().__init__(chunk_size, chunk_overlap, separator="", keep_separator=False)
    
    def split_text(self, text: str) -> List[str]:
        """
        Chia văn bản thành các đoạn có độ dài tối đa chunk_size,
        chồng lấp chunk_overlap ký tự
        """
        chunks = []
        
        if len(text) <= self.chunk_size:
            return [text]
        
        for i in range(0, len(text), self.chunk_size - self.chunk_overlap):
            end = min(i + self.chunk_size, len(text))
            chunks.append(text[i:end])
            if end == len(text):
                break
        
        return chunks


class SentenceTextSplitter(TextSplitter):
    """Chia văn bản theo câu, đảm bảo không cắt giữa câu"""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        super().__init__(chunk_size, chunk_overlap, separator="", keep_separator=False)
    
    def split_text(self, text: str) -> List[str]:
        """Chia văn bản thành các đoạn, tôn trọng ranh giới câu"""
        # Tách thành các câu
        sentences = sent_tokenize(text)
        
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence_length = len(sentence)
            
            # Nếu một câu dài hơn chunk_size, chia nhỏ câu đó
            if sentence_length > self.chunk_size:
                # Thêm chunk hiện tại vào kết quả
                if current_chunk:
                    chunks.append(" ".join(current_chunk))
                    current_chunk = []
                    current_length = 0
                
                # Chia câu dài thành các đoạn nhỏ hơn
                for i in range(0, len(sentence), self.chunk_size - self.chunk_overlap):
                    end = min(i + self.chunk_size, len(sentence))
                    chunks.append(sentence[i:end])
                    if end == len(sentence):
                        break
            
            # Nếu thêm câu hiện tại vẫn nằm trong giới hạn
            elif current_length + sentence_length + 1 <= self.chunk_size:  # +1 cho khoảng trắng
                current_chunk.append(sentence)
                current_length += sentence_length + 1  # +1 cho khoảng trắng
            
            # Nếu thêm vào sẽ vượt quá giới hạn
            else:
                chunks.append(" ".join(current_chunk))
                current_chunk = [sentence]
                current_length = sentence_length
        
        # Thêm chunk cuối cùng
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        return chunks


class ParagraphTextSplitter(TextSplitter):
    """Chia văn bản theo đoạn, giữ nguyên các đoạn văn"""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        super().__init__(chunk_size, chunk_overlap, separator="\n\n", keep_separator=True)


class RecursiveTextSplitter(TextSplitter):
    """
    Chia văn bản thành các đoạn nhỏ sử dụng chiến lược đệ quy.
    Thử chia theo thứ tự ưu tiên: đoạn, câu, từ, ký tự.
    """
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        super().__init__(chunk_size, chunk_overlap)
        # Danh sách các separator theo thứ tự ưu tiên
        self.separators = ["\n\n", "\n", ". ", ", ", " ", ""]
    
    def split_text(self, text: str) -> List[str]:
        """Chia văn bản đệ quy sử dụng các separator khác nhau"""
        # Trả về text nếu đủ nhỏ
        if len(text) <= self.chunk_size:
            return [text]
        
        # Thử từng separator theo thứ tự ưu tiên
        for separator in self.separators:
            # Cấu hình TextSplitter với separator hiện tại
            splitter = TextSplitter(
                chunk_size=self.chunk_size, 
                chunk_overlap=self.chunk_overlap,
                separator=separator,
                keep_separator=(separator != "")
            )
            
            # Thử chia văn bản
            chunks = splitter.split_text(text)
            
            # Nếu đã chia thành nhiều chunk, trả về kết quả
            if len(chunks) > 1:
                return chunks
        
        # Nếu không chia được bằng bất kỳ separator nào, 
        # trả về text gốc (trường hợp này hiếm xảy ra)
        return [text]


def main():
    """Hàm chính để demo việc phân chia văn bản"""
    sample_text = """
    TextSplitter là một công cụ quan trọng trong xử lý ngôn ngữ tự nhiên. Nó giúp chia văn bản lớn thành các đoạn nhỏ hơn, phù hợp với giới hạn đầu vào của các mô hình ngôn ngữ lớn như GPT hay BERT. Việc chia văn bản đúng cách có thể giúp tăng hiệu suất của các ứng dụng NLP.

    Có nhiều chiến lược khác nhau để chia văn bản:
    1. Chia theo số ký tự cố định
    2. Chia theo ranh giới câu
    3. Chia theo đoạn văn
    4. Chia đệ quy theo nhiều separator

    Mỗi phương pháp đều có ưu điểm và hạn chế riêng, tùy thuộc vào nhu cầu cụ thể của ứng dụng.
    
    Lớp TextSplitter cơ bản cung cấp các chức năng cần thiết để phân chia văn bản và hỗ trợ các tùy chọn như kích thước chunk, độ chồng lấp, và separator tùy chỉnh. Các lớp con của TextSplitter triển khai các phương pháp cụ thể để phân chia văn bản theo các chiến lược khác nhau.
    """
    
    print("=== Chia văn bản theo ký tự ===")
    char_splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=50)
    char_chunks = char_splitter.split_text(sample_text)
    for i, chunk in enumerate(char_chunks):
        print(f"Chunk {i+1} ({len(chunk)} ký tự): {chunk[:50]}...")
    
    print("\n=== Chia văn bản theo câu ===")
    sentence_splitter = SentenceTextSplitter(chunk_size=200, chunk_overlap=50)
    sentence_chunks = sentence_splitter.split_text(sample_text)
    for i, chunk in enumerate(sentence_chunks):
        print(f"Chunk {i+1} ({len(chunk)} ký tự): {chunk[:50]}...")
    
    print("\n=== Chia văn bản theo đoạn ===")
    paragraph_splitter = ParagraphTextSplitter(chunk_size=200, chunk_overlap=50)
    paragraph_chunks = paragraph_splitter.split_text(sample_text)
    for i, chunk in enumerate(paragraph_chunks):
        print(f"Chunk {i+1} ({len(chunk)} ký tự): {chunk[:50]}...")
    
    print("\n=== Chia văn bản đệ quy ===")
    recursive_splitter = RecursiveTextSplitter(chunk_size=200, chunk_overlap=50)
    recursive_chunks = recursive_splitter.split_text(sample_text)
    for i, chunk in enumerate(recursive_chunks):
        print(f"Chunk {i+1} ({len(chunk)} ký tự): {chunk[:50]}...")


if __name__ == "__main__":
    main()