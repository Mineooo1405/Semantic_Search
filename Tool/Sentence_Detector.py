from nltk.tokenize import sent_tokenize
import nltk

def sentence_detector(text):
    
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
 
    return sent_tokenize(text)