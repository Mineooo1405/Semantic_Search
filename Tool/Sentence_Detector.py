from nltk.tokenize import sent_tokenize
import nltk
text = "Hello world. This is a test. How are you?"
def sentence_detector(text):
    
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
 
    return sent_tokenize(text)

#print(sentence_detector(text))