import pandas as pd
from Tool.Database import connect_to_db
import json
import numpy as np
from dotenv import load_dotenv
import os
from Tool.Sentence_Detector import sentence_detector
from Tool.Sentence_Embedding import sentence_embedding
load_dotenv()

doc = pd.read_csv('passages_1000.csv')
passage = doc['passage_text'].tolist()

def to_sentences(passage):
    sentences = []
    for text in passage:
        sentences.append(sentence_detector(text))
    return sentences

def to_vectors(sentences):
    vectors = []
    for sentence in sentences:
        vectors.append(sentence_embedding(sentence))
    return vectors

