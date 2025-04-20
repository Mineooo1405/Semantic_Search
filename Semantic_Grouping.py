import pandas as pd
from Tool.Database import connect_to_db
import json
import numpy as np
from dotenv import load_dotenv
import os

load_dotenv()
doc = pd.read_csv('ms_marco_1000.csv')
passage = doc['passage'].tolist()

def insert_semantic_grouping(original_text, sentences, embedding_vectors, semantic_chunks):
    """Insert data into Semantic_Grouping table."""
    conn = connect_to_db()
    cursor = conn.cursor()
    
    # Convert numpy arrays to lists for JSON serialization
    embedding_vectors_list = [vector.tolist() for vector in embedding_vectors]
    
    cursor.execute(
        """
        INSERT INTO Semantic_Grouping 
        (Original_Paragraph, Sentences, Embedding_Vectors, Semantic_Chunking)
        VALUES (%s, %s, %s, %s)
        RETURNING id
        """,
        (
            original_text,
            json.dumps(sentences),
            json.dumps(embedding_vectors_list),
            json.dumps(semantic_chunks)
        )
    )