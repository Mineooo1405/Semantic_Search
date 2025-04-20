import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

user = "postgres"
password = 140504
port = 5432
name_dbname = "semantic_search_db"

def create_database_and_tables():
    # Connection parameters - adjust these as needed
    db_params = {
        'initial_dbname': 'postgres',  
        'user': {user},
        'password': {password},             
        'host': 'localhost',
        'port': {port},
        'new_dbname': {name_dbname}
    }
    
    # Connect to default database
    conn = psycopg2.connect(
        dbname=db_params['initial_dbname'],
        user=db_params['user'],
        password=db_params['password'],
        host=db_params['host'],
        port=db_params['port']
    )
    
    conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
    cursor = conn.cursor()
    
    # Create new database
    try:
        cursor.execute(f"CREATE DATABASE {db_params['new_dbname']}")
        print(f"Database '{db_params['new_dbname']}' created successfully.")
    except psycopg2.errors.DuplicateDatabase:
        print(f"Database '{db_params['new_dbname']}' already exists.")
    finally:
        cursor.close()
        conn.close()
    
    # Connect to the new database
    conn = psycopg2.connect(
        dbname=db_params['new_dbname'],
        user=db_params['user'],
        password=db_params['password'],
        host=db_params['host'],
        port=db_params['port']
    )
    
    cursor = conn.cursor()
    
    # Create Semantic_Grouping table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS Semantic_Grouping (
        id SERIAL PRIMARY KEY,
        Original_Paragraph TEXT NOT NULL,
        Sentences JSONB NOT NULL,  -- List of sentences
        Embedding_Vectors JSONB NOT NULL,  -- List of embedding vectors
        Semantic_Chunking JSONB NOT NULL  -- List of chunks (each chunk is a list of related sentences)
    );
    """)
    
    # Create Semantic_Splitter table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS Semantic_Splitter (
        id SERIAL PRIMARY KEY,
        Original_Paragraph TEXT NOT NULL,
        Sentences JSONB NOT NULL,  -- List of sentences
        Embedding_Vectors JSONB NOT NULL,  -- List of embedding vectors
        Semantic_Chunking JSONB NOT NULL  -- List of chunks (each chunk is a list of related sentences)
    );
    """)
    
    # Create Text_Splitter table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS Text_Splitter (
        id SERIAL PRIMARY KEY,
        Original_Paragraph TEXT NOT NULL,
        Sentences JSONB NOT NULL,  -- List of sentences
        Chunking JSONB NOT NULL  -- List of chunks based on predefined length
    );
    """)
    
    conn.commit()
    print("Tables created successfully.")
    
    cursor.close()
    conn.close()

def connect_to_db():
    """Connect to the PostgreSQL database."""
    conn = psycopg2.connect(
        dbname="semantic_search_db",
        user={user},
        password={password},  # Change this
        host="localhost",
        port={port}
    )
    return conn

if __name__ == "__main__":
    create_database_and_tables()