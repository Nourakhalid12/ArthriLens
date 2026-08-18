import os
import sys
import time
from typing import List, Dict, Any

# Ensure the parent directory is in the path so we can import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from arthrilens.parser import DocumentParser
from arthrilens.splitter import RecursiveCharacterTextSplitter
from arthrilens.embedder import GeminiEmbedder
from arthrilens.vector_store import SimpleVectorStore
from arthrilens.config import EMBEDDING_MODEL

def run_ingestion_pipeline(
    data_dir: str, 
    vector_store_path: str, 
    chunk_size: int = 1200, 
    chunk_overlap: int = 150,
    model_name: str = EMBEDDING_MODEL
):
    print("=" * 60)
    print("Starting ArthriLens RAG Ingestion Pipeline...")
    print("=" * 60)
    
    start_time = time.time()
    
    # 1. Initialize modules
    print("\n[1/5] Initializing RAG Components...")
    parser = DocumentParser()
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    embedder = GeminiEmbedder(model_name=model_name)
    vector_store = SimpleVectorStore()
    
    # 2. Scan and parse files
    print(f"\n[2/5] Scanning and parsing files in: {data_dir}")
    if not os.path.exists(data_dir):
        print(f"[ERROR] Data directory '{data_dir}' does not exist!")
        sys.exit(1)
        
    parsed_docs = parser.parse_directory(data_dir)
    print(f"Parsed {len(parsed_docs)} document files/pages/rows.")
    
    if not parsed_docs:
        print("[WARN] No documents parsed. Exiting.")
        return
        
    # 3. Split documents into chunks
    print("\n[3/5] Splitting documents into overlapping chunks...")
    chunks = splitter.split_documents(parsed_docs)
    print(f"Created {len(chunks)} chunks (size={chunk_size}, overlap={chunk_overlap}).")
    
    # 4. Generate embeddings
    print(f"\n[4/5] Generating embeddings using local model '{model_name}'...")
    print("Please wait, loading model and processing embeddings...")
    chunk_texts = [chunk["text"] for chunk in chunks]
    
    embed_start = time.time()
    embeddings = embedder.embed_documents(chunk_texts)
    embed_duration = time.time() - embed_start
    
    print(f"[OK] Generated {len(embeddings)} embeddings of dimension {embeddings.shape[1]}.")
    print(f"Embedding generation took {embed_duration:.2f} seconds.")
    
    # 5. Store embeddings
    print(f"\n[5/5] Saving to local Vector Store...")
    vector_store.add_documents(chunks, embeddings)
    vector_store.save(vector_store_path)
    
    total_duration = time.time() - start_time
    print("\n" + "=" * 60)
    print("Ingestion Pipeline completed successfully!")
    print(f"Vector Store saved to: {vector_store_path}")
    print(f"Total time elapsed: {total_duration:.2f} seconds")
    print("=" * 60)

if __name__ == "__main__":
    # Get paths relative to this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_data_dir = os.path.join(script_dir, "data")
    default_vector_store_path = os.path.join(default_data_dir, "vector_store.pkl")
    
    run_ingestion_pipeline(
        data_dir=default_data_dir,
        vector_store_path=default_vector_store_path
    )
