import os
import sys
import argparse
from typing import List, Dict, Any, Tuple

# Reconfigure stdout to use UTF-8 for printing special characters (like mathematical symbols) in Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Ensure the parent directory is in the path so we can import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from arthrilens.embedder import GeminiEmbedder
from arthrilens.vector_store import SimpleVectorStore
from arthrilens.config import EMBEDDING_MODEL

class Retriever:
    """
    Retrieves the most relevant document chunks for a given query.
    """
    def __init__(self, vector_store_path: str, model_name: str = EMBEDDING_MODEL):
        self.vector_store_path = vector_store_path
        self.embedder = GeminiEmbedder(model_name=model_name)
        
        if not os.path.exists(vector_store_path):
            raise FileNotFoundError(
                f"Vector store file not found at: {vector_store_path}. "
                f"Please run the ingestion pipeline first."
            )
            
        self.vector_store = SimpleVectorStore.load(vector_store_path)

    def retrieve(self, query: str, k: int = 5) -> List[Tuple[Dict[str, Any], float]]:
        """
        Embeds the query and searches the vector store.
        
        Args:
            query: The user search query.
            k: Top-K chunks to retrieve.
            
        Returns:
            List of tuples (chunk_dict, similarity_score)
        """
        print(f"Embedding query: '{query}'...")
        query_embedding = self.embedder.embed_query(query)
        
        # 2. Retrieve chunks from vector store
        print(f"Searching vector store for top-{k} relevant chunks...")
        results = self.vector_store.search(query_embedding, query_text=query, k=k)
        
        return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Query the RAG vector database.")
    parser.add_argument("query", type=str, help="Search query")
    parser.add_argument("-k", type=int, default=3, help="Number of chunks to retrieve")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    vector_store_path = os.path.join(script_dir, "data", "vector_store.pkl")

    try:
        retriever = Retriever(vector_store_path=vector_store_path)
        results = retriever.retrieve(args.query, k=args.k)
        
        print("\n" + "=" * 60)
        print(f"Query Results for: '{args.query}'")
        print("=" * 60)
        
        for idx, (doc, score) in enumerate(results):
            meta = doc["metadata"]
            file_name = meta.get("file_name", "Unknown")
            page_info = f" | Page {meta.get('page')}" if "page" in meta else ""
            row_info = f" | Sheet '{meta.get('sheet')}' Row {meta.get('row')}" if "row" in meta else ""
            
            print(f"\n[{idx + 1}] Similarity Score: {score:.4f}")
            print(f"Source: {file_name}{page_info}{row_info}")
            print("-" * 40)
            print(doc["text"])
            print("-" * 40)
            
        print("\n" + "=" * 60)
        
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        print("Hint: Run the ingestion pipeline script first using: python arthrilens/pipeline.py")
    except Exception as e:
        print(f"[ERROR] An error occurred: {e}")
