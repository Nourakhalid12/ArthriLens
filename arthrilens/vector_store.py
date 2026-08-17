import os
import pickle
from typing import List, Dict, Any, Tuple
import numpy as np

class SimpleVectorStore:
    """
    A lightweight, robust local vector store that uses NumPy for cosine similarity search
    and pickles documents, metadata, and embeddings to disk.
    
    This avoids complex database setups and runs instantly for datasets with thousands of items.
    """
    def __init__(self):
        self.documents: List[Dict[str, Any]] = []
        self.embeddings: np.ndarray = None

    def add_documents(self, documents: List[Dict[str, Any]], embeddings: np.ndarray):
        """
        Adds documents and their corresponding pre-computed embeddings.
        
        Args:
            documents: List of dicts, each with 'text' and 'metadata'.
            embeddings: Numpy array of shape (num_docs, embedding_dim).
        """
        if len(documents) != len(embeddings):
            raise ValueError("The number of documents and embeddings must match.")
            
        if len(documents) == 0:
            return

        self.documents.extend(documents)
        
        if self.embeddings is None:
            self.embeddings = np.array(embeddings)
        else:
            self.embeddings = np.vstack([self.embeddings, embeddings])

    def search(self, query_embedding: np.ndarray, k: int = 5) -> List[Tuple[Dict[str, Any], float]]:
        """
        Performs a cosine similarity search for the top-k documents.
        
        Args:
            query_embedding: Numpy array of shape (embedding_dim,).
            k: Number of top documents to retrieve.
            
        Returns:
            List of tuples (doc, similarity_score) sorted in descending order of similarity.
        """
        if self.embeddings is None or len(self.documents) == 0:
            return []

        # Ensure query embedding is 1D
        query_embedding = query_embedding.flatten()
        
        # Calculate dot products
        dot_products = np.dot(self.embeddings, query_embedding)
        
        # Calculate magnitudes (norms)
        doc_norms = np.linalg.norm(self.embeddings, axis=1)
        query_norm = np.linalg.norm(query_embedding)
        
        # Calculate cosine similarity (add epsilon to prevent division by zero)
        similarities = dot_products / (doc_norms * query_norm + 1e-9)
        
        # Get the top-k indices
        top_k_indices = np.argsort(similarities)[::-1][:k]
        
        results = []
        for idx in top_k_indices:
            results.append((self.documents[idx], float(similarities[idx])))
            
        return results

    def save(self, file_path: str):
        """
        Saves the vector store to disk.
        """
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
        data = {
            "documents": self.documents,
            "embeddings": self.embeddings
        }
        with open(file_path, "wb") as f:
            pickle.dump(data, f)
        print(f"Vector store saved successfully to: {file_path}")

    @classmethod
    def load(cls, file_path: str) -> 'SimpleVectorStore':
        """
        Loads a vector store from disk.
        """
        store = cls()
        if not os.path.exists(file_path):
            print(f"No existing vector store file found at: {file_path}. Initializing new store.")
            return store
            
        try:
            with open(file_path, "rb") as f:
                data = pickle.load(f)
            store.documents = data.get("documents", [])
            store.embeddings = data.get("embeddings", None)
            print(f"Vector store loaded successfully from: {file_path} (containing {len(store.documents)} chunks)")
        except Exception as e:
            print(f"Error loading vector store from {file_path}: {e}. Initializing new store.")
            
        return store
